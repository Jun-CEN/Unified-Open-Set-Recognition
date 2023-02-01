from os.path import join as pjoin  # pylint: disable=g-importing-member
import time

import sys
import os
# MARK: For conflict between terminal and Pycharm
project='big_transfer'
sys.path.append(os.getcwd().split(project)[0]+project)

import numpy as np
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F


import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
# Mark: 1st difference
import bit_hyperrule
import copy
from sklearn import metrics

from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset

np.random.seed(0)
torch.cuda.manual_seed(0)

def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]



def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
    ood_precrop, ood_crop = bit_hyperrule.get_resolution_from_dataset(args.ood_dataset)
    train_tx = tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.RandomCrop((crop, crop)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((crop, crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ood_val_tx = tv.transforms.Compose([
        tv.transforms.Resize((ood_crop, ood_crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if args.dataset == "cifar10":
        train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
        valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
    elif args.dataset == "cifar100":
        train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
        valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
    elif args.dataset == "imagenet2012":
        train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
        valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
    else:
        raise ValueError(f"Sorry, we have not spent time implementing the "
                         f"{args.dataset} dataset in the PyTorch codebase. "
                         f"In principle, it should be easy to add :)")

    # Mark: Try 2 kinds of data @1.original size @2. resize_version -> upsample -> run
    if args.ood_dataset == 'TinyImagenet':
        ood_set = tv.datasets.ImageFolder(root=args.datadir + '/tiny-imagenet-200/test', transform=ood_val_tx)
    elif args.ood_dataset == 'TinyImagenet_resize':
        ood_set = tv.datasets.ImageFolder(root=args.datadir + '/TinyImagenet_resize', transform=ood_val_tx)
    elif args.ood_dataset == 'LSUN':
        ood_set = tv.datasets.ImageFolder(root=args.datadir + '/LSUN_resize', transform=ood_val_tx)


    if args.examples_per_class is not None:
        logger.info(f"Looking for {args.examples_per_class} imag es per class...")
        indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
        train_set = torch.utils.data.Subset(train_set, indices=indices)

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch // args.batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=micro_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    ood_loader = torch.utils.data.DataLoader(dataset=ood_set,
                                             batch_size=micro_batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.workers)

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader, ood_set, ood_loader


def run_eval(model, data_loader, args, logger):
    # switch to evaluate mode
    model.eval()

    out = []
    classifier = []
    xent = nn.CrossEntropyLoss()

    mode = args.process

    for data in data_loader:
        if type(data) == list:
            images, labels = data
        else:
            images = data
        images = Variable(images, requires_grad=True).cuda()
        images.retain_grad()

        # Select out wrongly classified case
        classified_results = model(images)
        classified_results = classified_results[0] if isinstance(classified_results, tuple) else classified_results
        _, classified_idx = torch.max(classified_results, 1)
        classifier.append(classified_idx.cpu().numpy())

        if mode == 'LC':
            _, confidence = model(images)
            confidence = torch.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'TCP':
            pred, _ = model(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

        elif mode == 'BCE':
            pred, _ = model(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

        elif mode == 'DOCTOR':
            pred, _ = model(images)
            pred = F.softmax(pred, dim=-1)
            sum_square = 1 - torch.sum(pred ** 2, dim=-1)
            uncertainties = sum_square / (1 - sum_square)
            confidence = 1 - uncertainties
            out.append(confidence.data.cpu().numpy())

        elif mode == 'ODIN':
            # https://arxiv.org/abs/1706.02690
            T = 1000
            epsilon = 0.001

            model.zero_grad()
            pred, _ = model(images)
            _, pred_idx = torch.max(pred.data, 1)
            labels = Variable(pred_idx)
            pred = pred / T
            loss = xent(pred, labels)
            loss.backward()

            images = images - epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            pred, _ = model(images)

            pred = pred / T
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)


        elif mode == 'EB':
            pred = model(images)
            pred = F.softmax(pred, dim=-1)[:,-1]
            pred = pred.data.cpu().numpy()
            out.append(1 - pred)

        else:
            pred = model(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

    out = np.concatenate(out)
    classifier = np.concatenate(classifier)
    return out, classifier




def main(args):
    logger = bit_common.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    if args.pretrained:
        args.name = 'PRETRAINED_' + args.process
        args.save_name = 'PRETRAINED_' + args.process + '_' + args.ood_dataset
    else:
        args.name = args.process
        args.save_name = args.process + '_' + args.ood_dataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Going to train on {device}")

    train_set, test_set, train_loader, test_loader, ood_set, ood_loader = mktrainval(args, logger)

    # Mark: Change model for different evaluation process
    if args.process == 'EB':
        model = models.KNOWN_MODELS[args.model](head_size=len(test_set.classes),
                                                zero_head=True,
                                                extra_class=True)
    elif args.process == 'VOS':
        model = models.KNOWN_MODELS[args.model](head_size=len(test_set.classes),
                                                zero_head=True,
                                                forward_virtual=True)
    elif args.process == 'LC' or args.process == 'TCP' or args.process == 'BCE' \
                              or args.process == 'DOCTOR' or args.process == 'ODIN':
        model = models.KNOWN_MODELS[args.model](head_size=len(test_set.classes),
                                                zero_head=True,
                                                confidnet=True)
    else:
        model = models.KNOWN_MODELS[args.model](head_size=len(test_set.classes),
                                                zero_head=True)
    OE_list = ['OE', 'VOS', 'EB', 'MCD', 'ENERGY']
    if args.process not in OE_list:
        model = torch.nn.DataParallel(model)
    logger.info('#########################################################')
    logger.info(f"Evaluation OOD Detection Performance with {args.name}")
    ###################################################################
    # MARK: Load Different Pretrained Model According to different process
    # MARK: ODIN, DOCTOR shared same pth as LC
    # pretrained_model = torch.load(f'../log/{args.name}/best_bit.pth.tar')
    pretrained_model = torch.load(f'../log/{args.name}/bit.pth.tar')
    model.load_state_dict(pretrained_model["model"])
    # model.load_from(np.load(f"{args.model}.npz"))
    #######################################################################
    model = model.to(device)

    ind_scores, ind_classifier = run_eval(model, test_loader, args, logger)
    ind_labels = np.ones(ind_scores.shape[0])
    ood_scores, ood_classfier = run_eval(model, ood_loader, args, logger)
    ood_labels = np.zeros(ood_scores.shape[0])

    # prediction results
    ind_results = copy.deepcopy(ind_classifier)
    ood_results = copy.deepcopy(ood_classfier)

    ind_classfier = (ind_classifier == np.array(test_loader.dataset.targets))
    ood_classfier = (ood_classfier == ood_loader.dataset.targets)
    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])


    auroc = metrics.roc_auc_score(labels, scores)
    aupr_in = metrics.average_precision_score(labels, scores)
    aupr_out = metrics.average_precision_score(-1 * labels + 1, 1 - scores)

    ind_wrong_num = len(ind_classfier) - sum(ind_classfier)
    unified_labels = np.concatenate([ind_classfier*1.0, ood_labels])
    unified_auroc = metrics.roc_auc_score(unified_labels , scores)
    unified_aupr_in = metrics.average_precision_score(unified_labels , scores)
    unified_aupr_out = metrics.average_precision_score(-1 * unified_labels  + 1, 1 - scores)

    print("")
    print("Method: " + args.name)


    print("AUROC (higher is better): ", auroc)
    print("AUPR_IN (higher is better): ", aupr_in)
    print("AUPR_OUT (higher is better): ", aupr_out)

    print("Wrong numbers in InD class: ", ind_wrong_num)
    print("UNIFIED_AUROC (higher is better): ", unified_auroc)
    print("UNIFIED_AUPR_IN (higher is better): ", unified_aupr_in)
    print("UNIFIED_AUPR_OUT (higher is better): ", unified_aupr_out)

    # MARK: Sava test results into a folder and evaluate with different protocals
    np.savez(f'./test_results_saved/{args.save_name}.npz',
             ind_unctt = 1 - ind_scores,
             ood_unctt = 1 - ood_scores,
             ind_pred = ind_results,
             ood_pred = ood_results,
             ind_label = test_loader.dataset.targets,
             ood_label = ood_loader.dataset.targets)
    print('____________ TEST RESULTS SAVED ________________')
    print('*************************************************')
    # uosar, sp, inc_ood, inw_ood
    from selective_prediction.baselines_simple_u.experiments.uosar_hmdb_scratch import eval_uncertainty_methods as eval_usar
    from selective_prediction.baselines_simple_u.experiments.sp_hmdb_scratch import eval_uncertainty_methods as eval_sp
    from selective_prediction.baselines_simple_u.experiments.inc_ood_hmdb_scratch import eval_uncertainty_methods as eval_inc
    from selective_prediction.baselines_simple_u.experiments.inw_ood_hmdb_scratch import eval_uncertainty_methods as eval_inw

    result_path = f'./test_results_saved/{args.save_name}.npz'
    # result_path = f'test_results_saved/OLTR__ood__TinyImagenet_resize.npz'
    print('acc', 'aurc', 'eaurc', 'auroc', 'aupr', 'fpr95', 'ece')
    # results = np.load(result_path, allow_pickle=True)
    acc, aurc, eaurc, auroc, aupr, fpr95, ece = eval_sp(result_path, threshold=-1)
    print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))

if __name__ == "__main__":
    process_options = ['SOFTMAX', 'ODIN', 'LC',
                       'OLTR','OPENMAX',
                       'BCE','TCP','DOCTOR',
                       'OE','EB','ENERGY','VOS','MCD']
    ood_options = ['TinyImagenet', 'TinyImagenet_resize', 'LSUN']

    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument('--process', default='SOFTMAX', choices=process_options)
    parser.add_argument('--ood_dataset', default='tinyImageNet', choices=ood_options)
    parser.add_argument("--pretrained", action="store_true")

    main(parser.parse_args())
