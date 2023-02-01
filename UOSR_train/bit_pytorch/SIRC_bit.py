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

from sirc.sirc_utils import uncertainties, metric_stats

np.random.seed(0)
torch.cuda.manual_seed(0)

def get_vim_params(model, labels, logits, features):

    # Get classifier weights from model: Conv for Resnet101 and MLP for VGG13
    W = model.state_dict()["module.classifier.weight"].detach().clone().cpu()[...,0,0]
    b = model.state_dict()["module.classifier.bias"].detach().clone().cpu()

    u = -torch.linalg.pinv(W) @ b

    # size of subspace
    # pretty much just a heuristic
    D = 1000 if features.shape[-1] > 1500 else 512
    # Mark: change from features.shape[-1] < 512
    if features.shape[-1] <= 512:
        D = features.shape[-1]//2
    centered_feats = features - u
    U = torch.linalg.eigh(
        centered_feats.T@centered_feats
    ).eigenvectors.flip(-1)
    R = U[:,D:] # eigenvectors in columns
    assert R.shape[0] == features.shape[-1]
    vlogits = torch.norm(
        (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1),
        p=2, dim=-1
    )
    alpha = logits.max(dim=-1).values.mean() / vlogits.mean()
    alpha = alpha.item()

    vim_params_dict = {
        "alpha": alpha,
        "u":  u,
        "R": R
    }

    centered_feats = F.normalize(centered_feats, dim=-1)
    U = torch.linalg.eigh(
        centered_feats.T@centered_feats
    ).eigenvectors.flip(-1) # rev order to descending eigenvalue
    R = U[:,D:] # eigenvectors in columns
    assert R.shape[0] == features.shape[-1]
    vlogits = torch.norm(
        (R.T @ centered_feats.unsqueeze(dim=-1)).squeeze(dim=-1),
        p=2, dim=-1
    )
    alpha = logits.max(dim=-1).values.mean() / vlogits.mean()
    alpha = alpha.item()
    vim_params_dict.update({
        "norm_alpha": alpha,
        "norm_R": R
    })

    return vim_params_dict



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
    features = []

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

        if mode == 'SIRC':
            pred, feature = model(images)
            out.append(pred.detach().clone().cpu().numpy())
            features.append(feature.detach().clone().cpu().numpy())


    out = np.concatenate(out)
    classifier = np.concatenate(classifier)
    features = np.concatenate(features)

    return out, features, classifier




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


    model = models.KNOWN_MODELS[args.model](head_size=len(test_set.classes),
                                            zero_head=True,
                                            return_feature=True)
    model = torch.nn.DataParallel(model)
    logger.info('#########################################################')
    logger.info(f"Evaluation OOD Detection Performance with {args.name}")
    ###################################################################
    model_name = 'PRETRAINED_SOFTMAX' if args.pretrained else 'SOFTMAX'
    # pretrained_model = torch.load(f'../log/{args.name}/best_bit.pth.tar')
    pretrained_model = torch.load(f'../log/{model_name}/bit.pth.tar')
    model.load_state_dict(pretrained_model["model"])
    # model.load_from(np.load(f"{args.model}.npz"))

    model = model.to(device)

    ##############      OOD DETECTION EVALUATION    #################
    # confidence results
    ind_logits, ind_features, ind_classifier = run_eval(model, test_loader, args, logger)
    ind_labels = np.ones(ind_logits.shape[0])

    ood_logits, ood_features, ood_classifier = run_eval(model, ood_loader, args, logger)
    ood_labels = np.zeros(ood_logits.shape[0])

    # Network outputs
    ind_logits = torch.from_numpy(ind_logits)
    ood_logits = torch.from_numpy(ood_logits)
    ind_features = torch.from_numpy(ind_features)
    ood_features = torch.from_numpy(ood_features)

    # prediction results
    ind_results = copy.deepcopy(ind_classifier)
    ood_results = copy.deepcopy(ood_classifier)


    # Mark: get combined SIRC results:
    combined_logits = torch.cat([ind_logits, ood_logits])
    combined_features = torch.cat([ind_features, ood_features])
    vim_params_dict = get_vim_params(model,
                                     test_loader.dataset.targets,
                                     ind_logits,
                                     ind_features)


    train_stats = metric_stats(uncertainties(ind_logits, features=ind_features, vim_params=vim_params_dict))
    sirc_metrics = uncertainties(
        combined_logits,
        features=combined_features, gmm_params=None,
        vim_params=vim_params_dict,
        stats=train_stats
    )

    sirc_list = ['SIRC_MSP_res','SIRC_MSP_z','SIRC_H_res', 'SIRC_H_z']
    for sirc_id in sirc_list:
        scores = sirc_metrics[sirc_id].numpy()
        scores = np.exp(scores)
        ind_scores = scores[:len(test_set)]
        ood_scores = scores[len(test_set):]

        print("")
        print("Method: " + args.name)


        results = {}
        results['ind_unctt'] = ind_scores
        results['ood_unctt'] = ood_scores
        results['ind_pred'] = ind_results
        results['ood_pred'] = ood_results
        results['ind_label'] = test_loader.dataset.targets
        results['ood_label'] = ood_loader.dataset.targets
        PREFIX = 'PRETRAINED_' if args.pretrained else ''
        np.savez(f'test_results_saved/{PREFIX}{sirc_id}_{args.ood_dataset}.npz',
                 ind_unctt = ind_scores,
                 ood_unctt = ood_scores,
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



        result_path = f'test_results_saved/{PREFIX}{sirc_id}_{args.ood_dataset}.npz'


        print(f'TESTING {result_path}--------------------------------------------')
        # result_path = f'test_results_saved/OLTR__ood__TinyImagenet_resize.npz'
        print('acc', 'aurc', 'eaurc', 'auroc', 'aupr', 'fpr95', 'ece')
        # results = np.load(result_path, allow_pickle=True)
        acc, aurc, eaurc, auroc, aupr, fpr95, ece = eval_usar(result_path, threshold=-1)
        print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))




if __name__ == "__main__":
    process_options = ['SIRC']
    ood_options = ['TinyImagenet', 'TinyImagenet_resize', 'LSUN']

    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument('--process', default='SIRC', choices=process_options)
    parser.add_argument('--ood_dataset', default='tinyImageNet', choices=ood_options)
    parser.add_argument("--pretrained", action="store_true")

    main(parser.parse_args())