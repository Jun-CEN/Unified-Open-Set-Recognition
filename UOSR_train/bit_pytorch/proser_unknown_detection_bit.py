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

from torch.autograd import Variable
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR

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

def dummypredict(model, x):
    out = model.module.head(model.module.body(model.module.root(x)))[...,0,0]
    out = model.module.clf2(out)

    return out

def pre2block(model, x):
    out = model.module.root(x)
    out = model.module.body.block1(out)
    out = model.module.body.block2(out)

    return out

def latter2blockclf1(model, x):
    out = model.module.body.block3(x)
    out = model.module.body.block4(out)
    out = model.module.head(out)
    pred = model.module.classifier(out)[...,0,0]

    return pred

def latter2blockclf2(model, x):
    out = model.module.body.block3(x)
    out = model.module.body.block4(out)
    out = model.module.head(out)[...,0,0]
    pred = model.module.clf2(out)

    return pred


def train_dummy(net, trainloader, args):
    criterion = nn.CrossEntropyLoss()
    if args.pretrained:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[10, 25, 40], gamma=1.0)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[25, 40], gamma=0.5)



    net.train()
    alpha = 1
    train_loss = 0
    correct = 0
    total = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        totallenth=len(inputs)
        halflenth=int(len(inputs)/2)
        beta=torch.distributions.beta.Beta(alpha, alpha).sample([]).item()

        prehalfinputs=inputs[:halflenth]
        prehalflabels=targets[:halflenth]
        laterhalfinputs=inputs[halflenth:]
        laterhalflabels=targets[halflenth:]

        index = torch.randperm(prehalfinputs.size(0)).cuda()
        pre2embeddings=pre2block(net,prehalfinputs)
        mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index]

        dummylogit=dummypredict(net,laterhalfinputs)
        lateroutputs=net(laterhalfinputs)
        latterhalfoutput=torch.cat((lateroutputs,dummylogit),1)
        prehalfoutput=torch.cat((latter2blockclf1(net,mixed_embeddings),latter2blockclf2(net,mixed_embeddings)),1)

        maxdummy,_=torch.max(dummylogit.clone(),dim=1)
        maxdummy=maxdummy.view(-1,1)
        dummpyoutputs=torch.cat((lateroutputs.clone(),maxdummy),dim=1)
        for i in range(len(dummpyoutputs)):
            nowlabel=laterhalflabels[i]
            dummpyoutputs[i][nowlabel]=-1e9
        # MARK: args.known_class
        dummytargets=torch.ones_like(laterhalflabels)*100


        outputs=torch.cat((prehalfoutput,latterhalfoutput),0)
        loss1= criterion(prehalfoutput, (torch.ones_like(prehalflabels)*100).long().cuda())
        loss2=criterion(latterhalfoutput,laterhalflabels )
        loss3= criterion(dummpyoutputs, dummytargets)
        # MARK: Fine tune the loss here
        loss=0.01*loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx == 97:
            print(f'### BATCH{batch_idx} ###\n'
                  f'Loss_total: {loss.item()}, Loss_1: {0.01*loss1.item()} \n'
                  f'Loss_2: {loss2.item()}, Loss_3: {loss3.item()} \n')
            print('####################################################')
            print("Train Acc is %.2f" % (100.*correct/total))

    scheduler.step()



def val_dummy(epoch, net, closerloader, openloader):
    known_class = 100
    net.eval()
    CONF_AUC=True
    CONF_DeltaP=False
    auclist1=[]
    auclist2=[]
    linspace=[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    closelogits=torch.zeros((len(closerloader.dataset.data),known_class+1)).cuda()
    closelogits_ind_right=torch.zeros((len(closerloader.dataset.data), 1)).cuda()
    openlogits=torch.zeros((len(openloader.dataset.samples),known_class+1)).cuda()
    openlogits_ood_right=torch.zeros((len(openloader.dataset.samples), 1)).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batchnum=len(targets)
            logits=net(inputs)
            _, ind_classfied_idx = torch.max(logits,1)
            # ind_right = (classfied_idx == targets)
            dummylogit=dummypredict(net,inputs)
            maxdummylogit,_=torch.max(dummylogit,1)
            maxdummylogit=maxdummylogit.view(-1,1)
            totallogits=torch.cat((logits,maxdummylogit),dim=1)
            closelogits[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=totallogits
            closelogits_ind_right[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=ind_classfied_idx.view(-1,1)
        for batch_idx, (inputs, targets) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batchnum=len(targets)
            logits=net(inputs)
            _, ood_classfied_idx = torch.max(logits,1)
            dummylogit=dummypredict(net,inputs)
            maxdummylogit,_=torch.max(dummylogit,1)
            maxdummylogit=maxdummylogit.view(-1,1)
            totallogits=torch.cat((logits,maxdummylogit),dim=1)
            openlogits[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=totallogits
            openlogits_ood_right[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=ood_classfied_idx.view(-1,1)
    Logitsbatchsize=200
    maxauc=0
    maxaucbias=0
    for biasitem in linspace:
        if CONF_AUC:
            for temperature in [1024.0]:
                closeconf=[]
                openconf=[]
                closeiter=int(len(closelogits)/Logitsbatchsize)
                openiter=int(len(openlogits)/Logitsbatchsize)
                for batch_idx  in range(closeiter):
                    logitbatch=closelogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    conf=embeddings[:,-1]
                    closeconf.append(conf.cpu().numpy())
                closeconf=np.reshape(np.array(closeconf),(-1))
                closelabel=np.ones_like(closeconf)
                for batch_idx  in range(openiter):
                    logitbatch=openlogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    conf=embeddings[:,-1]
                    openconf.append(conf.cpu().numpy())
                openconf=np.reshape(np.array(openconf),(-1))
                openlabel=np.zeros_like(openconf)
                totalbinary=np.hstack([closelabel,openlabel])
                totalconf=np.hstack([closeconf,openconf])
                auc1=metrics.roc_auc_score(1-totalbinary,totalconf)
                auc2=metrics.roc_auc_score(totalbinary,totalconf)
                auc3=metrics.roc_auc_score(totalbinary,totalconf)
                auroc = metrics.roc_auc_score(totalbinary,totalconf)
                aupr_in = metrics.average_precision_score(totalbinary, totalconf)
                aupr_out = metrics.average_precision_score(-1*totalbinary+1, 1-totalconf)
                # Todo: Compare the macro F1 result with the original paper with CIFAR 10
                acc = metrics.accuracy_score(closerloader.dataset.targets,
                                             closelogits_ind_right.cpu().numpy().squeeze()[:len(closelabel)])
                print(f'Acc is {acc} ----------------------------------------------------------')
                print('Temperature:',temperature,'bias',biasitem,'AUROC (higher is better',auroc)
                print('Temperature:',temperature,'bias',biasitem,'AUPR_IN (higher is better',aupr_in)
                print('Temperature:',temperature,'bias',biasitem,'AUPR_OUT (higher is better',aupr_out)
                print('Temperature:',temperature,'bias',biasitem,'AUC',auc2)

                return closeconf, openconf, \
                       closelogits_ind_right.cpu().numpy().squeeze()[:len(closelabel)],\
                       openlogits_ood_right.cpu().numpy().squeeze()[:len(openlabel)]

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
    ############################################
    print('Now processing with Proser, resuming from checkpoints')
    model = models.KNOWN_MODELS[args.model](head_size=len(test_set.classes),
                                            zero_head=True)
    model = torch.nn.DataParallel(model)
    logger.info('#########################################################')
    logger.info(f"Evaluation OOD Detection Performance with {args.name}")
    ###################################################################
    model_name = 'PRETRAINED_SOFTMAX' if args.pretrained else 'SOFTMAX'
    # pretrained_model = torch.load(f'../log/{args.name}/best_bit.pth.tar')
    pretrained_model = torch.load(f'../log/{model_name}/bit.pth.tar')
    model.load_state_dict(pretrained_model["model"])
    # model.load_from(np.load(f"{args.model}.npz"))

    model.module.clf2 = nn.Linear(2048, 1)
    model = model.to(device)

    FINE_TUNE_MAX_EPOCH = 10 if args.pretrained else 50

    for finetune_epoch in range(FINE_TUNE_MAX_EPOCH):
        print(f"##### {args.name} FINETUNE EPOCH {finetune_epoch} #####")
        train_dummy(model, train_loader, args)

        ind_scores, ood_scores, ind_results, ood_results = \
            val_dummy(finetune_epoch, model, test_loader, ood_loader)
        if finetune_epoch == FINE_TUNE_MAX_EPOCH - 1:
            np.savez(f'./test_results_saved/{args.save_name}.npz',
                     ind_unctt = 1 - ind_scores,
                     ood_unctt = 1 - ood_scores,
                     ind_pred = ind_results,
                     ood_pred = ood_results,
                     ind_label = test_loader.dataset.targets,
                     ood_label = ood_loader.dataset.targets)
            print('____________ TEST RESULTS SAVED ________________')
            print('*************************************************')





if __name__ == "__main__":
    process_options = ['PROSER']
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