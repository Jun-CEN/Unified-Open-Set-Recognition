# TODO: Check if we need to rearrange lr_scheduler
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
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
import bit_hyperrule_cifar100 as bit_hyperrule

from torch.autograd import Variable
from PIL import Image

np.random.seed(0)
torch.cuda.manual_seed(0)


class OE_300k_dataloader(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = np.load(data)
        self.transforms = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        hdct = self.data[index, :, :, :]
        hdct = np.squeeze(hdct)

        hdct = Image.fromarray(np.uint8(hdct))
        hdct = self.transforms(hdct)

        return hdct, 100


def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i

def mktrainval(args, logger):
    """Returns train and validation datasets."""
    precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
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

    train_out_set = OE_300k_dataloader(args.datadir +'/300K_random_images.npy', transform=train_tx)
    if args.examples_per_class is not None:
        logger.info(f"Looking for {args.examples_per_class} imag es per class...")
        indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
        train_set = torch.utils.data.Subset(train_set, indices=indices)

    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = args.batch // args.batch_split

    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=500, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    if micro_batch_size <= len(train_set):
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        train_out_loader = torch.utils.data.DataLoader(
            train_out_set,
            batch_size=args.oe_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

        train_out_loader = torch.utils.data.DataLoader(
            train_out_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
            sampler=torch.utils.data.RandomSampler(train_out_set, replacement=True, num_samples=micro_batch_size))

    return train_set, valid_set, train_loader, valid_loader, train_out_set, train_out_loader


def run_eval(model, data_loader, device, chrono, logger, step):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1, all_top5 = [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits = model(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
                top1, top5 = topk(logits, y, ks=(1, 5))
                all_c.extend(c.cpu())  # Also ensures a sync point.
                all_top1.extend(top1.cpu())
                all_top5.extend(top5.cpu())

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
                f"top1 {np.mean(all_top1):.2%}, "
                f"top5 {np.mean(all_top5):.2%}")
    logger.flush()
    return all_c, all_top1, all_top5


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)


def main(args):
    logger = bit_common.setup_logger(args)

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Going to train on {device}")

    train_set, valid_set, train_loader, valid_loader, \
    train_out_set, train_out_loader = mktrainval(args, logger)


    if args.score == 'EB':
        model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes),
                                                zero_head=True,
                                                extra_class=True)
    elif args.score == 'VOS':
        model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes),
                                                zero_head=True,
                                                forward_virtual=True)
    else:
        model = models.KNOWN_MODELS[args.model](head_size=len(valid_set.classes),
                                            zero_head=True)



    logger.info(f"PRETRAINED Outlier Exposure from pretrained model {args.model}.npz with {args.score}###############################")
    ###################################################################
    # MARK: Only update certain layer, changed in models.py
    model.load_from(np.load(f"{args.model}.npz"))

    # ###################################################################
    # logger.info("Moving model onto all GPUs")
    # model = torch.nn.DataParallel(model)


    # Optionally resume from a checkpoint.
    # Load it to CPU first as we'll move the model to GPU later.
    # This way, we save a little bit of GPU memory when loading.
    step = 0

    # Note: no weight-decay!
    # optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    # optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, nesterov=True, weight_decay=5e-4)
    optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, nesterov=True, weight_decay=5e-4)

    # Resume fine-tuning if we find a saved model.
    savename = pjoin(args.logdir, args.name, "bit.pth.tar")
    savename_best =  pjoin(args.logdir, args.name, "best_bit.pth.tar")
    try:
        logger.info(f"Model will be saved in '{savename}'")
        checkpoint = torch.load(savename, map_location="cpu")
        logger.info(f"Found saved model to resume from at '{savename}'")

        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info(f"Resumed at step {step}")
    except FileNotFoundError:
        logger.info("Fine-tuning from BiT")

    model = model.to(device)
    optim.zero_grad()
    print(model)
    model.train()
    mixup = bit_hyperrule.get_mixup(len(train_set))
    mixup = 0
    cri = torch.nn.CrossEntropyLoss().to(device)

    logger.info("Starting training!")
    chrono = lb.Chrono()
    accum_steps = 0
    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
    end = time.time()

    best_acc = 0

    with lb.Uninterrupt() as u:
        # for in_set, out_set in zip(train_loader, train_out_loader):
        for in_set, out_set in zip(recycle(train_loader), recycle(train_out_loader)):
            # measure data loading time, which is spent in the `for` statement.
            chrono._done("load", time.time() - end)

            if u.interrupted:
                break
            x = torch.cat((in_set[0], out_set[0]), 0)
            y = in_set[1]
            y_out = out_set[1]
            # Schedule sending to GPU(s)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_out = y_out.to(device, non_blocking=True)


            # Update learning-rate, including stop training if over.
            lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
            if lr is None:
                break
            for param_group in optim.param_groups:
                param_group["lr"] = lr

            if mixup > 0.0:
                x, y_a, y_b = mixup_data(x, y, mixup_l)

            # compute output
            with chrono.measure("fprop"):

                # forward
                if args.score == 'VOS':
                    logits, vos_logits = model.forward_virtual(x)
                else:
                    logits = model(x)

                if mixup > 0.0:
                    c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
                else:
                    # Mark: 1. train_ind classification loss
                    c = cri(logits[:len(in_set[0])], y)
                    # cross-entropy from softmax distribution to uniform distribution
                    if args.score == 'ENERGY':
                        Ec_out = -torch.logsumexp(logits[len(in_set[0]):], dim=1)
                        Ec_in = -torch.logsumexp(logits[:len(in_set[0])], dim=1)
                        c += 0.01*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())
                    elif args.score == 'OE':
                        c += 0.5 * -(logits[len(in_set[0]):].mean(1) - torch.logsumexp(logits[len(in_set[0]):], dim=1)).mean()
                    elif args.score == 'EB':
                        target = torch.cat((y, y_out), 0)
                        c = F.cross_entropy(logits, target)
                    elif args.score == 'VOS':
                        binary_labels = torch.zeros(len(logits)).cuda()
                        binary_labels[len(in_set[0]):] = 1
                        loss_energy = F.binary_cross_entropy_with_logits(vos_logits.squeeze(), binary_labels)
                        c += 0.5 * -(logits[len(in_set[0]):].mean(1) - torch.logsumexp(logits[len(in_set[0]):], dim=1)).mean()
                        c += loss_energy
                    elif args.score == 'MCD':
                        Ec_out = -torch.logsumexp(logits[len(in_set[0]):], dim=1)
                        Ec_in = -torch.logsumexp(logits[:len(in_set[0])], dim=1)
                        c += 0.03*(torch.pow(F.relu(Ec_in-args.m_in), 2).mean() + torch.pow(F.relu(args.m_out-Ec_out), 2).mean())

                    else:
                        print('Please choose at least one outlier exposure method !!!!!!!!')

                c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

            # Accumulate grads
            with chrono.measure("grads"):
                (c / args.batch_split).backward()
                accum_steps += 1

            accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
            if args.eval_every and step % args.eval_every == 0:
                logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
                logger.flush()

            # Update params
            if accum_steps == args.batch_split:
                with chrono.measure("update"):
                    optim.step()
                    optim.zero_grad()
                step += 1
                accum_steps = 0
                # Sample new mixup ratio for next batch
                mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

                # Run evaluation and save the model.
                if args.eval_every and step % args.eval_every == 0:
                    _, acc_tmp, _ = run_eval(model, valid_loader, device, chrono, logger, step)
                    if 1:
                        torch.save({
                            "step": step,
                            "model": model.state_dict(),
                            "optim" : optim.state_dict(),
                        }, savename)
                    if np.mean(acc_tmp) > best_acc:
                        best_acc = np.mean(acc_tmp)
                        torch.save({
                            "step": step,
                            "model": model.state_dict(),
                            "optim" : optim.state_dict(),
                        }, savename_best)

            end = time.time()

        # Final eval at end of training.
        run_eval(model, valid_loader, device, chrono, logger, step='end')

    logger.info(f"Timings:\n{chrono}")


if __name__ == "__main__":
    oe_options = ['OE', 'ENERGY', 'VOS','EB']

    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument("--datadir", required=True,
                        help="Path to the ImageNet data folder, preprocessed for torchvision.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument('--score', default='OE', choices=oe_options)
    parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--m_in', type=float, default=-25., help='margin for in-distribution; above this value will be penalized')
    parser.add_argument('--m_out', type=float, default=-7., help='margin for out-distribution; below this value will be penalized')
    main(parser.parse_args())
