# python out_of_distribution_detection.py --ind_dataset cifar100 --ood_dataset tinyImageNet_resize --model vgg13 --process baseline --epsilon 0.001 --checkpoint cifar100_vgg13_budget_0.3_seed_0
import copy
import pdb
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torch.autograd import Variable
from tinyimagenet import load_train_data, load_val_data
from torchvision.transforms.functional import InterpolationMode

import seaborn as sns

from models.vgg import VGG
from models.densenet import DenseNet3
from models.wideresnet import WideResNet
from utils.ood_metrics import tpr95, detection
from utils.datasets import GaussianNoise, UniformNoise
# MARK: Choose different eval_methods for saved npz files
# # uosar, sp, inc_ood, inw_ood
# from selective_prediction.baselines_simple_u.experiments.uosar_hmdb_scratch import eval_uncertainty_methods
from osr_evaluate import eval_uncertainty_methods_uosr, eval_uncertainty_methods_osr, eval_uncertainty_methods_sp, eval_uncertainty_methods_inc_ood
import models.resnet_bit as resnet_bit

ind_options = ['cifar10', 'svhn', 'cifar100']
ood_options = ['tinyImageNet_crop',
               'tinyImageNet_resize',
               'LSUN_crop',
               'LSUN_resize',
               'iSUN',
               'Uniform',
               'Gaussian',
               'all']
model_options = ['densenet', 'wideresnet', 'vgg13', 'vgg11', 'vgg16', 'vgg19', 'resnet50_pretrained_bit']
process_options = ['baseline', 'baseline_logits','ODIN', 'confidence',
                   'confidence_scaling',
                   'doctor','BCE','TCP']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--ind_dataset', default='cifar100', choices=ind_options)
parser.add_argument('--ood_dataset', default='tinyImageNet_resize', choices=ood_options)
parser.add_argument('--model', default='resnet50_pretrained_bit', choices=model_options)
parser.add_argument('--process', default='baseline', choices=process_options)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--T', type=float, default=1000., help='Scaling temperature')
parser.add_argument('--epsilon', type=float, default=0.001, help='Noise magnitude')
parser.add_argument('--checkpoint', default='R50', type=str,
                    help='filepath for checkpoint to load')
parser.add_argument('--validation', action='store_true', default=False,
                    help='only use first 1000 samples from OOD dataset for validation')

args = parser.parse_args()
cudnn.benchmark = True  # Should make training should go faster for large models

filename = args.checkpoint

if args.ind_dataset == 'svhn' and args.model == 'wideresnet':
    args.model = 'wideresnet16_8'

print(args)

###########################
### Set up data loaders ###
###########################

# if args.ind_dataset == 'svhn':
#     normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
#                                      std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
# else:
#     normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                      std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

if args.ind_dataset == 'svhn':
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
else:
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

transform = transforms.Compose([transforms.Resize((32*4, 32*4)),
                                transforms.ToTensor(),
                                normalize])


# tinyImageNet_crop and LSUN_crop are 36x36, so crop to 32x32
crop_transform = transforms.Compose([transforms.CenterCrop(size=(32, 32)),
                                     transforms.ToTensor(),
                                     normalize])

if args.ind_dataset == 'cifar10':
    num_classes = 10
    ind_dataset = datasets.CIFAR10(root='data/',
                                   train=False,
                                   transform=transform,
                                   download=True)
elif args.ind_dataset == 'cifar100':
    num_classes = 100
    ind_dataset = datasets.CIFAR100(root='data/',
                                   train=False,
                                   transform=transform,
                                   download=True)
    ind_train_dataset = datasets.CIFAR100(root='data/',
                                   train=True,
                                   transform=transform,
                                   download=True)
elif args.ind_dataset == 'svhn':
    num_classes = 10
    ind_dataset = datasets.SVHN(root='data/',
                                split='test',
                                transform=transform,
                                download=True)

data_path = 'data/'

if args.ood_dataset == 'tinyImageNet_crop':
    ood_dataset = datasets.ImageFolder(root=data_path + 'TinyImageNet_crop', transform=crop_transform)
elif args.ood_dataset == 'tinyImageNet_resize':
    ood_dataset = datasets.ImageFolder(root=data_path + 'TinyImagenet_resize', transform=transform)
elif args.ood_dataset == 'LSUN_crop':
    ood_dataset = datasets.ImageFolder(root=data_path + 'LSUN_crop', transform=crop_transform)
elif args.ood_dataset == 'LSUN_resize':
    ood_dataset = datasets.ImageFolder(root=data_path + 'LSUN_resize', transform=transform)
elif args.ood_dataset == 'iSUN':
    ood_dataset = datasets.ImageFolder(root=data_path + 'iSUN', transform=transform)
elif args.ood_dataset == 'Uniform':
    ood_dataset = UniformNoise(size=(3, 32, 32), n_samples=10000, low=0., high=1.)
elif args.ood_dataset == 'Gaussian':
    ood_dataset = GaussianNoise(size=(3, 32, 32), n_samples=10000, mean=0.5, variance=1.0)
elif args.ood_dataset == 'all':
    ood_dataset = torch.utils.data.ConcatDataset([
        datasets.ImageFolder(root=data_path + 'TinyImageNet_crop', transform=crop_transform),
        datasets.ImageFolder(root=data_path + 'TinyImagenet_resize', transform=transform),
        datasets.ImageFolder(root=data_path + 'LSUN_crop', transform=crop_transform),
        datasets.ImageFolder(root=data_path + 'LSUN_resize', transform=transform),
        datasets.ImageFolder(root=data_path + 'iSUN', transform=transform)])

fs_loader = load_val_data(128, 128)

ind_loader = torch.utils.data.DataLoader(dataset=ind_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=False,
                                         num_workers=8)

ood_loader = torch.utils.data.DataLoader(dataset=ood_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=False,
                                         num_workers=8)

ind_train_loader = torch.utils.data.DataLoader(dataset=ind_train_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=False,
                                         num_workers=8)                                         

if args.validation:
    # Limit dataset to first 1000 samples for validation and fine-tuning
    # Based on validation procedure from https://arxiv.org/abs/1706.02690
    if args.ood_dataset in ['Gaussian', 'Uniform']:
        ood_loader.dataset.data = ood_loader.dataset.data[:1000]
        ood_loader.dataset.n_samples = 1000
    elif args.ood_dataset == 'all':
        for i in range(len(ood_dataset.datasets)):
            ood_loader.dataset.datasets[i].imgs = ood_loader.dataset.datasets[i].imgs[:1000]
        ood_loader.dataset.cummulative_sizes = ood_loader.dataset.cumsum(ood_loader.dataset.datasets)
    else:
        ood_loader.dataset.imgs = ood_loader.dataset.imgs[:1000]
        ood_loader.dataset.__len__ = 1000
else:
    # Use remaining samples for test evaluation
    if args.ood_dataset in ['Gaussian', 'Uniform']:
        ood_loader.dataset.data = ood_loader.dataset.data[1000:]
        ood_loader.dataset.n_samples = 9000
    elif args.ood_dataset == 'all':
        for i in range(len(ood_dataset.datasets)):
            ood_loader.dataset.datasets[i].imgs = ood_loader.dataset.datasets[i].imgs[1000:]
        ood_loader.dataset.cummulative_sizes = ood_loader.dataset.cumsum(ood_loader.dataset.datasets)
    else:
        ood_loader.dataset.imgs = ood_loader.dataset.imgs[1000:]
        ood_loader.dataset.__len__ = len(ood_loader.dataset.imgs)

##############################
### Load pre-trained model ###
##############################

if args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10).cuda()
elif args.model == 'wideresnet16_8':
    cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8).cuda()
elif args.model == 'densenet':
    cnn = DenseNet3(depth=100, num_classes=num_classes, growth_rate=12, reduction=0.5).cuda()
elif args.model == 'vgg13':
    cnn = VGG(vgg_name='VGG13', num_classes=num_classes).cuda()
elif 'resnet50_pretrained_bit' in args.model:
    cnn = resnet_bit.KNOWN_MODELS['BiT-M-R50x1'](head_size=100, zero_head=True).cuda()

model_dict = cnn.state_dict()
# pretrained_dict = torch.load('checkpoints/' + filename + '.pt')

pretrained_dict = torch.load('/mnt/data_nas/pretrained_model/softmax_r50/best_bit.pth.tar')
for k in list(pretrained_dict['model'].keys()):
    new_k = k.replace('module.', '')
    pretrained_dict['model'][new_k] = pretrained_dict['model'].pop(k)

# MARK: Load certain pretrained model here
# pretrained_dict = torch.load('checkpoints/cifar100_vgg13_budget_0.3_seed_0_loss_confidence.pt')
if args.process == 'TCP' or args.process =='BCE':
    filename = filename + f'_loss_{args.process}'
cnn.load_state_dict(pretrained_dict['model'], strict=False)
mismatch = cnn.load_state_dict(pretrained_dict['model'], strict=False)
print("Keys in model not matched: {}".format(mismatch[0]))
print("Keys in checkpoint not matched: {}".format(mismatch[1]))
print(mismatch)
cnn = cnn.cuda()

cnn.eval()


##############################################
### Evaluate out-of-distribution detection ###
##############################################

def evaluate(data_loader, mode):
    out = []
    classifier = []
    features_all = []
    labels_all = []
    xent = nn.CrossEntropyLoss()
    progress_bar = tqdm(data_loader)
    for i, data in enumerate(progress_bar):
        if type(data) == list:
            images, labels = data
        else:
            images = data
        images = Variable(images, requires_grad=True).cuda()
        images.retain_grad()
        labels_all.append(labels)
        # Select out wrongly classified case
        classified_results, _, features = cnn(images)
        _, classified_idx = torch.max(classified_results, 1)
        classifier.append(classified_idx.cpu().numpy())
        features_all.append(features.detach().cpu().numpy())
        if mode == 'confidence':
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'TCP':
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'BCE':
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'doctor':
            pred, _ = cnn(images)
            pred = F.softmax(pred, dim=-1)
            sum_square = 1 - torch.sum(pred ** 2, dim=-1)
            uncertainties = sum_square / (1 - sum_square)
            confidence = 1 - uncertainties
            out.append(confidence.data.cpu().numpy())


        elif mode == 'confidence_scaling':
            epsilon = args.epsilon

            cnn.zero_grad()
            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence).view(-1)
            loss = torch.mean(-torch.log(confidence))
            loss.backward()

            images = images - args.epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            _, confidence = cnn(images)
            confidence = F.sigmoid(confidence)
            confidence = confidence.data.cpu().numpy()
            out.append(confidence)

        elif mode == 'baseline':
            # https://arxiv.org/abs/1610.02136
            pred, _, features = cnn(images)
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

        elif mode == 'baseline_logits':
            # https://arxiv.org/abs/1610.02136
            pred, _ = cnn(images)
            pred = pred.data.cpu().numpy()
            out.append(pred)

        elif mode == 'ODIN':
            # https://arxiv.org/abs/1706.02690
            T = args.T
            epsilon = args.epsilon

            cnn.zero_grad()
            pred, _ = cnn(images)
            _, pred_idx = torch.max(pred.data, 1)
            labels = Variable(pred_idx)
            pred = pred / T
            loss = xent(pred, labels)
            loss.backward()

            images = images - epsilon * torch.sign(images.grad)
            images = Variable(images.data, requires_grad=True)

            pred, _ = cnn(images)

            pred = pred / T
            pred = F.softmax(pred, dim=-1)
            pred = torch.max(pred.data, 1)[0]
            pred = pred.cpu().numpy()
            out.append(pred)

    out = np.concatenate(out)
    classifier = np.concatenate(classifier)
    features_all = np.concatenate(features_all)
    labels_all = np.concatenate(labels_all)
    return out, classifier, features_all, labels_all

# confidence results
fs_ood_scores, fs_ood_classfier, fs_ood_features, fs_labels = evaluate(fs_loader, args.process)
fs_ood_labels = np.zeros(fs_ood_scores.shape[0])
print(fs_ood_features.shape)


ind_scores, ind_classfier, ind_features, _ = evaluate(ind_loader, args.process)
ind_labels = np.ones(ind_scores.shape[0])
print(ind_features.shape)

ind_train_scores, ind_train_classfier, ind_train_features, _ = evaluate(ind_train_loader, args.process)
ind_train_labels = np.ones(ind_train_scores.shape[0])
print(ind_train_features.shape)

ood_scores, ood_classfier, ood_features, _ = evaluate(ood_loader, args.process)
ood_labels = np.zeros(ood_scores.shape[0])
print(ood_features.shape)

# prediction results
ind_results = copy.deepcopy(ind_classfier)
ood_results = copy.deepcopy(ood_classfier)

ind_classfier = (ind_classfier == ind_loader.dataset.targets)
ood_classfier = (ood_classfier == ood_loader.dataset.targets)
labels = np.concatenate([ind_labels, ood_labels])
scores = np.concatenate([ind_scores, ood_scores])

fpr_at_95_tpr = tpr95(ind_scores, ood_scores)
detection_error, best_delta = detection(ind_scores, ood_scores)
auroc = metrics.roc_auc_score(labels, scores)
aupr_in = metrics.average_precision_score(labels, scores)
aupr_out = metrics.average_precision_score(-1 * labels + 1, 1 - scores)

ind_wrong_num = len(ind_classfier) - sum(ind_classfier)
unified_labels = np.concatenate([ind_classfier*1.0, ood_labels])
unified_auroc = metrics.roc_auc_score(unified_labels , scores)
unified_aupr_in = metrics.average_precision_score(unified_labels , scores)
unified_aupr_out = metrics.average_precision_score(-1 * unified_labels  + 1, 1 - scores)

print("")
print("Method: " + args.process)
print("TPR95 (lower is better): ", fpr_at_95_tpr)
print("Detection error (lower is better): ", detection_error)
print("Best threshold:", best_delta)
print("AUROC (higher is better): ", auroc)
print("AUPR_IN (higher is better): ", aupr_in)
print("AUPR_OUT (higher is better): ", aupr_out)

print("Wrong numbers in InD class: ", ind_wrong_num)
print("UNIFIED_AUROC (higher is better): ", unified_auroc)
print("UNIFIED_AUPR_IN (higher is better): ", unified_aupr_in)
print("UNIFIED_AUPR_OUT (higher is better): ", unified_aupr_out)

results = {}
results['ind_unctt'] = 1 - ind_scores
results['ood_unctt'] = 1 - ood_scores
results['ind_pred'] = ind_results
results['ood_pred'] = ood_results
results['ind_label'] = ind_loader.dataset.targets
results['ind_train_label'] = ind_train_loader.dataset.targets
results['ood_label'] = ood_loader.dataset.targets
results['fs_features'] = fs_ood_features
results['fs_labels'] = fs_labels
results['ind_features'] = ind_features
results['ind_train_features'] = ind_train_features
results['ood_features'] = ood_features

np.savez(f'test_results_saved/{filename}__ood__{args.ood_dataset}__mode__{args.process}.npz',
         ind_unctt = 1 - ind_scores,
         ood_unctt = 1 - ood_scores,
         ind_pred = ind_results,
         ood_pred = ood_results,
         ind_label = ind_loader.dataset.targets,
         ind_train_label = ind_train_loader.dataset.targets,
         ood_label = ood_loader.dataset.targets,
         fs_features = fs_ood_features,
         ind_features = ind_features,
         ind_train_features = ind_train_features,
         ood_features = ood_features,
         fs_label = fs_labels)
print('____________ TEST RESULTS SAVED ________________')
print('*************************************************')
result_path = f'test_results_saved/{filename}__ood__{args.ood_dataset}__mode__{args.process}.npz'

print('acc', 'aurc', 'eaurc', 'auroc', 'aupr', 'fpr95', 'ece')
# result_path = f'test_results_saved/OLTR__ood__TinyImagenet_resize.npz'
# results = np.load(result_path, allow_pickle=True)

# MARK: Choose different evaluation protocals for saved npz file
acc, aurc, eaurc, auroc, aupr, fpr95, ece = eval_uncertainty_methods_uosr(result_path, threshold=-1)
print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))

acc, aurc, eaurc, auroc, aupr, fpr95, ece = eval_uncertainty_methods_osr(result_path, threshold=-1)
print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))

acc, aurc, eaurc, auroc, aupr, fpr95, ece = eval_uncertainty_methods_sp(result_path, threshold=-1)
print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))

acc, aurc, eaurc, auroc, aupr, fpr95, ece = eval_uncertainty_methods_inc_ood(result_path, threshold=-1)
print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))

# # Save the results for fast evaluation
# ind_uncertainties = results['ind_unctt']  # (N1,)
# ood_uncertainties = results['ood_unctt']  # (N2,)
# ind_results = results['ind_pred']  # (N1,)
# ood_results = results['ood_pred']  # (N2,)
# ind_labels = results['ind_label']
# ood_labels = results['ood_label']

test_all = False
if test_all:
    name_list = ['SoftMax', 'ODIN', 'LC',
                 'OpenMax', 'OLTR', 'PROSER',
                 'BCE150', 'TCP', 'DOCTOR',
                 'SIRC_MSP_Z','SIRC_MSP_RES',
                 'SIRC_H_Z','SIRC_H_Z',]
    for name in name_list:
        result_path = f'final_results_saved/{name}.npz'
        results = np.load(result_path, allow_pickle=True)
        print(f'********** {name} ****************')
        # print(f'Ind untt mean {np.mean(results.f.ind_unctt)}\t'
        #       f'Ood untt mean {np.mean(results.f.ood_unctt)}')

        acc, aurc, eaurc, auroc, aupr, fpr95, ece = \
            eval_uncertainty_methods(result_path, threshold=-1)

        print('&'+"%.2f"%(acc * 100), '&'+"%.2f"%(aurc * 1000), '&'+"%.2f"%(eaurc * 1000), '&'+"%.2f"%(auroc * 100), '&'+"%.2f"%(aupr * 100), '&'+"%.2f"%(fpr95 * 100), '&'+"%.3f"%(ece))




ranges = (np.min(scores), np.max(scores))
plt.figure()
sns.distplot(ind_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='In-distribution')
sns.distplot(ood_scores.ravel(), hist_kws={'range': ranges}, kde=False, bins=50, norm_hist=True, label='Out-of-distribution')
plt.xlabel('Confidence')
plt.ylabel('Density')
plt.legend()
plt.show()
