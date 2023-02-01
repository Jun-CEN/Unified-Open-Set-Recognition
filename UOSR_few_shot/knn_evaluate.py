import os
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.covariance import ledoit_wolf
import time
import torch
from scipy.special import xlogy
import torch.nn.functional as F


def parse_args():
    '''Command instruction:
        source activate mmaction
        python experiments/compare_openness.py --ind_ncls 101 --ood_ncls 51
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--base_model', default='i3d', help='the backbone model name')
    parser.add_argument('--baselines', nargs='+', default=['maha_distance'])
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.000423, 0.000024, 0.495783, 0.495783])
    parser.add_argument('--styles', nargs='+', default=['-b'])
    parser.add_argument('--ind_ncls', type=int, default=101, help='the number of classes in known dataset')
    parser.add_argument('--ood_ncls', type=int, help='the number of classes in unknwon dataset')
    parser.add_argument('--ood_data', default='HMDB', help='the name of OOD dataset.')
    parser.add_argument('--num_rand', type=int, default=10, help='the number of random selection for ood classes')
    parser.add_argument('--result_png', default='F1_openness_compare_HMDB.png')
    parser.add_argument('--analyze', default=True, help="analyze score distribution")
    args = parser.parse_args()
    return args


def main():

    result_file = "/mnt/data-nas/test_results_saved/vgg13_pretrained_tiny_val.npz"
    result_file = "/mnt/data-nas/test_results_saved/R50_0.5_64__ood__tinyImageNet_resize__mode__baseline.npz"
    result_file = "/mnt/data_nas/test_results_saved/R50__ood__tinyImageNet_resize__mode__baseline.npz"
    # result_file_softmax = "/mnt/data-nas/DAMO-Action/output/test/tsm_ft_softmax.npz"
    png_file = "/mnt/data-nas/DAMO-Action/output/test/test.png"
    plt.figure(figsize=(8,5))  # (w, h)
    plt.rcParams["font.family"] = "Arial"  # Times New Roman
    fontsize = 15
    for style, baseline in zip(args.styles, args.baselines):
        # result_file = "/mnt/data-nas/DAMO-Action/output/test/tsm_softmax_" + baseline + ".npz"
        print(result_file)
        assert os.path.exists(result_file), "File not found! Run ood_detection first!"
        # load the testing results
        results = np.load(result_file, allow_pickle=True)
        ind_uncertainties = results['ind_unctt']  # (N1,)
        ood_uncertainties = results['ood_unctt']  # (N2,)
        ind_results = results['ind_pred']  # (N1,)
        ood_results = results['ood_pred']  # (N2,)
        ind_labels = results['ind_label']
        ind_train_labels = results['ind_train_label']
        ood_labels = results['ood_label']
        fs_labels = results['fs_label']
        ind_features = results['ind_features'].squeeze()
        ind_train_features = results['ind_train_features'].squeeze()
        ood_features = results['ood_features'].squeeze()
        fs_ood_features = results['fs_features'].squeeze()
        print(ind_features.shape, ood_features.shape, fs_ood_features.shape, ind_results.shape, ood_results.shape, ind_labels.shape, ood_labels.shape, fs_labels.shape)

        # close-set accuracy (multi-class)
        acc = accuracy_score(ind_labels, ind_results)

        d_in, d_ood, preds_maha_in, preds_maha_ood = get_eval_results(
        np.copy(ind_train_features),
        np.copy(ind_features),
        np.copy(ood_features),
        np.copy(fs_ood_features),
        np.copy(ind_train_labels),
        np.copy(ind_results),
        np.copy(ood_results),
        np.copy(fs_labels),
        clusters=1,
        )
        # acc = accuracy_score(ind_labels, preds_maha_in)
        np.savez('/mnt/data_nas/test_results_saved/resnet50_knn.npz', d_in=d_in, d_ood=d_ood,
        ind_pred = results['ind_pred'],  # (N1,)
        ood_pred = ood_results,  # (N2,)
        ind_label = results['ind_label'],
        ood_label = ood_labels,
        msp_in = ind_uncertainties,
        msp_ood = ood_uncertainties)
        
def get_eval_results(ftrain, ftest, food, f_fs_ood, labelstrain, labelstest, labelsood, labelsfsood, clusters=1):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10
    f_fs_ood /= np.linalg.norm(f_fs_ood, axis=-1, keepdims=True) + 1e-10

    # m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    # ftrain = (ftrain - m) / (s + 1e-10)
    # ftest = (ftest - m) / (s + 1e-10)
    # food = (food - m) / (s + 1e-10)

    d_in, d_ood, preds_maha_in, preds_maha_ood = get_scores_2_k(ftrain, ftest, food, f_fs_ood, labelstrain, labelstest, labelsood, labelsfsood, clusters)

    return d_in, d_ood, preds_maha_in, preds_maha_ood

def get_scores_2_k(ftrain, ftest, food, f_fs_ood, labelstrain, labelstest, labelsood, labelsfsood, clusters=1):
    ftrain = torch.tensor(ftrain).cuda()
    ftest = torch.tensor(ftest).cuda()
    food = torch.tensor(food).cuda()
    sim_matrix_ind = torch.matmul(ftest, ftrain.T)
    sim_matrix_ood = torch.matmul(food, ftrain.T)
    K_list = [1,2,5,10,50,100,500,1000]
    K = 50000 - 1
    d_in = 1 - torch.sort(sim_matrix_ind, axis=-1)[0][:,K]
    d_ood = 1 - torch.sort(sim_matrix_ood, axis=-1)[0][:,K]

    return d_in.cpu().numpy(), d_ood.cpu().numpy(), None, None

if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()