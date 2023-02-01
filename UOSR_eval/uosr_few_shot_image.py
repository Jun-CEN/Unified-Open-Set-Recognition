import os
import argparse
from matplotlib.pyplot import axis
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve
from terminaltables import AsciiTable
import torch

def parse_args():
    '''Command instruction:
        source activate mmaction
        python experiments/compare_openness.py
    '''
    parser = argparse.ArgumentParser(description='Compare the performance of openness')
    # model config
    parser.add_argument('--base_model', default='resnet50', help='the backbone model name')
    parser.add_argument('--ood_data', default='TinyImagenet', help='the name of OOD dataset.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    parser.add_argument('--baseline_results', nargs='+', help='the testing results files.')
    parser.add_argument('--UOSAR', default='True', help='whether UOSAR or not')
    args = parser.parse_args()
    return args

def get_sirc_params_u(unc_stats):

    mean_tmp = np.mean(unc_stats)
    std_tmp = np.std(unc_stats)
    a = mean_tmp + 3 * std_tmp
    b = 1/std_tmp

    return a, b

def sirc_2_u(s1, s2, a,b, s1_max=0):
    "Combine 2 confidence metrics with SIRC."
    # use logarithm for stability
    soft = s1
    additional = np.exp(np.zeros(len(s2))) + np.exp(b * (s2 - a))
    res = np.log(soft * additional + 1e-6)
    return res # return as confidence

def get_fs_knns_params_u(unc_stats, k):

    mean_tmp = np.mean(unc_stats)
    std_tmp = np.std(unc_stats)
    b = mean_tmp - k * std_tmp

    return b

def sigmoid_own(x, a, b):
    res = 1 / (1+np.exp(a * (-(x-b))))
    return res

def eval_osr(y_true, y_pred):
    # open-set auc-roc (binary class)
    auroc = roc_auc_score(y_true, y_pred)

    # open-set auc-pr (binary class)
    # as an alternative, you may also use `ap = average_precision_score(labels, uncertains)`, which is approximate to aupr.
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)

    # open-set fpr@95 (binary class)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    operation_idx = np.abs(tpr - 0.95).argmin()
    fpr95 = fpr[operation_idx]  # FPR when TPR at 95%

    return auroc, aupr, fpr95

def eval_uncertainty_methods(result_file, method):
    assert os.path.exists(result_file), "File not found! Run ood_detection first!\n%s"%(result_file)
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    msp_in = results['msp_in']  # (N1,)
    msp_ood = results['msp_ood']  # (N2,)
    d_in_all = results['d_in']  # (N1,)
    d_ood_all = results['d_ood']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    if 'knns' in result_file:
        d_in_train_all = results['d_train']
        d_ood_fs_all = results['d_ood_fs']

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    if method == 'softmax':
        ind_uncertainties = msp_in
        ood_uncertainties = msp_ood
    elif method == 'knn':
        ind_uncertainties = d_in_all
        ood_uncertainties = d_ood_all

    repeated_clss = [35, 29, 15, 26, 30, 34, 43, 31]
    index_repeated = np.zeros_like(ood_results)
    for i in repeated_clss:
        index_repeated[ood_labels == i] = 1
    index_no_repeated = 1 - index_repeated
    ood_uncertainties = ood_uncertainties[index_no_repeated==1]
    ood_results = ood_results[index_no_repeated==1]
    ood_labels = ood_labels[index_no_repeated==1]
    
    u_ind_gt = np.zeros_like(ind_labels)
    u_ood_gt = np.ones_like(ood_labels)

    u_ind_gt_uosr = u_ind_gt.copy()
    u_ind_gt_uosr[ind_results != ind_labels] = 1

    labels_uosr = np.concatenate((u_ind_gt_uosr, u_ood_gt))
    labels_osr = np.concatenate((u_ind_gt, u_ood_gt))
    
    preds = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    auroc_uosr, aupr, fpr95 = eval_osr(labels_uosr, preds)
    auroc_osr, aupr, fpr95 = eval_osr(labels_osr, preds)

    labels_inc_ood = np.concatenate((u_ind_gt[ind_results == ind_labels], u_ood_gt))
    preds_inc_ood = np.concatenate((ind_uncertainties[ind_results == ind_labels], ood_uncertainties), axis=0)
    auroc_inc_ood, aupr, fpr95 = eval_osr(labels_inc_ood, preds_inc_ood)

    labels_inc_inw = u_ind_gt_uosr
    preds_inc_inw = ind_uncertainties
    auroc_inc_inw, aupr, fpr95 = eval_osr(labels_inc_inw, preds_inc_inw)

    aurc_uosr, eaurc = calc_aurc_eaurc(1-preds, 1-labels_uosr)

    return aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw

def eval_uncertainty_methods_few_shot_knns(result_file, method):
    assert os.path.exists(result_file), "File not found! Run ood_detection first!\n%s"%(result_file)
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    msp_in = results['msp_in']  # (N1,)
    msp_ood = results['msp_ood']  # (N2,)
    d_in_all = results['d_in']  # (N1,)
    d_ood_all = results['d_ood']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']
    d_in_train_all = results['d_train']
    d_ood_fs_all = results['d_ood_fs']

    num_try = d_in_all.shape[0]
    res_matrix_auroc_uosr = np.zeros((num_try,))
    res_matrix_auroc_osr = np.zeros((num_try,))
    res_matrix_auroc_inc_ood = np.zeros((num_try,))
    res_matrix_auroc_inc_inw = np.zeros((num_try,))
    res_matrix_aurc = np.zeros((num_try,))

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    for k in range(num_try):
        d_in = d_in_all[k]
        d_ood = d_ood_all[k]
        d_in_train = d_in_train_all[k]
        d_ood_fs = d_ood_fs_all[k]

        baseline = "test"
        msp_all = np.concatenate((msp_in, msp_ood), axis=0)
        d_all = np.concatenate((d_in, d_ood, d_in_train, d_ood_fs), axis=0)

        d_in = (d_in - np.min(d_all)) / (np.max(d_all) - np.min(d_all))
        d_ood = (d_ood - np.min(d_all)) / (np.max(d_all) - np.min(d_all))
        d_in_train = (d_in_train - np.min(d_all)) / (np.max(d_all) - np.min(d_all))
        d_ood_fs = (d_ood_fs - np.min(d_all)) / (np.max(d_all) - np.min(d_all))
        msp_in = (msp_in - np.min(msp_all)) / (np.max(msp_all) - np.min(msp_all))
        msp_ood = (msp_ood - np.min(msp_all)) / (np.max(msp_all) - np.min(msp_all))

        d_in_c = d_in[ind_results == ind_labels]
        d_in_w = d_in[ind_results != ind_labels]
        msp_in_c = msp_in[ind_results == ind_labels]
        msp_in_w = msp_in[ind_results != ind_labels]

        if method == 'fs-knn':
            ind_uncertainties = d_in
            ood_uncertainties = d_ood
        elif method == 'fs-knn+s':
            ind_uncertainties = d_in + msp_in
            ood_uncertainties = d_ood + msp_ood
        elif method == 'fs-knn*s':
            ind_uncertainties = d_in * msp_in
            ood_uncertainties = d_ood * msp_ood
        elif method == 'sirc':
            a, b = get_sirc_params_u(d_in_train)
            uncertains_msp = np.concatenate((msp_in, msp_ood))
            uncertains_d = np.concatenate((d_in, d_ood))
            uncertains = sirc_2_u(
            uncertains_msp, uncertains_d, a, b
            )
            ind_uncertainties = uncertains[:ind_labels.shape[0]]
            ood_uncertainties = uncertains[ind_labels.shape[0]:]
        elif method == 'fs-knns':
            a = 50
            b = get_fs_knns_params_u(d_ood_fs, 1)
            uncertains_msp = np.concatenate((msp_in, msp_ood))
            uncertains_d = np.concatenate((d_in, d_ood))
            uncertains_msp += sigmoid_own(uncertains_d, a=a, b=b) * uncertains_d
            uncertains = uncertains_msp
            ind_uncertainties = uncertains[:ind_labels.shape[0]]
            ood_uncertainties = uncertains[ind_labels.shape[0]:]

        repeated_clss = [35, 29, 15, 26, 30, 34, 43, 31]
        index_repeated = np.zeros_like(ood_results)
        for i in repeated_clss:
            index_repeated[ood_labels == i] = 1
        index_no_repeated = 1 - index_repeated
        ood_uncertainties = ood_uncertainties[index_no_repeated==1]
        ood_results = ood_results[index_no_repeated==1]
        ood_labels = ood_labels[index_no_repeated==1]
    
        u_ind_gt = np.zeros_like(ind_labels)
        u_ood_gt = np.ones_like(ood_labels)

        u_ind_gt_uosr = u_ind_gt.copy()
        u_ind_gt_uosr[ind_results != ind_labels] = 1

        labels_uosr = np.concatenate((u_ind_gt_uosr, u_ood_gt))
        labels_osr = np.concatenate((u_ind_gt, u_ood_gt))
        
        preds = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
        auroc_uosr, aupr, fpr95 = eval_osr(labels_uosr, preds)
        auroc_osr, aupr, fpr95 = eval_osr(labels_osr, preds)

        labels_inc_ood = np.concatenate((u_ind_gt[ind_results == ind_labels], u_ood_gt))
        preds_inc_ood = np.concatenate((ind_uncertainties[ind_results == ind_labels], ood_uncertainties), axis=0)
        auroc_inc_ood, aupr, fpr95 = eval_osr(labels_inc_ood, preds_inc_ood)

        labels_inc_inw = u_ind_gt_uosr
        preds_inc_inw = ind_uncertainties
        auroc_inc_inw, aupr, fpr95 = eval_osr(labels_inc_inw, preds_inc_inw)

        aurc_uosr, eaurc = calc_aurc_eaurc(1-preds, 1-labels_uosr)

        res_matrix_auroc_uosr[k] = auroc_uosr
        res_matrix_auroc_osr[k] = auroc_osr
        res_matrix_auroc_inc_ood[k] = auroc_inc_ood
        res_matrix_auroc_inc_inw[k] = auroc_inc_inw
        res_matrix_aurc[k] = aurc_uosr
    
    return np.mean(res_matrix_aurc), np.mean(res_matrix_auroc_uosr), np.mean(res_matrix_auroc_osr), np.mean(res_matrix_auroc_inc_ood), np.mean(res_matrix_auroc_inc_inw)

def eval_uncertainty_methods_few_shot_ssd(result_file, method):
    assert os.path.exists(result_file), "File not found! Run ood_detection first!\n%s"%(result_file)
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    msp_in = results['msp_in']  # (N1,)
    msp_ood = results['msp_ood']  # (N2,)
    d_in_all = results['d_in']  # (N1,)
    d_ood_all = results['d_ood']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']

    num_try = d_in_all.shape[0]
    res_matrix_auroc_uosr = np.zeros((num_try,))
    res_matrix_auroc_osr = np.zeros((num_try,))
    res_matrix_auroc_inc_ood = np.zeros((num_try,))
    res_matrix_auroc_inc_inw = np.zeros((num_try,))
    res_matrix_aurc = np.zeros((num_try,))

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    for k in range(num_try):
        d_in = d_in_all[k]
        d_ood = d_ood_all[k]

        baseline = "test"
        msp_all = np.concatenate((msp_in, msp_ood), axis=0)
        d_all = np.concatenate((d_in, d_ood), axis=0)

        d_in = (d_in - np.min(d_all)) / (np.max(d_all) - np.min(d_all))
        d_ood = (d_ood - np.min(d_all)) / (np.max(d_all) - np.min(d_all))
        msp_in = (msp_in - np.min(msp_all)) / (np.max(msp_all) - np.min(msp_all))
        msp_ood = (msp_ood - np.min(msp_all)) / (np.max(msp_all) - np.min(msp_all))

        d_in_c = d_in[ind_results == ind_labels]
        d_in_w = d_in[ind_results != ind_labels]
        msp_in_c = msp_in[ind_results == ind_labels]
        msp_in_w = msp_in[ind_results != ind_labels]

        if method == 'ssd':
            ind_uncertainties = d_in
            ood_uncertainties = d_ood

        repeated_clss = [35, 29, 15, 26, 30, 34, 43, 31]
        index_repeated = np.zeros_like(ood_results)
        for i in repeated_clss:
            index_repeated[ood_labels == i] = 1
        index_no_repeated = 1 - index_repeated
        ood_uncertainties = ood_uncertainties[index_no_repeated==1]
        ood_results = ood_results[index_no_repeated==1]
        ood_labels = ood_labels[index_no_repeated==1]
    
        u_ind_gt = np.zeros_like(ind_labels)
        u_ood_gt = np.ones_like(ood_labels)

        u_ind_gt_uosr = u_ind_gt.copy()
        u_ind_gt_uosr[ind_results != ind_labels] = 1

        labels_uosr = np.concatenate((u_ind_gt_uosr, u_ood_gt))
        labels_osr = np.concatenate((u_ind_gt, u_ood_gt))
        
        preds = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
        auroc_uosr, aupr, fpr95 = eval_osr(labels_uosr, preds)
        auroc_osr, aupr, fpr95 = eval_osr(labels_osr, preds)

        labels_inc_ood = np.concatenate((u_ind_gt[ind_results == ind_labels], u_ood_gt))
        preds_inc_ood = np.concatenate((ind_uncertainties[ind_results == ind_labels], ood_uncertainties), axis=0)
        auroc_inc_ood, aupr, fpr95 = eval_osr(labels_inc_ood, preds_inc_ood)

        labels_inc_inw = u_ind_gt_uosr
        preds_inc_inw = ind_uncertainties
        auroc_inc_inw, aupr, fpr95 = eval_osr(labels_inc_inw, preds_inc_inw)

        aurc_uosr, eaurc = calc_aurc_eaurc(1-preds, 1-labels_uosr)

        res_matrix_auroc_uosr[k] = auroc_uosr
        res_matrix_auroc_osr[k] = auroc_osr
        res_matrix_auroc_inc_ood[k] = auroc_inc_ood
        res_matrix_auroc_inc_inw[k] = auroc_inc_inw
        res_matrix_aurc[k] = aurc_uosr
    
    return np.mean(res_matrix_aurc), np.mean(res_matrix_auroc_uosr), np.mean(res_matrix_auroc_osr), np.mean(res_matrix_auroc_inc_ood), np.mean(res_matrix_auroc_inc_inw)


def eval_calibration(predictions, confidences, labels, M=15):
    """
    M: number of bins for confidence scores
    """
    num_Bm = np.zeros((M,), dtype=np.int32)
    accs = np.zeros((M,), dtype=np.float32)
    confs = np.zeros((M,), dtype=np.float32)
    for m in range(M):
        interval = [m / M, (m+1) / M]
        Bm = np.where((confidences > interval[0]) & (confidences <= interval[1]))[0]
        if len(Bm) > 0:
            acc_bin = np.sum(predictions[Bm] == labels[Bm]) / len(Bm)
            conf_bin = np.mean(confidences[Bm])
            # gather results
            num_Bm[m] = len(Bm)
            accs[m] = acc_bin
            confs[m] = conf_bin
    conf_intervals = np.arange(0, 1, 1/M)
    return accs, confs, num_Bm, conf_intervals

def calc_aurc_eaurc(softmax, correct):

    sort_values = sorted(zip(softmax[:], correct[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    return aurc, eaurc


def main():

    # print(f'\nResults by using all thresholds (open-set data: {args.ood_data}, backbone: {args.base_model})')
    display_data = [["Methods", "AURC-UOSR", "AUROC-UOSR", "AUROC-OSR", "AUROC-InC/OoD", "AUROC-InC/InW"], 
                    ["SoftMax"], ["KNN"], ["FS-KNN"], ["SSD"], ["FS-KNN+S"], ["FS-KNNI*S"], ["SIRC"], ["FS-KNNS"]
                    ]  # table heads and rows
    exp_dir = f'./{args.base_model}_image/{args.ood_data}/'

    FS_KNNS_file = os.path.join(exp_dir, args.baseline_results[0])
    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods(FS_KNNS_file, method='softmax')
    display_data[1].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])
    
    KNN_file = os.path.join(exp_dir, args.baseline_results[1])
    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods(KNN_file, method='knn')
    display_data[2].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])

    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods_few_shot_knns(FS_KNNS_file, method='fs-knn')
    display_data[3].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])
    
    SSD_file = os.path.join(exp_dir, args.baseline_results[2])
    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods_few_shot_ssd(SSD_file, method='ssd')
    display_data[4].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])

    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods_few_shot_knns(FS_KNNS_file, method='fs-knn+s')
    display_data[5].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])

    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods_few_shot_knns(FS_KNNS_file, method='fs-knn*s')
    display_data[6].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])

    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods_few_shot_knns(FS_KNNS_file, method='sirc')
    display_data[7].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])

    aurc_uosr, auroc_uosr, auroc_osr, auroc_inc_ood, auroc_inc_inw = eval_uncertainty_methods_few_shot_knns(FS_KNNS_file, method='fs-knns')
    display_data[8].extend(["%.2f"%(aurc_uosr * 1000), "%.2f"%(auroc_uosr * 100), "%.2f"%(auroc_osr * 100), "%.2f"%(auroc_inc_ood * 100), "%.2f"%(auroc_inc_inw * 100)])

    table = AsciiTable(display_data)
    table.inner_footing_row_border = True
    table.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center'}
    print(table.table)
    print("\n")



if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()