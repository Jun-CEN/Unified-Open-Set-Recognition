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
    parser.add_argument('--base_model', default='tsm', help='the backbone model name')
    parser.add_argument('--ood_data', default='HMDB', help='the name of OOD dataset.')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1])
    parser.add_argument('--baseline_results', nargs='+', help='the testing results files.')
    parser.add_argument('--UOSAR', default='True', help='whether UOSAR or not')
    args = parser.parse_args()
    return args

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

def eval_uncertainty_methods(result_file, threshold=-1):
    assert os.path.exists(result_file), "File not found! Run ood_detection first!\n%s"%(result_file)
    # load the testing results
    results = np.load(result_file, allow_pickle=True)
    if "bnn" in result_file or "dear" in result_file:
        ind_uncertainties = results['ind_unctt'][:,0]  # (N1,)
        ood_uncertainties = results['ood_unctt'][:,0]  # (N2,)
    else:
        ind_uncertainties = results['ind_unctt']  # (N1,)
        ood_uncertainties = results['ood_unctt']  # (N2,)
    ind_results = results['ind_pred']  # (N1,)
    ood_results = results['ood_pred']  # (N2,)
    ind_labels = results['ind_label']
    ood_labels = results['ood_label']

    # close-set accuracy (multi-class)
    acc = accuracy_score(ind_labels, ind_results)

    repeated_clss = [35, 29, 15, 26, 30, 34, 43, 31]
    index_repeated = np.zeros_like(ood_results)
    for i in repeated_clss:
        index_repeated[ood_labels == i] = 1
    index_no_repeated = 1 - index_repeated
    ood_uncertainties = ood_uncertainties[index_no_repeated==1]
    ood_results = ood_results[index_no_repeated==1]
    ood_labels = ood_labels[index_no_repeated==1]


    # open-set evaluation (binary class)
    if threshold > 0:
        uncertain_sort = np.sort(ind_uncertainties)[::-1]
        N = ind_uncertainties.shape[0]
        topK = N - int(N * 0.85)
        threshold = uncertain_sort[topK-1]
        preds = np.concatenate((ind_results, ood_results), axis=0)
        uncertains = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
        preds[uncertains > threshold] = 1
        preds[uncertains <= threshold] = 0
    else:
        preds = np.concatenate((ind_uncertainties, ood_uncertainties), axis=0)
    
    u_ind_gt = np.zeros_like(ind_labels)
    u_ood_gt = np.ones_like(ood_labels)

    u_ind_gt_uosr = u_ind_gt.copy()
    u_ind_gt_uosr[ind_results != ind_labels] = 1

    labels_uosr = np.concatenate((u_ind_gt_uosr, u_ood_gt))
    labels_osr = np.concatenate((u_ind_gt, u_ood_gt))
    
    auroc_uosr, aupr, fpr95 = eval_osr(labels_uosr, preds)
    auroc_osr, aupr, fpr95 = eval_osr(labels_osr, preds)

    labels_inc_ood = np.concatenate((u_ind_gt[ind_results == ind_labels], u_ood_gt))
    preds_inc_ood = np.concatenate((ind_uncertainties[ind_results == ind_labels], ood_uncertainties), axis=0)
    auroc_inc_ood, aupr, fpr95 = eval_osr(labels_inc_ood, preds_inc_ood)

    labels_inc_inw = u_ind_gt_uosr
    preds_inc_inw = ind_uncertainties
    auroc_inc_inw, aupr, fpr95 = eval_osr(labels_inc_inw, preds_inc_inw)

    labels_inw_ood = np.concatenate((u_ind_gt[ind_results != ind_labels], u_ood_gt))
    preds_inw_ood = np.concatenate((ind_uncertainties[ind_results != ind_labels], ood_uncertainties), axis=0)
    auroc_inw_ood, aupr, fpr95 = eval_osr(labels_inw_ood, preds_inw_ood)

    return auroc_inc_ood, auroc_inc_inw, auroc_inw_ood, auroc_osr, auroc_uosr

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
    display_data = [["Methods", "InC/OoD", "InC/InW", "InW/OoD", "OSR", "UOSR"], 
                    ["OpenMax"], ["MC Dropout"], ["BNN SVI"], ["SoftMax"], ["RPL"], ["DEAR"]
                    ]  # table heads and rows
    exp_dir = './tsm_video'

    for i in range(6):
        result_path = os.path.join(exp_dir, args.baseline_results[i])
        InC_OoD, InC_InW, InW_OoD, OSR, UOSR = eval_uncertainty_methods(result_path, threshold=-1)
        display_data[i+1].extend(["%.2f"%(InC_OoD * 100), "%.2f"%(InC_InW * 100), "%.2f"%(InW_OoD * 100), "%.2f"%(OSR * 100), "%.2f"%(UOSR * 100)])


    table = AsciiTable(display_data)
    table.inner_footing_row_border = True
    table.justify_columns = {0: 'left', 1: 'center', 2: 'center', 3: 'center', 4: 'center', 5: 'center'}
    print(table.table)
    print("\n")

if __name__ == "__main__":

    np.random.seed(123)
    args = parse_args()

    main()