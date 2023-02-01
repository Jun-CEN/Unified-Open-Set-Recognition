import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# this is for plotting and latex tables
METRIC_NAME_MAPPING = {
    "confidence": "MSP",
    "doctor": "DOCTOR",
    "entropy": "$-\mathcal{H}$",
    "max_logit": "Max Logit",
    "energy": "Energy",
    "feature_norm": "$||\\b z||_1$", 
    "residual": "Residual",
    "gradnorm": "Gradnorm",
    "vim":"ViM",
    "SIRC_MSP_z": "(MSP,$||\\b z||_1$)",
    "SIRC_MSP_res": "(MSP,Res.)",
    "SIRC_doctor_z": "(DR,$||\\b z||_1$)",
    "SIRC_doctor_res": "(DR,Res.)",
    "SIRC_H_res": "($-\mathcal{H}$,Res.)",
    "SIRC_H_z": "($-\mathcal{H}$,$||\\b z||_1$)",
    "mahalanobis": "Mahal",
}

def get_metric_name(unc):
    if unc in METRIC_NAME_MAPPING:
        return METRIC_NAME_MAPPING[unc]
    else:
        return unc


def entropy(probs: torch.Tensor, dim=-1):
    "Calcuate the entropy of a categorical probability distribution."
    log_probs = probs.log()
    ent = (-probs*log_probs).sum(dim=dim)
    return ent


def sirc(s1, s2, a,b, s1_max=1):
    "Combine 2 confidence metrics with SIRC."
    # use logarithm for stability
    soft = (s1_max - s1).log()
    additional = torch.logaddexp(
        torch.zeros(len(s2)),
        -b * (s2 - a)
    )
    return - soft - additional # return as confidence

def uncertainties(
    logits: torch.Tensor, 
    features=None, 
    gmm_params=None, vim_params=None,
    stats=None
) -> dict:
    """Calculate uncertainty measures given tensors of logits and features.
    Due to legacy reasons, confidence scores will be dealt with as 
    uncertainties throughout, apart from MSP. Returns a dictionary where
    each key is a specific score over the data.
    """

    # increase precision
    logits = logits.type(torch.DoubleTensor)



    probs = logits.softmax(dim=-1)
    max_logit = -logits.max(dim=-1).values
    conf = probs.max(dim=-1).values
    ent = entropy(probs, dim=-1)
    doctor = -(probs**2).sum(dim=-1)
    energy = -torch.logsumexp(logits, dim=-1)

    # logit and softmax based scores

    uncertainty = {
        'confidence': conf,
        "doctor": doctor,
        'entropy': ent, 
        "max_logit": max_logit,
        "energy": energy,
    }


    if features is not None:
        # make negative so that higher is more uncertain
        feature_norm = torch.norm(features, p=1, dim=-1)
        uncertainty["feature_norm"] = - feature_norm
        uncertainty["gradnorm"] = - torch.norm(
            probs-1/probs.shape[-1], p=1, dim=-1
        ) * feature_norm

        
        if gmm_params is not None:
            
            distance_list = []
            class_means = torch.stack(gmm_params["class_means"], dim=0)
            print("calculating mahalanobis distances")
            precision = gmm_params["precision"]
            # batched due to system memory contraints
            batched_features = DataLoader(
                # for broadcasting, additional dimension over classes
                features.unsqueeze(dim=1),  
                drop_last=False, shuffle=False, batch_size=64,
                num_workers=4
            )

            # send stuff in batches to the gpu
            # Mahalanobis distance can be parallelised 
            for feature in tqdm(batched_features):

                # broadcast over classes
                norm_feature = feature.to("cuda") - class_means.to("cuda")

                # insert dims for matmul
                norm_feature = norm_feature.unsqueeze(dim=-1)
                # uncertainty comes from closest class mean

                precision = precision.to("cuda")
                # vector matrix vector multiply
                # over batch and class dimension
                distance_list.append(
                    (
                        norm_feature.transpose(-1, -2) 
                        @ precision 
                        @ norm_feature
                    ).to("cpu")
                )

            mahal_ds = torch.cat(distance_list, dim=0)
            mahal_ds = mahal_ds.squeeze(dim=-1).squeeze(dim=-1).min(
                dim=1
            ).values # over classes
            uncertainty["mahalanobis"] = mahal_ds

        if vim_params is not None:
            
            # for broadcasting
            centered_feats = (features - vim_params["u"]).unsqueeze(dim=-1)   
                        
            vlogits = torch.norm(
                (
                    vim_params["R"].T @ centered_feats
                ).squeeze(dim=-1), 
                p=2, dim=-1 # l2-norm
            ) * vim_params["alpha"]

            # subtract energy
            uncertainty["residual"] = vlogits
            uncertainty["vim"] = vlogits - torch.logsumexp(logits, dim=-1)

        if stats is not None:

            # feature norm
            a, b = get_sirc_params(stats["feature_norm"])

            uncertainty[f"SIRC_H_z"] = - sirc(
                -ent, feature_norm, a, b, s1_max=0
            )
            uncertainty[f"SIRC_MSP_z"] = - sirc(
                conf, feature_norm, a, b, s1_max=1
            )
            uncertainty[f"SIRC_doctor_z"] = - sirc(
                -doctor, feature_norm, a, b, s1_max=1
            )

            if vim_params is not None:
                a, b = get_sirc_params(stats["residual"])

                uncertainty[f"SIRC_H_res"] = -sirc(
                    -ent, -vlogits, a, b, s1_max=0
                )
                uncertainty[f"SIRC_MSP_res"] = -sirc(
                    conf, -vlogits, a, b, s1_max=1
                )

                uncertainty[f"SIRC_doctor_res"] = -sirc(
                    -doctor, -vlogits, a, b, s1_max=1
                )

    return uncertainty


def metric_stats(metrics):
    # get the stats of a dictionary of metrics
    stats = {}

    # mean, mode, std for now
    for metric in metrics:
        metric_data = np.array(metrics[metric])
        metric_stats = {}
        metric_stats["mean"] = np.mean(metric_data)
        metric_stats["std"] = np.std(metric_data)
        density_estimate = gaussian_kde(metric_data)

        # initialise optimiser at mean
        metric_stats["mode"] = minimize(
            lambda z:-density_estimate.pdf(z), metric_stats["mean"]
        ).x[0]

        # get quantile
        quant = 0.99 if metric != "confidence" else 0.01
        metric_stats["quantile_99"] = np.quantile(metric_data, quant)
        stats[metric] = metric_stats
    return stats

def get_sirc_params(unc_stats):

    # remember that the values are negative
    # MARK: REMOVE FIRST - BEFORE MEAN
    a = - unc_stats["mean"] - 3 * unc_stats["std"]

    # investigate effect of varying b
    b =1/unc_stats["std"] 
    print(a, b)
    return a, b


class TopKError(nn.Module):
    """
    Calculate the top-k error rate of a model. 
    """

    def __init__(self, k=1, percent=True):
        super().__init__()
        self.k = k
        self.percent = percent

    def forward(self, labels, outputs):
        # get rid of empty dimensions
        if type(labels) == np.ndarray:
            labels = torch.tensor(labels)
        if type(outputs) == np.ndarray:
            outputs = torch.tensor(outputs)
        labels, outputs = labels.squeeze(), outputs.squeeze()
        _, topk = outputs.topk(self.k, dim=-1)
        # same shape as topk with repeated values in class dim
        labels = labels.unsqueeze(-1).expand_as(topk)
        acc = torch.eq(labels, topk).float().sum(dim=-1).mean()
        err = 1 - acc
        err = 100 * err if self.percent else err
        return err.item()


# printing --------------------------------------------------------------------

def print_results(results: dict):
    """Print the results in a results dictionary."""
    print("="*80)
    for k, v in results.items():
        if type(v) == float:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    print("="*80)


def get_ood_metrics_from_combined(metrics, domain_labels):
    """Extract metrics only related to OOD data from combined data."""
    OOD_metrics = {}
    for key, metric in metrics.items():
        OOD_metrics[key] = metric[domain_labels == 1]

    return OOD_metrics


# code adapted from
# https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/misc_detection.py
# https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/rejection.py

# OOD detection ---------------------------------------------------------------

def fpr_at_recall(labels, scores, recall_level):
    """Get the false positive rate at a specific recall."""

    # positive is ID now
    labels = ~labels.astype(bool)
    scores = -scores
    precision, recall, thresholds = precision_recall_curve(
            labels, scores
    )

    # postive if >= threshold, recall and precision have an extra value
    # for 0 recall (all data classified as negative) at the very end
    # get threshold closest to specified (e.g.95%) recall
    cut_off = np.argmin(np.abs(recall-recall_level))
    t = thresholds[cut_off]


    negatives = ~labels 

    # get positively classified samples and filter
    fps = np.sum(negatives * (scores >= t))

    return fps/np.sum(negatives)


def detect_results(
    domain_labels,
    metrics,
    mode="ROC",
):
    """Evaluate OOD data detection using different uncertainty metrics."""

    # iterate over different metrics (e.g. mutual information)
    assert mode in ["ROC", "FPR@95"]
    domain_labels = np.asarray(domain_labels)
    results = {"mode": mode}
    for key in metrics.keys():
        pos_label = 1

        # for legacy reasons MSP is referred to as confidence
        # it is also the only score stored as a confidence rather than an 
        # uncertainty
        if key == 'confidence':
            pos_label = 0

        results[key] = detect(
            domain_labels,
            metrics[key],
            mode=mode,
            pos_label=pos_label
        )

    return results


def detect(
    domain_labels,
    metric,
    mode,
    pos_label=1, 
):
    """Calculate the AUROC or FPR@95.
    Note that input positive labels are for OOD/misclassifications due 
    to legacy reasons. They are flipped later on as we want correct 
    classifications to be the positive class.
    """
    scores = metric
    scores = np.asarray(scores, dtype=np.float128)

    # just flip MSP scores to uncertainty
    if pos_label != 1:
        scores *= -1.0
        
    # if there is overflow just clip to highest float
    scores = np.nan_to_num(scores) 

    # receiver operating characteristic
    if mode == 'ROC':
        # symmetric so don't care 
        roc_auc = roc_auc_score(domain_labels, scores)

        # percent
        return roc_auc * 100

    elif mode == "FPR@95":
        recall_level = 0.95
        # note that the labels are reversed
        # OOD is positive for PR
        fpr = fpr_at_recall(domain_labels, scores, recall_level)
        # percent
        return fpr * 100


