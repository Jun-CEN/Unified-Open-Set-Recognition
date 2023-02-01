# **Towards UOSR (Unified Open-set Recognition)**

This is the codebase of: "[The Devil is in the Wrongly-classified Samples: Towards Unified Open-set Recognition](https://openreview.net/forum?id=xLr0I_xYGAs)",  
[Jun Cen](https://cen-jun.com), Di Luan, Shiwei Zhang, Yixuan Pei, Yingya Zhang, Deli Zhao, Shaojie Shen, Qifeng Chen.  
In _International Conference on Learning Representations_ (**ICLR**), 2023.

## __Table of Contents__
1. [Overview](#overview)
1. [Datasets](#datasets)
1. [UOSR-Evaluation](#UOSR-Evaluation)
1. [UOSRTraining](#UOSR-Training)
1. [Few-shot-UOSR](#Few-shot-UOSR)
1. [Citation](#citation)


## __Overview__

We deeply analyze the **U**nified **O**pen-**s**et **R**ecognition (**UOSR**) task under different training and evaluation settings. UOSR has been proposed to reject not only unknown samples but also known but wrongly classified samples. Specifically, we first evaluate the UOSR performance of existing OSR methods and demonstrate a significant finding that the uncertainty distribution of almost all existing methods designed for OSR is actually closer to the expectation of UOSR than OSR. Second, we analyze how two training settings of OSR (i.e.,**pre-training** and **outlier exposure**) effect the UOSR. Finally, we formulate the **Few-shot Unified Open-set Recognition** setting, where only one or five samples per unknown class are available during evaluation to help identify unknown samples. Our proposed FS-KNNS method for the few-shot UOSR achieved state-of-the-art performance under all settings.

## __Datasets__

This repo uses standard image datasets, i.e., CIFAR-100 for closed set training, and TinuImagenet-resize and LSUN resize test sets as two different unknowns. Please refer to [ODIN](https://github.com/facebookresearch/odin) to download the out-of-distribution dataset. For outlier data, we choose the cleaned("debiased) image dataset [300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy). More details could be found in  [Outlier Exposure](https://github.com/hendrycks/outlier-exposure).

## __UOSR-Evaluation__

`./UOSR_eval` contains all uncertainty score result files to reproduce Table 2, 3, 5, 6, 7, and 19 in the manuscript. This codebase is like a evalaution server to calculate the UOSR performance based on the uncetainty score result files. Simply follow the __README_eval.md__ in `./UOSR_eval` to get all table results directly in the terminal.

## __UOSR-Training__
`./UOSR_train` provide the code about the how to train the model with the methods mentioned in the manuscript. Readers may refer to __README_train.md__ in `./UOSR_train` for details.

## __Few-shot-UOSR__
`./UOSR_few_shot` provide the code about the how to conduct few-shot UOSR evaluation. Readers may refer to __README_few_shot.md__ in `./UOSR_few_shot` for details.

## __Citation__

If you find the code useful in your research, please cite:

```
@inproceedings{
jun2023devil
title={The Devil is in the Wrongly-classified Samples: Towards Unified Open-set Recognition},
author={Jun Cen, Di Luan, Shiwei Zhang, Yixuan Pei, Yingya Zhang, Deli Zhao, Shaojie Shen, Qifeng Chen},
booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
year={2023},
}

```

## __License__
See [Apache-2.0 License](https://github.com//LICENSE)
## __Acknowledgement__

This repo contains modified codes from:

- [Learning Confidence Estimates for Neural Networks](https://github.com/uoguelph-mlrg/confidence_estimation): for implementation of baseline method [LC ](https://arxiv.org/abs/1802.04865).
- [Out-of-Distribution Detector for Neural Networks](https://github.com/facebookresearch/odin): for implementation of baseline method [ODIN ](https://arxiv.org/abs/1706.02690).
- [Learning Placeholders for Open-Set Recognition](https://github.com/zhoudw-zdw/CVPR21-Proser): for implementation of baseline method [PROSER ](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Learning_Placeholders_for_Open-Set_Recognition_CVPR_2021_paper.pdf).
- [Deep Anomaly Detection with Outlier Exposure](https://github.com/hendrycks/outlier-exposure): for implementation of baseline method [OE ](https://arxiv.org/abs/1812.04606).
- [Evidential Deep Learning for Open Set Action Recognition](https://github.com/Cogito2012/DEAR): for implementation of methods in the video domain.

We sincerely thank the owners of all these great repos!
