# **Few-shot UOSR Evaluation**

## **Introduction**
In this codebase, we provide how to get the uncertainty score result files of all few-shot USOR methods. Take 5-shot and ResNet50 as an example, we aim to generate _resnet50_fs_knns_5shot.npz/resnet50_fs_ssd_5shot.npz/resnet50_knn.npz_
## **Dataset preparation**
We need to use the validation set of TinyImageNet as OoD reference samples, so we first transfer the validation set of TinyImageNet into the tensor file for convinience.
```shell
python fileio.py
```
You should get the _val_dataset.pkl_ which contains the validation set of TinyImageNet.
## **Step 1**
Obtain the feature representation of all training samples and test samples and store the result file.
```shell
python test_fs_R50.py
```
The path of pretrained model weight is defined at Line 210. This model should be trained with CIFAR100 dataset based on BiT-M pretrained weights. The checkpoint of trained mode is [here](). Or you can use the code in `../UOSR_train` to train this model. The path of result file is at Line 408 and the default name is _R50__ood__tinyImageNet_resize__mode__baseline.npz_

## **Step 2**
We aim to obtain the _resnet50_knn.npz_ in step 2. Note that KNN is not a few-shot method, but it is the base of FS-KNN and FS-KNNS so we still evaluate this method. Change the path in Line 38 to the npz file obtained in step 1, and then
```shell
python knn_evaluate.py
```
The result file path is in Line 79.

## **Step 3**
SSD is the baseline method of few-shot UOSR, and we obtain the _resnet50_fs_ssd_5shot.npz_ in step 3. Change the path in Line 38 to the npz file obtained in step 1, and then 
```shell
python ssd_evaluate.py
```
The result file path is in Line 79. Note that change the Line 143 to 50 if you want to get the 1-shot result.

## **Step 4**
In step 4, we obtain the _resnet50_fs_knns_5shot.npz_. Change the path in Line 38 to the npz file obtained in step 1, and then 
```shell
python fs_evaluate.py
```
The result file path is in Line 79. Note that change the Line 130 to 50 if you want to get the 1-shot result.

## **Evaluate for few-shot UOSR**
The user should get _resnet50_knn.npz/resnet50_fs_ssd_5shot.npz/resnet50_fs_knns_5shot.npz_ in step 2, 3, and 4 respectively. Then put the path of these results into _../UOSR_eval/Table_7_5_shot.sh_ and then
```shell
cd ../UOSR_eval
./Table_7_5_shot.sh
```
You can obtain the Table 7 in the manuscript.