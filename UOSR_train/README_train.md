# **UOSR Training**

## __Testing__

To test our pre-trained models , you need to download a model file and unzip it in **log** dir. The pre-trained weights are in [Model Zoo](#model-zoo). After testing, you will get the uncertainty score result files which can be used in `../UOSR_eval` for UOSR performance evaluation. The following directory tree is for your reference to place the files.
```
work_dirs    
├── log
│    ├── SOFTMAX
│    │   └── best_bit.pth
│    │   └── bit.pth
│    │   └── train.log
│    ├── ... ... 
├── bit_pytorch
├── bit_hyperrule.py
├── data
└── ... ...
```
Evaluation for models trained from scratch:
```
cd bit_pytorch
# run the method SOFTMAX trained from scratch on GPU_ID=0
sh run_test_scratch.sh 0 SOFTMAX

```
Evaluation for models with imagenet-pretrained:
```
cd bit_pytorch
# run the method SOFTMAX on GPU_ID=0
sh run_test_pretrained.sh 0 SOFTMAX
```



## __Training__
Let's take the softmax method as an example
Train from scartch
```
cd bit_pytorch
# run the method baseline with save name SOFTMAX trained from scratch on GPU_ID=0 
sh run_train.sh 0 SOFTMAX baseline

```
Fine tune from pretrained model
```
cd bit_pytorch
# run the method baseline with save name PRETRAINED_SOFTMAX trained from scratch on GPU_ID=0
sh run_train_pretrained.sh 0 PRETRAINED_SOFTMAX baseline

```
Train from scartch with outlier data
```
cd bit_pytorch
# run the method OE with save name OE trained from scratch on GPU_ID=0 with outlier data batch size 128
bash run_oe_train.sh 0 OE OE 128

```
Fine tune with outlier data from pretrained model
```
cd bit_pytorch
# run the method OE with save name PRETRAINED_OE from pretrained model on GPU_ID=0 with outlier data batch size 128
bash run_oe_pretrained.sh 0 PRETRAINED_OE OE 128

```

## __Model Zoo__
Pre-trained weight are in [here](). If you want to get the results of Table 2, 3, 5, 6, 7, and 19 in the manuscript, simply follow `../UOSR_eval` to directly print tables in your terminal based on the uncertainty score result files we provide.
