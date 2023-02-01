#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("few_shot/resnet50_fs_knns_5shot.npz" \
              "few_shot/resnet50_knn.npz" \  
              "few_shot/resnet50_fs_ssd_5shot.npz" \
            )

echo 'Results of Table 7 (5-shot)'
python uosr_few_shot_image.py \
       --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"