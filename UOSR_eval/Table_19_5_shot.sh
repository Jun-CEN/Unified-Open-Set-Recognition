#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_few_shot/TSM_fs_knns_5shot.npz" \
              "results_few_shot/TSM_knn.npz" \  
              "results_few_shot/TSM_fs_ssd_5shot.npz" \
            )

echo 'Results of Table 19 (5-shot)'
python uosr_few_shot_video.py \
       --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"