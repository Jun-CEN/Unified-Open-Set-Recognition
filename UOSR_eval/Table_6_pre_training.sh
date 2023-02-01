#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd

source activate mmaction

RESULT_FILES=("results_ft_hmdb/TSM_openmax_ft.npz" \  
              "results/TSM_DNN_BALD_HMDB_result.npz" \                   
              "results/TSM_BNN_BALD_HMDB_result.npz" \                   
              "results_baselines/openmax/TSM_OpenMax_HMDB_result.npz" \  
              "results_baselines/rpl/TSM_RPL_HMDB_result.npz" \          
              "results_ft_hmdb/tsm_dear_ft.npz" \
              "results_ft_hmdb/TSM_BCE.npz" \
              "results_ft_hmdb/TSM_CRL.npz" \
              "results_ft_hmdb/TSM_DOCTOR.npz" \
              "results_ft_hmdb/TSM_ft_SIRC_MSP_z.npz" \
              "results_ft_hmdb/TSM_ft_SIRC_MSP_res.npz" \
              "results_ft_hmdb/TSM_ft_SIRC_H_z.npz" \
              "results_ft_hmdb/TSM_ft_SIRC_H_res.npz"
            )       

echo 'Results of Table 6 (training from pre-training)'
python uosr_evaluation_video_ft.py \
       --baseline_results ${RESULT_FILES[@]} \

cd $pwd_dir
echo "Experiments finished!"