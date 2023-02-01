#!/bin/bash

export CUDA_HOME='/usr/local/cuda'

pwd_dir=$pwd

source activate mmaction

OOD_DATA=$1  # HMDB or MiT
RESULT_FILES=("results_scratch_hmdb/TSM_openmax.npz" \  
              "results_scratch_hmdb/TSM_dropout.npz" \                   
              "results_scratch_hmdb/TSM_bnn.npz" \                   
              "results_scratch_hmdb/TSM_softmax.npz" \  
              "results_scratch_hmdb/TSM_rpl.npz" \          
              "results_scratch_hmdb/TSM_dear.npz" \      
              "results_scratch_hmdb/TSM_BCE.npz" \
              "results_scratch_hmdb/TSM_CRL.npz" \
              "results_scratch_hmdb/TSM_DOCTOR.npz" \
              "results_scratch_hmdb/TSM_sc_MSP_z.npz" \
              "results_scratch_hmdb/TSM_sc_MSP_res.npz" \
              "results_scratch_hmdb/TSM_sc_H_z.npz" \
              "results_scratch_hmdb/TSM_sc_H_res.npz"
            )

echo 'Results of Table 6 (training from scratch)'
python uosr_evaluation_video_sc.py \
       --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"