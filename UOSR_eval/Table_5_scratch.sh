#!/bin/bash

RESULT_FILES=("scratch/SoftMax.npz" \
              "scratch/ODIN.npz" \
              "scratch/LC.npz" \
              "scratch/OpenMax.npz" \
              "scratch/OLTR.npz" \
              "scratch/PROSER.npz" \
              "scratch/BCE.npz" \
              "scratch/TCP.npz" \
              "scratch/DOCTOR.npz" \
              "scratch/SIRC_MSP_z.npz" \
              "scratch/SIRC_MSP_res.npz" \
              "scratch/SIRC_H_z.npz" \
              "scratch/SIRC_H_res.npz"
            )

echo 'Results of Table 5 (training from scratch)'
python uosr_evaluation_image.py \
       --baseline_results ${RESULT_FILES[@]} --ood_data TinyImagenet --base_model resnet50\

cd $pwd_dir
echo "Experiments finished!"