#!/bin/bash

RESULT_FILES=("ft/SoftMax.npz" \
              "ft/ODIN.npz" \
              "ft/LC.npz" \
              "ft/OpenMax.npz" \
              "ft/OLTR.npz" \
              "ft/PROSER.npz" \
              "ft/BCE.npz" \
              "ft/TCP.npz" \
              "ft/DOCTOR.npz" \
              "ft/SIRC_MSP_z.npz" \
              "ft/SIRC_MSP_res.npz" \
              "ft/SIRC_H_z.npz" \
              "ft/SIRC_H_res.npz"
            )

echo 'Results of Table 5 (training from pre-training)'
python uosr_evaluation_image.py \
       --baseline_results ${RESULT_FILES[@]} --ood_data TinyImagenet --base_model resnet50\

cd $pwd_dir
echo "Experiments finished!"