#!/bin/bash

RESULT_FILES=("scratch/SoftMax.npz" \
              "scratch/ODIN.npz" \
              "scratch/LC.npz" \
              "scratch/OpenMax.npz" \
              "scratch/OLTR.npz" \
              "scratch/PROSER.npz" \
            )

echo 'Results of Table 2'
python uosr_analysis_image_sc.py \
       --baseline_results ${RESULT_FILES[@]}

cd $pwd_dir
echo "Experiments finished!"