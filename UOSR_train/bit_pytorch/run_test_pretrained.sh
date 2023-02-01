printf "\033[1m\033[45;33m 1_ GPU  2_METHOD_NAME
            Description:Choose from  ['SOFTMAX', 'ODIN', 'LC',
                       'BCE','TCP','DOCTOR',
                       'OE','EB','ENERGY','VOS','MCD'] \033[0m\n"
                       
CUDA_VISIBLE_DEVICES=$1 python out_of_distribution_detection.py \
--model BiT-M-R50x1 \
--logdir ../test_results_saved/ \
--dataset cifar100 --datadir ../data \
--eval_every 400 --no-save --name ood_test --batch 50 --process $2 \
--ood_dataset LSUN ;
