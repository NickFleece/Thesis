#! /bin/bash

LEARNING_RATES=('0.01' '0.001' '0.0001')
BATCH_SIZES=('8' '16' '32' '64' '128')
NUM_FILTERS=('32' '64' '128' '256')

for batchsize in "${BATCH_SIZES[@]}"
do
    for filters in "${NUM_FILTERS[@]}"
    do
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters $filters --version $1-$lr-$filters-$batchsize &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.001 --batch_size $batchsize --num_filters $filters --version $1-$lr-$filters-$batchsize &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.0001 --batch_size $batchsize --num_filters $filters --version $1-$lr-$filters-$batchsize &
        wait
    done
done 