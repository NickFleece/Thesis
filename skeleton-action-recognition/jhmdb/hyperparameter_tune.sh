#! /bin/bash

LEARNING_RATES=('0.1' '0.01' '0.001' '0.0001' '0.00001' '0.000001')
BATCH_SIZES=('8' '16' '32' '64' '128' '256')

for lr in "${LEARNING_RATES[@]}"
do
    for batchsize in "${BATCH_SIZES[@]}"
    do

        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate $lr --batch_size $batchsize --version $1-$lr-$batchsize

    done
done
