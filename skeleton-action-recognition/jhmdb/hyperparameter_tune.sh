#! /bin/bash

LEARNING_RATES=('0.01' '0.001' '0.0001')
BATCH_SIZES=('8' '16' '32' '64' '128')
NUM_FILTERS=('32' '64' '128' '256')
WEIGHT_DECAY=('0.1' '0.01' '0.001' '0.0075')

for batchsize in "${BATCH_SIZES[@]}"
do
    for weightdecay in "${WEIGHT_DECAY[@]}"
    do
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 32 --weight_decay $weightdecay --gpu 0 --version $1-0.01-32-$batchsize-$weightdecay &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 64 --weight_decay $weightdecay --gpu 0 --version $1-0.01-64-$batchsize-$weightdecay &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 128 --weight_decay $weightdecay --gpu 0 --version $1-0.01-128-$batchsize-$weightdecay &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 256 --weight_decay $weightdecay --gpu 0 --version $1-0.01-256-$batchsize-$weightdecay &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 512 --weight_decay $weightdecay --gpu 0 --version $1-0.01-512-$batchsize-$weightdecay &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 1024 --weight_decay $weightdecay --gpu 0 --version $1-0.01-1024-$batchsize-$weightdecay &
        python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 2048 --weight_decay $weightdecay --gpu 1 --version $1-0.01-2048-$batchsize-$weightdecay &
        wait
    done
done 