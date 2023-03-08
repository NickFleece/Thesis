#! /bin/bash

LEARNING_RATES=('0.01' '0.001' '0.0001')
BATCH_SIZES=('8' '16' '32' '64' '128')
NUM_FILTERS=('32' '64' '128' '256')

for batchsize in "${BATCH_SIZES[@]}"
do
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 32 --gpu 0 --version $1-0.01-32-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.001 --batch_size $batchsize --num_filters 32 --gpu 1 --version $1-0.001-32-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.0001 --batch_size $batchsize --num_filters 32 --gpu 1 --version $1-0.0001-32-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 64 --gpu 0 --version $1-0.01-64-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.001 --batch_size $batchsize --num_filters 64 --gpu 0 --version $1-0.001-64-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.0001 --batch_size $batchsize --num_filters 64 --gpu 1 --version $1-0.0001-64-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 128 --gpu 0 --version $1-0.01-128-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.001 --batch_size $batchsize --num_filters 128 --gpu 1 --version $1-0.001-128-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.0001 --batch_size $batchsize --num_filters 128 --gpu 1 --version $1-0.0001-128-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 256 --gpu 0 --version $1-0.01-256-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.001 --batch_size $batchsize --num_filters 256 --gpu 0 --version $1-0.001-256-$batchsize &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.0001 --batch_size $batchsize --num_filters 256 --gpu 1 --version $1-0.0001-256-$batchsize &
    wait
done 