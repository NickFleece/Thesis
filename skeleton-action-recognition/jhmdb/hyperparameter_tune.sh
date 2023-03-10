#! /bin/bash

LEARNING_RATES=('0.01' '0.001' '0.0001')
BATCH_SIZES=('8' '16' '32' '64' '128')
NUM_FILTERS=('32' '64' '128' '256')

for batchsize in "${BATCH_SIZES[@]}"
do
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 32 --weight_decay 0.001 --gpu 0 --version $1-0.01-32-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 64 --weight_decay 0.001 --gpu 0 --version $1-0.01-64-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 128 --weight_decay 0.001 --gpu 0 --version $1-0.01-128-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 256 --weight_decay 0.001 --gpu 0 --version $1-0.01-256-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 512 --weight_decay 0.001 --gpu 0 --version $1-0.01-512-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 1024 --weight_decay 0.001 --gpu 0 --version $1-0.01-1024-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 2048 --weight_decay 0.001 --gpu 0 --version $1-0.01-2048-$batchsize-0.001 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 32 --weight_decay 0.01 --gpu 1 --version $1-0.01-32-$batchsize-0.01 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 64 --weight_decay 0.01 --gpu 1 --version $1-0.01-64-$batchsize-0.01 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 128 --weight_decay 0.01 --gpu 1 --version $1-0.01-128-$batchsize-0.01 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 256 --weight_decay 0.01 --gpu 1 --version $1-0.01-256-$batchsize-0.01 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 512 --weight_decay 0.01 --gpu 1 --version $1-0.01-512-$batchsize-0.01 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 1024 --weight_decay 0.01 --gpu 1 --version $1-0.01-1024-$batchsize-0.01 &
    python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size $batchsize --num_filters 2048 --weight_decay 0.01 --gpu 1 --version $1-0.01-2048-$batchsize-0.01 &
    wait
done 