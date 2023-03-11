#! /bin/bash

python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 128 --gpu 0 --version $1-0.01-128-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 256 --gpu 0 --version $1-0.01-256-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 512 --gpu 0 --version $1-0.01-512-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 1024 --gpu 0 --version $1-0.01-1024-16 &
wait