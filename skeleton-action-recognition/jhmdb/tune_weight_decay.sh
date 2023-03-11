python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 512 --gpu 0 --version $1-0.01-512-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 1024 --gpu 0 --version $1-0.01-1024-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 2048 --gpu 1 --version $1-0.01-2048-16 &
wait