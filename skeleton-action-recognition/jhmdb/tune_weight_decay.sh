python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 512 --gpu 1 --verbose 2 --weight_decay 0.0053 --version $1-0.0055-0.01-512-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 512 --gpu 1 --verbose 2 --weight_decay 0.0052 --version $1-0.0052-0.01-512-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 512 --gpu 1 --verbose 2 --weight_decay 0.0051 --version $1-0.0051-0.01-512-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 512 --gpu 1 --verbose 2 --weight_decay 0.005 --version $1-0.005-0.01-512-16 &
wait