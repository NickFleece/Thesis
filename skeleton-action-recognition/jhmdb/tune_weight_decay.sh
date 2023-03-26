python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 128 --gpu 1 --verbose 2 --weight_decay 0.005 --version $1-0.005-0.01-128-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 128 --gpu 1 --verbose 2 --weight_decay 0.004 --version $1-0.004-0.01-128-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 256 --gpu 1 --verbose 2 --weight_decay 0.005 --version $1-0.005-0.01-256-16 &
python skeleton_model.py --drive_dir /comm_dat/nfleece/JHMDB --learning_rate 0.01 --batch_size 16 --num_filters 256 --gpu 1 --verbose 2 --weight_decay 0.004 --version $1-0.004-0.01-256-16 &
wait