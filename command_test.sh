CUDA_VISIBLE_DEVICES=0 python test.py --camera kinect --dump_dir logs/log_kn/dump_kinect --checkpoint_path logs/log_kn/minkuresunet_kinect.tar --batch_size 1 --dataset_root data/GraspNet-1Billion --infer --eval --collision_thresh -1