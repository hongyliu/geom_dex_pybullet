#!/bin/bash
category=$1
case ${category} in
1)
  CUDA_VISIBLE_DEVICES=2 nohup python train_classification.py --model pointnet2_cls_ssg --use_normals --num_category 10 --num_point 2048 --log_dir pointnet2_cls_ssg_normal > train_classification.log 2>&1 &
  ;;
2)
  CUDA_VISIBLE_DEVICES=3 nohup python train_pointnet.py --expID 2 --use_gpu > train_pointnet.log 2>&1 &
  ;;
3)
  CUDA_VISIBLE_DEVICES=1 nohup python run_ddpg.py --expID 3 --use_gpu --video_count 0 --n_cycles 40000 --serial --point_cloud --pointnet_load_path 2 --no_save_buffer > run_ddpg.log 2>&1 &
  ;;
*)
  echo "arguments should be among list [1, 2, 3]"
  ;;
esac