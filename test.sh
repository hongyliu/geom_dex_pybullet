#!/bin/bash
CUDA_VISIBLE_DEVICES=2 nohup python test_classification.py --use_normals --log_dir pointnet2_cls_ssg_normal > test_classification.log 2>&1 &