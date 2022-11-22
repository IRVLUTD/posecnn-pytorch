#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./ros/test_images_prbpf.py --gpu 0 \
  --instance $2 \
  --network posecnn \
  --pretrained data/checkpoints/ycb_object/vgg16_ycb_object_detection_self_supervision_epoch_8.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_prbpf_realsense_ycb.yml
