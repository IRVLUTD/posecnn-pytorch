#!/bin/bash
	
set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./ros/test_images_prbpf.py --gpu 0 \
  --instance $2 \
  --network posecnn \
  --pretrained data/checkpoints/vgg16_ycb_object_slim_blocks_median_epoch_16.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg experiments/cfgs/ycb_object_subset_prbpf_azure.yml
