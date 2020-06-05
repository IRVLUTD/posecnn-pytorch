#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py \
  --network seg_resnet34_8s_embedding \
  --pretrained output/ycb_object/shapenet_object_train/seg_resnet34_8s_embedding_multi_epoch_$2.checkpoint.pth  \
  --dataset shapenet_object_test \
  --dataset_background background_texture \
  --cfg experiments/cfgs/seg_resnet34_8s_embedding.yml
