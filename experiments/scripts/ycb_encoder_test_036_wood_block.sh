#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

./tools/test_net.py \
  --network autoencoder \
  --pretrained output/ycb_object/ycb_encoder_train/encoder_ycb_object_036_wood_block_epoch_200.checkpoint.pth \
  --dataset ycb_encoder_test \
  --cfg experiments/cfgs/ycb_encoder_036_wood_block.yml \
