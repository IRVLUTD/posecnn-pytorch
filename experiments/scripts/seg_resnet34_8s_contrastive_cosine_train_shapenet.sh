#!/bin/bash

set -x
set -e
export PYTHONUNBUFFERED="True"

./tools/train_net.py \
  --network seg_resnet34_8s_contrastive \
  --dataset shapenet_object_train \
  --dataset_background background_texture \
  --cfg experiments/cfgs/seg_resnet34_8s_contrastive_cosine.yml \
  --solver adam \
  --epochs 16
