#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/ycb_video_train.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py \
  --network posecnn \
  --pretrained output/ycb_video/ycb_video_debug/vgg16_ycb_video_epoch_80.checkpoint.pth \
  --dataset ycb_video_debug \
  --cfg experiments/cfgs/ycb_video.yml \
  --cad data/YCB_Video/models.txt \
  --pose data/YCB_Video/poses.txt \
  --solver sgd \
  --epochs 1000 \
  --startepoch 80