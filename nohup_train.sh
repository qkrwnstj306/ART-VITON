#!/bin/bash

# For DressCode

# nohup env CUDA_VISIBLE_DEVICES=0,1 python train.py \
#  --config_name DressCode \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --save_name test \
#  --data_root_dir ./dataset/DressCode_1024/upper \
#  --resume_from_checkpoint /home/qkrwnstj/StableVITON/logs/20250627_test/models/[Train]_[epoch=119]_[train_loss_epoch=0.0421].ckpt \
#  --batch_size 8 > train.log 2>&1 &


# For DressCode and School server

# nohup env CUDA_VISIBLE_DEVICES=2 python train.py \
#  --config_name DressCode \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --save_name test \
#  --data_root_dir ./dataset/DressCode_1024/upper \
#  --resume_path /home/gpu_02/LatentVITON/ckpts/VITONHD.ckpt \
#  --batch_size 32 > train.log 2>&1 &

# resume from checkpoint
nohup env CUDA_VISIBLE_DEVICES=2 python train.py \
 --config_name DressCode \
 --transform_size shiftscale3 hflip \
 --transform_color hsv bright_contrast \
 --save_name test \
 --data_root_dir ./dataset/DressCode_1024/upper \
 --resume_path /home/gpu_02/LatentVITON/ckpts/VITONHD.ckpt \
 --resume_from_checkpoint /home/gpu_02/LatentVITON/logs/20250706_test/models/[Train]_[epoch=299]_[train_loss_epoch=0.0252].ckpt \
 --batch_size 32 > train.log 2>&1 &

 # GPU 4개면 배치는 16, 2개면 배치는 8로 설정

 # L-VITON with Large Image Encoder (= DINOv2)
# nohup env CUDA_VISIBLE_DEVICES=3 python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --resume_path ./logs/base/models/[Train]_[epoch=19]_[train_loss_epoch=0.0390].ckpt \
#  --resume_from_checkpoint ./logs/base/models/[Train]_[epoch=19]_[train_loss_epoch=0.0390].ckpt \
#  --save_name L-VITON \
#  --batch_size 16 \
#  --data_root_dir ../viton/dataset/zalando-hd-resized \
#  --seed 23 \
#  --save_every_n_epochs 5 > inference.log 2>&1 &