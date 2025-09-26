#!/bin/bash


# For VITONHD inference
## Paired setting
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --data_root_dir ./dataset/zalando-hd-resized \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --modify_final_t \
    --use_pure_to_prior \
    --apply_lm \
    --save_dir ./results

## Unpaired setting
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --data_root_dir ./dataset/zalando-hd-resized \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --modify_final_t \
    --use_pure_to_prior \
    --apply_lm \
    --unpair \
    --save_dir ./results