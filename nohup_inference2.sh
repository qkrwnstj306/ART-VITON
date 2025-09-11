#!/bin/bash

if [ $1 == 1 ]; then
    nohup env CUDA_VISIBLE_DEVICES=0 python lm_inference.py \
    --data_root_dir ./dataset/zalando-hd-resized \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --use_pure_to_prior \
    --apply_lm \
    --unpair \
    --save_dir ./intro_figure_final/ours > intro_figure_final.log 2>&1 &

elif [ $1 == 2 ]; then
    nohup env CUDA_VISIBLE_DEVICES=1 python lm_inference.py \
    --data_root_dir ./dataset/zalando-hd-resized \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --code_name "whole" \
    --unpair \
    --save_dir ./intro_figure_final/stableviton > intro_figure_final2.log 2>&1 &

elif [ $1 == 3 ]; then
    nohup env CUDA_VISIBLE_DEVICES=1 python lm_inference.py \
    --data_root_dir ./dataset/zalando-hd-resized \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --use_pure_to_prior \
    --treg \
    --unpair \
    --save_dir ./intro_figure_final/treg > intro_figure_final3.log 2>&1 &
fi