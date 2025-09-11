#!/bin/bash


# For VITONHD inference
nohup env CUDA_VISIBLE_DEVICES=0 python lm_inference.py \
    --data_root_dir ./dataset/zalando-hd-resized \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --modify_final_t \
    --use_pure_to_prior \
    --[apply_lm, replacement, mcg, dps, noisy_mcg, noisy_dps, treg, dreamsampler] \
    --{apply_stochastic_noise} \
    --{unpair} \
    --save_dir ./vitonhd_results > viton_inference.log 2>&1 &

# For DressCode inference
nohup env CUDA_VISIBLE_DEVICES=1 python lm_inference.py \
    --config_path ./configs/DressCode.yaml \
    --data_root_dir ./dataset/DressCode_1024/upper \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path /home/qkrwnstj/StableVITON/ckpts/DressCode/models/DressCode.ckpt \
    --modify_final_t \
    --use_pure_to_prior \
    --[apply_lm, replacement, mcg, dps, noisy_mcg, noisy_dps, treg, dreamsampler] \
    --{apply_stochastic_noise} \
    --{unpair} \
    --save_dir ./dresscode_results > dresscode_inference.log 2>&1 &

# For SHHQ-1.0 inference
nohup env CUDA_VISIBLE_DEVICES=2 python lm_inference.py \
    --config_path ./configs/SHHQ.yaml \
    --data_root_dir ./dataset/SHHQ-1.0 \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --modify_final_t \
    --use_pure_to_prior \
    --[apply_lm, replacement, mcg, dps, noisy_mcg, noisy_dps, treg, dreamsampler] \
    --{apply_stochastic_noise} \
    --{unpair} \
    --save_dir ./shhq_results > shhq_inference.log 2>&1 &