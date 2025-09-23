import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.lmddim import LMDDIMSampler
from cldm.model import create_model
from utils import tensor2img

from pytorch_lightning import seed_everything
from latent_code import get_code
from einops import rearrange
from solver import LMGrad

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./dataset/zalando-hd-resized")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--cfg_scale", type=float, default=1.0)

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--img_callback", action="store_true")
    parser.add_argument("--predicted_x0", action="store_true")

    # Our Method
    parser.add_argument("--code_name", type=str, default="pure") # Create novel z_0
    parser.add_argument("--apply_lm", action="store_true") # latent manipulation by controlling predicted x_0
    parser.add_argument("--use_pure_to_prior", action="store_true")
    parser.add_argument("--add_noise", action="store_true") # add noise into prior noise
    parser.add_argument("--apply_stochastic_noise", action="store_true")
    parser.add_argument("--use_ddim_inversion", action="store_true")

    # Exisiting inverse solvers
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--mcg", action="store_true")
    parser.add_argument("--dps", action="store_true")
    parser.add_argument("--noisy_mcg", action="store_true")
    parser.add_argument("--noisy_dps", action="store_true")
    parser.add_argument("--treg", action="store_true")
    parser.add_argument("--dreamsampler", action="store_true") 

    parser.add_argument("--sampling_schedule", type=str, default="uniform")
    parser.add_argument("--modify_final_t", action="store_true")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main(args):
    seed_everything(args.seed)
    
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda()
    model.eval()

    '''Choose sampler among PLMS and DDIM'''
    sampler = LMDDIMSampler(model)
    lmgrad = LMGrad()
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not args.unpair,
        is_test=True,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        # z: [BS, 4, 64, 48], c.keys(): 'c_crossattn', 'c_concat', 'first_stage_cond'
        # c_crossattn[0]: [BS, 3, 224, 244] for CLIP Image Encoder
        # c_concat[0]: [BS, 3, 512, 384] for ControlNet Input
        # first_stage_cond[BS]: [9, 64, 48] -> agnostic map: [4, 64, 48], resized mask: [1, 64, 48], pose: [4, 64, 48]
        z, c = model.get_input(batch, params.first_stage_key)
        
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code, mask = get_code(model, z, ts, c, batch, code_name=args.code_name)

        img_callback_fn = None
        predicted_x0_fn = None
        if args.img_callback:
            def save_feature_map(feature_map, filename):
                save_feature_dir = './feature_map'
                os.makedirs(save_feature_dir, exist_ok=True)
                save_path = os.path.join(save_feature_dir, f"{filename}.pt")
                torch.save(feature_map, save_path)
            def img_callback(img, time_step):
                for sample_idx, (sample, fn,  cloth_fn) in enumerate(zip(img, batch['img_fn'], batch["cloth_fn"])):
                    save_feature_map(sample, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}_{time_step}")
            img_callback_fn = img_callback

        if args.predicted_x0:
            def save_feature_map(feature_map, filename):
                save_feature_dir = './feature_map_999'
                os.makedirs(save_feature_dir, exist_ok=True)
                save_path = os.path.join(save_feature_dir, f"{filename}.pt")
                torch.save(feature_map, save_path)
            def predicted_x0(predx0):
                for sample_idx, (sample, fn,  cloth_fn) in enumerate(zip(predx0, batch['img_fn'], batch["cloth_fn"])):
                    save_feature_map(sample, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}_{999}")
                x_samples = model.decode_first_stage(predx0)
                for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
                    x_sample_img = tensor2img(x_sample)  # [0, 255]
                    to_path = opj('./feature_map_999', f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
                    cv2.imwrite(to_path, x_sample_img[:,:,::-1])
            predicted_x0_fn = predicted_x0

        img = rearrange(batch['image'], 'b h w c -> b c h w')
        agn_inversion_mask = rearrange(batch['agn_mask'], 'b h w c -> b c h w')
        agn_inversion_mask = agn_inversion_mask.round()
        measurement = img * agn_inversion_mask
        lm_mask = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1)
        lm_mask = lm_mask.round()
        lmgrad.set_input(measurement=measurement, mask=agn_inversion_mask)
        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
            mask=mask,
            x0=z,
            img_callback=img_callback_fn,
            predicted_x0=predicted_x0_fn,
            sampling_schedule=args.sampling_schedule,
            unconditional_guidance_scale=args.cfg_scale,
            apply_lm=args.apply_lm, # apply our method 
            lmgrad=lmgrad, # latent manipulation class
            lm_mask=lm_mask, # latent mask 
            use_ddim_inversion=args.use_ddim_inversion, 
            use_pure_to_prior=args.use_pure_to_prior,
            add_noise=args.add_noise, # ddpm prior + noise 
            apply_stochastic_noise=args.apply_stochastic_noise, # stochastic noise
            modify_last_time_step=args.modify_final_t,
            replacement=args.replacement,
            mcg=args.mcg, 
            dps=args.dps,
            noisy_mcg=args.noisy_mcg,
            noisy_dps=args.noisy_dps,
            treg=args.treg,
            dreamsampler=args.dreamsampler,
        )
        
        lmgrad.set_init()

        if args.img_callback:
            continue
        
        if args.predicted_x0:
            continue
        
        if samples.size(1) == 4:
            x_samples = model.decode_first_stage(samples)
        else:
            x_samples = samples
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

if __name__ == "__main__":
    args = build_args()
    main(args)
