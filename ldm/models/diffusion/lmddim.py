"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import grad

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor
import function_freq as ffq

class LMDDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)
    
    @torch.no_grad()
    def inversion(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, predicted_x0=None,
                      apply_lm=False, lmgrad=None, lm_mask=None, use_ddim_inversion=False):
        b, *_, device = *x0.shape, x0.device

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]
            
        time_range = range(0,timesteps) if ddim_use_original_steps else timesteps
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Inversion with {total_steps} timesteps")
            
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        img = x0
        for i, step in enumerate(iterator):
            
            if i >= total_steps - 1:
                continue
            
            index = i
            next_index = index + 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_t_next = torch.full((b, 1, 1, 1), alphas[next_index], device=device)
        
            with torch.no_grad():
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                    quantize_denoised=quantize_denoised, temperature=temperature,
                                    noise_dropout=noise_dropout, score_corrector=score_corrector,
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                    unconditional_conditioning=unconditional_conditioning,
                                    dynamic_threshold=dynamic_threshold,
                                    apply_lm=False, lmgrad=lmgrad, lm_mask=lm_mask, x0=x0, get_eps=True)
                pred_x0, eps = outs
                
                img = (img - (1 - a_t).sqrt() * eps) * (a_t_next.sqrt() / a_t.sqrt()) + (1 - a_t_next).sqrt() * eps
            
        return img * lm_mask + (1. - lm_mask) * torch.randn(shape, device=device)
            
    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               predicted_x0=None,
               sampling_schedule="uniform",
               apply_lm=False,
               lmgrad=None,
               lm_mask=None,
               use_ddim_inversion=False,
               use_pure_to_prior=False,
               add_noise=False,
               apply_stochastic_noise=False,
               replacement=False, mcg=False, dps=False, noisy_mcg=False, noisy_dps=False, treg=False, dreamsampler=False,
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=sampling_schedule, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates, cond_output_dict = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    predicted_x0=predicted_x0,
                                                    apply_lm=apply_lm, 
                                                    lmgrad=lmgrad,
                                                    lm_mask=lm_mask,
                                                    use_ddim_inversion=use_ddim_inversion,
                                                    use_pure_to_prior=use_pure_to_prior,
                                                    add_noise=add_noise,
                                                    apply_stochastic_noise=apply_stochastic_noise,
                                                    replacement=replacement, mcg=mcg, dps=dps, noisy_mcg=noisy_mcg, noisy_dps=noisy_dps, treg=treg, dreamsampler=dreamsampler,
                                                    )
        return samples, intermediates, cond_output_dict

    @torch.no_grad()
    def pure_to_prior(self, cond, x_T=None, ddim_use_original_steps=True, quantize_denoised=False, x0=None,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      apply_lm=False, lmgrad=None, lm_mask=None, add_noise=False):
        
        last_time_step = 999
        b, C, H, W= x_T.shape
        device = x_T.device
        index = last_time_step
        ts = torch.full((b,), last_time_step, device=device, dtype=torch.long)
        size = (b, C, H, W)
        img = torch.randn(size, device=device) # pure

        outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                quantize_denoised=quantize_denoised, temperature=temperature,
                                noise_dropout=noise_dropout, score_corrector=score_corrector,
                                corrector_kwargs=corrector_kwargs,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning,
                                dynamic_threshold=dynamic_threshold,
                                apply_lm=apply_lm, lmgrad=lmgrad, lm_mask=lm_mask, x0=x0)
        sample, pred_x0, _ = outs
        img = sample # x_{t-1}
        noise = noise_like(size, device, False)

        if add_noise:
            img = (1. - self.betas[last_time_step]).sqrt() * img + self.betas[last_time_step].sqrt() * noise

        # img = lm_mask * x_T + (1. - lm_mask) * img
        
        return img

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None, predicted_x0=None,
                      apply_lm=False, lmgrad=None, lm_mask=None, use_ddim_inversion=False, use_pure_to_prior=False, add_noise=False, apply_stochastic_noise=False,
                      replacement=False, mcg=False, dps=False, noisy_mcg=False, noisy_dps=False, treg=False, dreamsampler=False):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        elif use_ddim_inversion: # 0 -> T
            img = self.inversion(cond, shape, callback=callback, img_callback=img_callback, quantize_denoised=quantize_denoised,
                                mask=mask, x0=x0, ddim_use_original_steps=False, noise_dropout=noise_dropout, temperature=temperature,
                                score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, x_T=x_T, log_every_t=log_every_t, unconditional_guidance_scale=unconditional_guidance_scale,
                                unconditional_conditioning=unconditional_conditioning, dynamic_threshold=dynamic_threshold, ucg_schedule=ucg_schedule,
                                predicted_x0=predicted_x0, apply_lm=apply_lm, lmgrad=lmgrad, lm_mask=lm_mask, use_ddim_inversion=use_ddim_inversion)
        else:
            img = x_T
        
        """Pure to Prior"""
        if use_pure_to_prior:
            img = self.pure_to_prior(cond, x_T=x_T, ddim_use_original_steps=True, quantize_denoised=quantize_denoised, x0=x0,
                        temperature=temperature, noise_dropout=noise_dropout, score_corrector=score_corrector, corrector_kwargs=corrector_kwargs,
                        unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning, dynamic_threshold=dynamic_threshold,
                        apply_lm=False, lmgrad=lmgrad, lm_mask=lm_mask, add_noise=add_noise)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")
            
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if img_callback: img_callback(img, i)

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]
            with torch.no_grad() if not (mcg or dps or noisy_mcg or noisy_dps) else torch.enable_grad():
                if mcg or dps or noisy_mcg or noisy_dps:
                    assert img.size(0) == 1
                    img = img.requires_grad_()
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        dynamic_threshold=dynamic_threshold,
                                        treg=treg, dreamsampler=dreamsampler,
                                        apply_lm=apply_lm, lmgrad=lmgrad, lm_mask=lm_mask, x0=x0, apply_stochastic_noise=apply_stochastic_noise)
                sample, pred_x0, cond_output_dict = outs

                if mask is not None:
                    mask = mask.round()

                alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
                a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)

                
                if (mcg or dps) and (index % 3 == 0 or index == 0):
                    norm = torch.norm((x0 * mask) - (pred_x0 * mask))
                    norm_grad = grad(outputs=norm, inputs=img)[0]
                    img = sample - norm_grad
                elif (noisy_mcg or noisy_dps) and (index % 3 == 0 or index == 0):
                    if len(time_range) - 1 != i:
                        prev_ts = torch.full((b,), time_range[i+1], device=device, dtype=torch.long)
                        img_orig = self.model.q_sample(x0, prev_ts)
                    else:
                        img_orig = x0
                    norm = torch.norm((img_orig * mask) - (sample * mask))
                    norm_grad = grad(outputs=norm, inputs=img)[0]
                    img = sample - norm_grad
                else:
                    img = sample

                if mask is not None and (replacement or mcg or noisy_mcg) and (index % 3 == 0 or index == 0):
                    assert x0 is not None
                    if replacement or mcg:
                        if len(time_range) - 1 != i:
                            prev_ts = torch.full((b,), time_range[i+1], device=device, dtype=torch.long)
                            img_orig = self.model.q_sample(x0, prev_ts)
                        else:
                            img_orig = x0
                    img = img_orig * mask + (1. - mask) * img
                
            img.detach_()

            if callback: callback(i)
            if predicted_x0 and step == 999: predicted_x0(pred_x0)
            
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        if cond_output_dict is not None:
            cond_output = cond_output_dict["cond_output"]     
            if self.model.use_noisy_cond:
                b = cond_output.shape[0]

                alphas = self.model.alphas_cumprod if ddim_use_original_steps else self.ddim_alphas
                alphas_prev = self.model.alphas_cumprod_prev if ddim_use_original_steps else self.ddim_alphas_prev
                sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if ddim_use_original_steps else self.ddim_sqrt_one_minus_alphas

                device = cond_output.device
                a_t = torch.full((b, 1, 1, 1), alphas[0], device=device)
                a_prev = torch.full((b, 1, 1, 1), alphas_prev[0], device=device)
                sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[0], device=device)

                c = cond_output_dict["cond_input"]
                e_t = cond_output
                pred_c0 = (c - sqrt_one_minus_at * e_t) / a_t.sqrt()
                dir_ct = (1. - a_prev - sigma_t**2).sqrt() * e_t
                noise = sigma_t * noise_like(c.shape, device, False) * temperature

                if noise_dropout > 0.:
                    noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                cond_output = a_prev.sqrt() * pred_c0 + dir_ct + noise               
                cond_output_dict[f"cond_sample"] = cond_output
        return img, intermediates, cond_output_dict

    # @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None,
                      treg=False, dreamsampler=False,
                      apply_lm=False, lmgrad=None, lm_mask=None, x0=None, apply_stochastic_noise=False, get_eps=False):
        b, *_, device = *x.shape, x.device
        
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
        
        if dreamsampler:
            x_in = x
            t_in = t
            model_t, cond_output_dict = self.model.apply_model(x_in, t_in, c)
            model_uncond, _ = self.model.apply_model(x_in, t_in, unconditional_conditioning)
            pred_x0_uc = (x - sqrt_one_minus_at * model_uncond) / a_t.sqrt()
            pred_x0_c = (x - sqrt_one_minus_at * model_t) / a_t.sqrt()
            if ((index + 1) % 3 == 0 or index == 0):
                assert apply_lm == False, "TREG is not compatible with Latent Manipulation"
                with torch.enable_grad():
                    updated_pred_x0_uc = lmgrad.cal_grad(pred_x0=pred_x0_uc, model=self.model, index=index)
                if index==0:
                    return updated_pred_x0_uc, pred_x0_c, cond_output_dict
                pred_x0 = a_prev * updated_pred_x0_uc + (1. - a_prev) * pred_x0_uc
                pred_x0 = lm_mask * pred_x0 + (1. - lm_mask) * ((1 - a_t) * pred_x0) + ((1. - lm_mask) * (a_t * pred_x0_c)) 
                noise = sqrt_one_minus_at * model_uncond + a_t.sqrt() * torch.randn_like(x, device=device)
                x_prev = a_prev.sqrt() * pred_x0 + (1. - a_prev).sqrt() * noise
            else: 
                model_uncond, cond_output_dict_uncond = self.model.apply_model(x_in, t_in, unconditional_conditioning)
                pred_x0 = (x - sqrt_one_minus_at * model_uncond) / a_t.sqrt()
                x_prev = a_prev.sqrt() * pred_x0 + (1. - a_prev).sqrt() * model_uncond
            if index == 0:
                return pred_x0, pred_x0, cond_output_dict 
            return x_prev, pred_x0, cond_output_dict
        elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output, cond_output_dict = self.model.apply_model(x, t, c)
        else:
            
            x_in = x
            t_in = t
            model_t, cond_output_dict_cond = self.model.apply_model(x_in, t_in, c)
            model_uncond, cond_output_dict_uncond = self.model.apply_model(x_in, t_in, unconditional_conditioning)
            
            if isinstance(model_t, tuple):
                model_t, _ = model_t
            if isinstance(model_uncond, tuple):
                model_uncond, _ = model_uncond
            if cond_output_dict_cond is not None:
                cond_output_dict = dict()
                for k in cond_output_dict_cond.keys():
                    cond_output_dict[k] = torch.cat([cond_output_dict_uncond[k], cond_output_dict_cond[k]])
            else:
                cond_output_dict = None
            
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)
        
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        
        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # """For intro figure, save the image"""
            # """Plot Pred_x0"""
            # img = self.model.decode_first_stage(pred_x0)[0]
            # img = (img + 1.0) / 2.0
            # img = img.clamp(0, 1)
            # import os 
            # from torchvision.utils import save_image
            # # 저장 경로
            # save_dir = f"./intro_figure_final/stableviton/during_generation" #hard_consistency, interpolation, masked_restore, unmasked_masked_restore
            # os.makedirs(save_dir, exist_ok=True)
            # save_image(img, os.path.join(save_dir, f"output_{t}.jpg"))
            if get_eps:
                return pred_x0, e_t
            
            if treg and ((index + 1) % 3 == 0 or index == 0):
                assert apply_lm == False, "TREG is not compatible with Latent Manipulation"
                with torch.enable_grad():
                    updated_pred_x0 = lmgrad.hard_consistency(pred_x0=pred_x0, model=self.model, index=index)
                if index==0:
                    return updated_pred_x0, pred_x0, cond_output_dict
                pred_x0 = a_prev * updated_pred_x0 + (1. - a_prev) * pred_x0 
                """For intro figure, save the image"""
                """Plot Pred_x0"""
                img = self.model.decode_first_stage(updated_pred_x0)[0]
                img = (img + 1.0) / 2.0
                img = img.clamp(0, 1)
                import os 
                from torchvision.utils import save_image
                # 저장 경로
                save_dir = f"./intro_figure_final/treg/during_generation" #hard_consistency, interpolation, masked_restore, unmasked_masked_restore
                os.makedirs(save_dir, exist_ok=True)
                save_image(img, os.path.join(save_dir, f"output_{t}.jpg"))
                
            """Apply Latent Manipulation"""
            if apply_lm and ((index + 1) % 3 == 0 or index == 0):
                assert treg == False, "Latent Manipulation is not compatible with TREG"

                updated_pred_x0 = lmgrad.hard_consistency(pred_x0=pred_x0, model=self.model, index=index)

                """Plot"""
                # import os 
                # from torchvision.utils import save_image
                # measure = lmgrad.measurement
                # measure_mask = lmgrad.mask
                # measure = (measure + 1.0) / 2.0
                # measure = measure.clamp(0, 1)
                # save_dir = f"./figure"
                # os.makedirs(save_dir, exist_ok=True)
                # save_image(measure, os.path.join(save_dir, f"measurement.png"))
                # save_image(measure_mask, os.path.join(save_dir, f"measurement_mask.png"))
                if index==0:
                    return updated_pred_x0, pred_x0, cond_output_dict

                else: # 원본의 특성을 보존하면서, 업데이트가 필요한 부분만 최적화하는 방식으로 적합합니다. Encoding과정에서 masked region에서의 일부 정보가 손실될 수 있기 때문에, 기존 latent code 정보를 그대로 사용한다.
                    updated_pred_x0 = ffq.replace_high_frequency_fft(updated_pred_x0, pred_x0)
                    pred_x0 = lm_mask * ( a_prev * updated_pred_x0 + (1. - a_prev) * pred_x0 ) + (1. - lm_mask) * pred_x0
                    
                    #pred_x0 = a_prev * updated_pred_x0 + (1. - a_prev) * pred_x0
                    #pred_x0 = updated_pred_x0 
                    
                     
                    #high_freq = ffq.extract_high_frequency_fft(pred_x0)
                    #low_freq = ffq.remove_high_frequency_fft(pred_x0)
                    # new_updated_pred_x0 = ffq.replace_high_frequency_fft(updated_pred_x0, pred_x0)
                    # replaced_freq = lm_mask * ( a_prev * new_updated_pred_x0 + (1. - a_prev) * pred_x0 ) + (1. - lm_mask) * pred_x0
                
                """For intro figure, save the image"""
                """Plot Pred_x0"""
                # img = self.model.decode_first_stage(pred_x0)[0]
                # img = (img + 1.0) / 2.0
                # img = img.clamp(0, 1)
                # import os 
                # from torchvision.utils import save_image
                # # 저장 경로
                # save_dir = f"./intro_figure_final/ours/during_generation" #hard_consistency, interpolation, masked_restore, unmasked_masked_restore
                # os.makedirs(save_dir, exist_ok=True)
                # save_image(img, os.path.join(save_dir, f"output_{t}.jpg"))

                
                """Plot Pred_x0"""
                # img = self.model.decode_first_stage(pred_x0)[0]
                # img = (img + 1.0) / 2.0
                # img = img.clamp(0, 1)
                # import os 
                # from torchvision.utils import save_image
                # # 저장 경로
                # save_dir = f"./figure/unmasked_masked_restore" #hard_consistency, interpolation, masked_restore, unmasked_masked_restore
                # os.makedirs(save_dir, exist_ok=True)
                # save_image(img, os.path.join(save_dir, f"output_{t}.png"))
                
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        """Stochastic Noise""" # iter를 맞춰줘야 한다.
        if apply_stochastic_noise and ((index + 1) % 3 == 0 or index == 0):
            # Inter
            stochastic_e_t = (1. - a_prev).sqrt() * a_prev.sqrt() * noise_like(x.shape, device, False) 
            x_prev = a_prev.sqrt() * pred_x0 + (1. - a_prev) * e_t + stochastic_e_t
        else:
            dir_xt = (1. - a_prev).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt 

        """Naive reverse process"""
        # dir_xt = (1. - a_prev).sqrt() * e_t
        # x_prev = a_prev.sqrt() * pred_x0 + dir_xt 
        if use_original_steps:
            sigmas = self.ddim_sigmas_for_original_num_steps
            sigmas_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            dir_xt = (1. - a_prev - sigmas_t**2).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + sigmas_t * noise_like(x.shape, device, False)
        if index == 0:
            return pred_x0, pred_x0, cond_output_dict 
        return x_prev, pred_x0, cond_output_dict

    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):
        num_reference_steps = self.ddpm_num_timesteps if use_original_steps else self.ddim_timesteps.shape[0]

        assert t_enc <= num_reference_steps
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps]
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), i, device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)[0]
            else:
                assert unconditional_conditioning is not None
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)[0]

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec