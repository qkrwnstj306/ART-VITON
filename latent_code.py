CODE_NAME = [
    "pure",
    "whole",
    "unmasked",
    "unmasked_wrp",
    "unmasked_statistics",
    "unmasked_offset",
    "unmasked_clothingAlignment",
    "smooth_latent"
]

import torch
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange

def get_code(model, z, ts, c, batch, code_name="pure"):
    bs, shape = z.shape[0], z.shape[1:]  # (B, C, H, W)
    C, H, W = shape

    mask = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1)
    mask = mask.round()

    assert code_name in CODE_NAME
    if code_name == "pure":
        size = (bs, C, H, W)
        start_code = torch.randn(size, device=z.device)

    elif code_name == "whole":
        start_code = model.q_sample(z, ts)

    elif code_name == "unmasked":
        size = (bs, C, H, W)
        noise = torch.randn(size, device=z.device)
        alleviated = model.q_sample(z, ts)
        start_code = noise * (1 - mask) + alleviated * mask

    elif code_name == "unmasked_wrp":
        img = rearrange(batch['image'], 'b h w c -> b c h w')
        agn_mask = rearrange(batch['agn_mask'], 'b h w c -> b c h w')
        wrp_cloth = rearrange(batch['wrp_cloth'], 'b h w c -> b c h w')
        wrp_cloth_mask = rearrange(batch['wrp_cloth_mask'], 'b h w c -> b c h w')
        
        start_code = wrp_cloth * wrp_cloth_mask * (1 - agn_mask) + img * agn_mask
        start_code = model.get_first_stage_encoding(model.encode_first_stage(start_code))
        start_code = model.q_sample(start_code, ts)

    elif code_name == "unmasked_offset":
        size = (bs, C, H, W)
        noise = torch.randn(size, device=z.device) + 0.1 * torch.randn((bs, C, 1, 1), device=z.device)
        alleviated = model.q_sample(z, ts)
        start_code = noise * (1 - mask) + alleviated * mask
    
    elif code_name == "unmasked_statistics":
        def initialize_masked_region(latent, mask, stats_path="./masked_feature/clothing_statistics.pth"):
            stats = torch.load(stats_path, map_location="cpu")
            mean, std = stats["mean"].to(latent.device), stats["std"].to(latent.device)
            noise = torch.randn_like(latent) * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
            latent = noise * (1. - mask) + latent * mask
            return latent

        start_code = initialize_masked_region(z, mask)
        start_code = model.q_sample(start_code, ts)
    
    elif code_name == "smooth_latent":
        def smooth_latent(latent, kernel_size=7, sigma=3):
            channels = latent.shape[1]
            x = torch.arange(kernel_size).float() - kernel_size // 2
            kernel_1d = torch.exp(-(x**2) / (2 * sigma**2))
            kernel_1d /= kernel_1d.sum()
            kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
            kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size).to(latent.device)
            return F.conv2d(latent, kernel_2d, padding=kernel_size // 2, groups=channels)

        z_T = model.q_sample(z, ts)
        z_meas_only = z_T * mask

        z_smooth = smooth_latent(z_meas_only)

        start_code = z_T * mask + z_smooth * (1 - mask)

    elif code_name == "unmasked_clothingAlignment":
        def get_center_of_mask(mask):
            coords = torch.nonzero(mask[0], as_tuple=False)  
            center = coords.float().mean(dim=0) if coords.numel() > 0 else torch.tensor([0, 0])
            
            return center.int()  


        def pad_or_crop_to_match(image, target_shape, is_mask=True):
            h, w = image.shape[1:3]
            th, tw = target_shape
            
            pad_h = max(0, th - h)
            pad_w = max(0, tw - w)

            padding_value = 0 if is_mask else 0.9294
            
            padded_image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=padding_value)
                
            return padded_image[..., :th, :tw]

        def shift_to_center(image, source_mask, target_mask, is_mask=True):
            source_center = get_center_of_mask(source_mask)
            target_center = get_center_of_mask(target_mask)

            shift_y = target_center[0] - source_center[0]
            shift_x = target_center[1] - source_center[1]
            
            height, width = image.shape[1], image.shape[2]

            shifted_image = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
            
            padding_value = 0 if is_mask else 0.9294

            if shift_y > 0:
                shifted_image[:, :shift_y, :] = padding_value  
            elif shift_y < 0:
                shifted_image[:, shift_y:, :] = padding_value  

            if shift_x > 0:
                shifted_image[:, :, :shift_x] = padding_value  
            elif shift_x < 0:
                shifted_image[:, :, shift_x:] = padding_value 

            return shifted_image


        def align_and_overlay(inpainting_mask, clothing_mask, clothing_image):
            inpainting_size = torch.count_nonzero(inpainting_mask, dim=(1, 2, 3))  
            clothing_size = torch.count_nonzero(clothing_mask, dim=(1, 2, 3))
            
            batch_size = inpainting_mask.shape[0]
            resized_clothing_masks = []
            resized_clothing_images = []
            
            for i in range(batch_size):
                inpainting_mask_single = inpainting_mask[i]
                clothing_mask_single = clothing_mask[i]
                clothing_image_single = clothing_image[i]
                
                if clothing_size[i] > 0:
                    scale_factor = (inpainting_size[i] / clothing_size[i]).sqrt()
                    resize_transform = transforms.Resize((int(clothing_mask_single.shape[-2] * scale_factor),
                                                        int(clothing_mask_single.shape[-1] * scale_factor)))
                    clothing_mask_resized = resize_transform(clothing_mask_single)
                    clothing_image_resized = resize_transform(clothing_image_single)

                    clothing_mask_resized = pad_or_crop_to_match(clothing_mask_resized, inpainting_mask_single.shape[-2:])
                    clothing_image_resized = pad_or_crop_to_match(clothing_image_resized, inpainting_mask_single.shape[-2:], is_mask=False)
                    
                    # 중심 맞추기
                    clothing_mask_resized = shift_to_center(clothing_mask_resized, clothing_mask_resized, inpainting_mask_single)
                    clothing_image_resized = shift_to_center(clothing_image_resized, clothing_image_resized, inpainting_mask_single, is_mask=False)
                else:
                    clothing_mask_resized = torch.zeros_like(inpainting_mask_single)  
                    clothing_image_resized = torch.zeros_like(inpainting_mask_single) 
                
                resized_clothing_masks.append(clothing_mask_resized)
                resized_clothing_images.append(clothing_image_resized)
            
            return torch.stack(resized_clothing_masks), torch.stack(resized_clothing_images)
        
        def initialize_masked_region(latent, mask, stats_path="./masked_feature/clothing_statistics.pth"):
            """ 
            저장된 통계를 이용하여 masked region(옷) 초기화
            - latent: (B, C, H, W)
            - mask: (B, 1, H, W)
            """
            stats = torch.load(stats_path, map_location="cpu")  # CPU에서 로드
            mean, std = stats["mean"].to(latent.device), stats["std"].to(latent.device)

            noise = torch.randn_like(latent) * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

            latent = noise * (1. - mask) + latent * mask

            print("Masked region initialized.")
            return latent
        
        agn_mask_inversion = rearrange(batch['agn_mask'], 'b h w c -> b c h w') # raw image space
        inpainting_mask = 1. - agn_mask_inversion
        clothing_mask = rearrange(batch['cloth_mask'], 'b h w c -> b c h w')
        clothing_mask, clothing_image = align_and_overlay(inpainting_mask=inpainting_mask, clothing_mask=clothing_mask, clothing_image=c["c_concat"][0])
    
        resize_transform = transforms.Resize((64, 48))
        resized_clothing_mask = torch.stack([resize_transform(mask) for mask in clothing_mask])
        latent_clothing = model.get_first_stage_encoding(model.encode_first_stage(clothing_image))

        start_code = initialize_masked_region(latent_clothing, resized_clothing_mask)
        start_code[mask.repeat(1, 4, 1, 1).to(torch.bool)] = z[mask.repeat(1, 4, 1, 1).to(torch.bool)]
        start_code = model.q_sample(start_code, ts) 

    else:
        raise ValueError(f"Unknown code_name: {code_name}")
    
    return start_code, mask