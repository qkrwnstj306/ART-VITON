import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.fft
import numpy as np

def extract_high_frequency_fft(x, keep_ratio=0.1):
    B, C, H, W = x.shape
    x_fft = torch.fft.fft2(x)
    x_fftshift = torch.fft.fftshift(x_fft)

    # 중앙 고주파 영역 제외
    center_h, center_w = H // 2, W // 2
    margin_h, margin_w = int(H * keep_ratio // 2), int(W * keep_ratio // 2)
    mask = torch.ones_like(x_fftshift)
    mask[:, :, center_h-margin_h:center_h+margin_h, center_w-margin_w:center_w+margin_w] = 0

    high_fft = x_fftshift * mask
    high = torch.fft.ifft2(torch.fft.ifftshift(high_fft)).real
    return high

def remove_high_frequency_fft(x, keep_ratio=0.1):
    B, C, H, W = x.shape
    x_fft = torch.fft.fft2(x)
    x_fftshift = torch.fft.fftshift(x_fft)

    center_h, center_w = H // 2, W // 2
    margin_h, margin_w = int(H * keep_ratio // 2), int(W * keep_ratio // 2)
    mask = torch.zeros_like(x_fftshift)
    mask[:, :, center_h-margin_h:center_h+margin_h, center_w-margin_w:center_w+margin_w] = 1

    low_fft = x_fftshift * mask
    low = torch.fft.ifft2(torch.fft.ifftshift(low_fft)).real
    return low

def replace_high_frequency_fft(target_latent, source_latent, keep_ratio=0.1):
    low = remove_high_frequency_fft(target_latent, keep_ratio)
    high = extract_high_frequency_fft(source_latent, keep_ratio)
    return low + high
