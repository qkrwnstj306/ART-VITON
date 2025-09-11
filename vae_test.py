from diffusers import AutoencoderKL

vae_path = "path/to/compvis/vae"  # CompVis VAE checkpoint 경로

vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")  # subfolder는 저장 구조에 따라 조정
