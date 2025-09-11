import os
from os.path import join as opj
import argparse

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./dataset/zalando-hd-resized/test")
    parser.add_argument("--generated_dir", type=str, default="./results")  # generated image directory
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()

# Optional: Feathering for mask smoothing
# def feather_mask(mask, kernel_size=21, sigma=10):
#     return cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)

class Generated_Dataset(Dataset):
    def __init__(self, data_root_dir, generated_dir, unpair, generated_img_extension=".jpg"):
        self.data_root_dir = data_root_dir
        self.generated_dir = generated_dir
        self.is_pair = "pair" if not unpair else "unpair"

        self.img_pth = opj(self.data_root_dir, "image")
        self.ang_mask_pth = opj(self.data_root_dir, "agnostic-mask")
        self.gen_pth = opj(self.generated_dir, self.is_pair)
        self.gen_list = os.listdir(self.gen_pth)

        self.generated_img_extension = generated_img_extension

        self.transform = A.Compose([
            A.Resize(height=512, width=384),
        ])

    def __len__(self):
        return len(self.gen_list)

    def __getitem__(self, idx):
        img_name = self.gen_list[idx]
        img_name_no_ext = img_name.replace(".jpg", "")
        parts = img_name_no_ext.split("_")
        first_id = parts[0] + "_" + parts[1]
        second_id = parts[2] + "_" + parts[3]

        gen_path = opj(self.gen_pth, img_name)
        img_path = opj(self.img_pth, first_id + ".jpg")
        mask_path = opj(self.ang_mask_pth, first_id + "_mask.png")

        gen_img = cv2.imread(gen_path)
        img = cv2.imread(img_path)
        agn_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        gen_img = cv2.cvtColor(gen_img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(image=img)['image']
        gen_img = self.transform(image=gen_img)['image']
        agn_mask = self.transform(image=agn_mask)['image']

        agn_mask = (agn_mask >= 128).astype(np.float32)
        agn_mask = np.expand_dims(agn_mask, axis=-1)  # [H, W, 1]

        return {
            "img_name": img_name_no_ext,
            "gen_img": torch.from_numpy(gen_img.astype(np.float32)),     # [H, W, 3]
            "img": torch.from_numpy(img.astype(np.float32)),
            "agn_mask": torch.from_numpy(agn_mask.astype(np.float32)),  # [H, W, 1]
        }

@torch.no_grad()
def main(args):
    seed_everything(args.seed)

    dataset = Generated_Dataset(args.data_root_dir, args.generated_dir, args.unpair)
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=args.batch_size, pin_memory=True)

    is_paired = "pair" if not args.unpair else "unpair"
    save_pth = args.generated_dir + '_blending'
    save_dir = f"{save_pth}/{is_paired}"
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")

        gen_img = batch['gen_img']      # [B, H, W, 3]
        img = batch['img']
        agn_mask = batch['agn_mask']   # [B, H, W, 1]

        blended_output = gen_img * agn_mask + img * (1. - agn_mask)  # [B, H, W, 3]

        for i in range(blended_output.shape[0]):
            x_sample_img = blended_output[i].clamp(0, 255).byte().numpy()  # [H, W, 3]
            x_sample_img = x_sample_img[:, :, ::-1]  # RGB -> BGR

            to_path = opj(save_dir, f"{batch['img_name'][i]}.jpg")
            cv2.imwrite(to_path, x_sample_img)

if __name__ == "__main__":
    args = build_args()
    main(args)
