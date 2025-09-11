from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import cv2

class CustomDataset(Dataset):
    def __init__(self, original_dir, generated_dir, generated_img_extension: str):
        self.original_dir = original_dir
        self.generated_dir = generated_dir
        
        self.original_images = os.listdir(self.original_dir)
        self.generated_images = os.listdir(self.generated_dir)

        self.generated_img_extension = generated_img_extension
        
        self.transform = A.Compose([
            A.Resize(height=512, width=384),
        ])

    def __len__(self):
        assert len(self.original_images) == len(self.generated_images), f"The number of paired datasets are not same...!(original/generated) ({len(self.original_images)/len(self.generated_images)})"
        return len(self.original_images)
    
    def __getitem__(self, idx):
        img_name = self.original_images[idx]
        img_path = os.path.join(self.original_dir, img_name)
        
        img_name_without_extension = img_name.replace(".jpg", "")
        generated_img_path = os.path.join(self.generated_dir, img_name_without_extension + "_" + img_name_without_extension + self.generated_img_extension)
        
        original_img = cv2.imread(img_path)
        generated_img = cv2.imread(generated_img_path)

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        generated_img = cv2.cvtColor(generated_img, cv2.COLOR_BGR2RGB)
        
        original_img = self.transform(image=original_img)['image']
        generated_img = self.transform(image=generated_img)['image']
        
        original_img = torch.from_numpy(original_img).permute(2, 0, 1).float()
        generated_img = torch.from_numpy(generated_img).permute(2, 0, 1).float()
        
        return {'original_img': original_img, 'generated_img': generated_img}


def collate_fn(samples):
    
    original_img = samples[0]['original_img'].unsqueeze(0)
    generated_img = samples[0]['generated_img'].unsqueeze(0)
    
    return {'original_img':original_img, 'generated_img':generated_img}

def main(original_dir, generated_dir, generated_img_extension):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_dir = original_dir
    generated_dir = generated_dir
    generated_img_extension = generated_img_extension

    dataset = CustomDataset(original_dir, generated_dir, generated_img_extension)

    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collate_fn, batch_size=1, shuffle=False)
    
    ssim_score = 0
    for idx, batch in enumerate(dataloader):
        original_img, generated_img = batch['original_img'].to(device), batch['generated_img'].to(device)
        # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
        # Y: (N,3,H,W)  

        # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
        ssim_score += ssim(original_img, generated_img, data_range=255, size_average=True) # return a scalar
        #ms_ssim_score = ms_ssim(original_img, generated_img, data_range=255, size_average=True )
        
    return ssim_score / len(dataloader)
    
if __name__ == "__main__":
    original_dir = '/home/qkrwnstj/StableVITON/dataset/zalando-hd-resized/test/image'
    
    generated_dir_lst = ['/home/qkrwnstj/StableVITON/stableVITON/pair', 
                         '/home/qkrwnstj/StableVITON/ours/pair',
                         '/home/qkrwnstj/StableVITON/original/pair',
                         '/home/qkrwnstj/StableVITON/ours_latents/pair',
                         '/home/qkrwnstj/StableVITON/s_l_VITON/pair',
                         '/home/qkrwnstj/StableVITON/s_l_VITON_repaint/pair',
                         '/home/qkrwnstj/StableVITON/ablation_highFreq/pair',
                         ]
    
    generated_dir = generated_dir_lst[6]
    
    generated_img_extension = '.jpg'
    
    SSIM_score = main(original_dir, generated_dir, generated_img_extension)
    print(f"SSIM_score: {SSIM_score:.4f}")