import torch
from torchmetrics.image.kid import KernelInceptionDistance
import cv2
import albumentations as A
import os

def cv2_to_tensor(img):
    """
    OpenCV 이미지를 RGB로 변환하고 torch.Tensor로 변환합니다.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # [H,W,C] -> [C,H,W]
    return img_tensor

def cal_kid(original_dir, generated_dir):
    kid = KernelInceptionDistance(subset_size=50, normalize=True)  # 이미지 범위 [0, 1]일 경우 normalize=True

    resize_transform = A.Resize(height=512, width=384)

    original_batch = []
    generated_batch = []

    original_files = sorted([f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.png'))])
    generated_files = sorted([f for f in os.listdir(generated_dir) if f.lower().endswith(('.jpg', '.png'))])

    for file in original_files:
        img_path = os.path.join(original_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 불러오기 실패: {file}")
            continue
        img = resize_transform(image=img)["image"]
        img_tensor = cv2_to_tensor(img)
        original_batch.append(img_tensor)
        if len(original_batch) == 50:
            kid.update(torch.stack(original_batch), real=True)
            original_batch = []

    for file in generated_files:
        img_path = os.path.join(generated_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 불러오기 실패: {file}")
            continue
        img = resize_transform(image=img)["image"]
        img_tensor = cv2_to_tensor(img)
        generated_batch.append(img_tensor)
        if len(generated_batch) == 50:
            kid.update(torch.stack(generated_batch), real=False)
            generated_batch = []

    kid_mean, kid_std = kid.compute()
    print(kid_mean, kid_std)

if __name__ == '__main__':
    generated_dir_lst = [
        '/home/qkrwnstj/StableVITON/results_999/stableVITON_blending_blending/unpair',
        '/home/qkrwnstj/StableVITON/results_999/999_ddpmPrior_woNoise_hardFFT_woStochastic_cfg1/unpair',
        '/home/qkrwnstj/StableVITON/ablation_highFreq/unpair',
        '/home/qkrwnstj/StableVITON/results_981/original/unpair',
        '/home/qkrwnstj/StableVITON/shhq_results_ours_whole/unpair',
    ]

    #original_dir = '/home/qkrwnstj/StableVITON/dataset/zalando-hd-resized/test/image'
    original_dir = '/home/qkrwnstj/StableVITON/dataset/SHHQ-1.0/test/image'
    generated_dir = generated_dir_lst[4]

    cal_kid(original_dir, generated_dir)
