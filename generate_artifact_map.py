import cv2
import numpy as np
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--unpair", action="store_true")
args = parser.parse_args()
is_paired = not args.unpair
if is_paired:
    pair = "pair"
else:
    pair = "unpair"

input_dir = f"./results/{pair}"
output_dir = os.path.join(input_dir, "artifact_maps")
os.makedirs(output_dir, exist_ok=True)


gamma = 0.5  

image_paths = glob(os.path.join(input_dir, "*.jpg"))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"[경고] 이미지를 불러올 수 없음: {img_path}")
        continue

    img = cv2.resize(img, (384, 512))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_channel, a_channel, b_channel = cv2.split(lab)

    sobel_l = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1, ksize=3)
    sobel_a = cv2.Sobel(a_channel, cv2.CV_64F, 1, 1, ksize=3)
    sobel_b = cv2.Sobel(b_channel, cv2.CV_64F, 1, 1, ksize=3)

    grad_mag = np.sqrt(sobel_l**2 + sobel_a**2 + sobel_b**2)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    grad_mag_gamma = np.power(grad_mag / 255.0, gamma) * 255
    grad_mag_gamma = grad_mag_gamma.astype(np.uint8)

    heatmap = cv2.applyColorMap(grad_mag_gamma, cv2.COLORMAP_TURBO)

    base_name = os.path.basename(img_path)
    name_no_ext = os.path.splitext(base_name)[0]
    save_path = os.path.join(output_dir, f"{name_no_ext}.jpg")

    cv2.imwrite(save_path, heatmap)

    print(f"[완료] 저장됨: {save_path}")
