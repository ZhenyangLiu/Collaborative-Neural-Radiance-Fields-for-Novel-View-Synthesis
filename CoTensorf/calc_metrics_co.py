import argparse
import os
import cv2
import skimage
import skimage
import numpy as np
import torch
import lpips
import colorama
from tqdm import tqdm

loss_fn = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

class MetricLogger:

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0
    def log(self, newvalue):
        self.sum += newvalue
        self.count += 1

    def avg(self):
        return self.sum/self.count

def parse_args():
    parser = argparse.ArgumentParser(description="计算指标")
    parser.add_argument("--gt", default="/home/guest/LZY/conerf/nerf-pytorch-master/logs/chair/testset_gt_500000", help="Ground Truth 所在的文件夹")
    parser.add_argument("--r_1", default="/home/guest/LZY/conerf/TensoRF-main_tensor/log/tensorf_chair_VM_1/imgs_test_all", help="Rendered Views 所在的文件夹")
    parser.add_argument("--r_2", default="/home/guest/LZY/conerf/TensoRF-main_tensor/log/tensorf_chair_VM_2/imgs_test_all", help="Rendered Views 所在的文件夹")
    parser.add_argument("--r_3", default="/home/guest/LZY/conerf/TensoRF-main_tensor/log/tensorf_chair_VM_3/imgs_test_all", help="Rendered Views 所在的文件夹")  
    parser.add_argument("--r_4", default="/home/guest/LZY/conerf/TensoRF-main_tensor/log/tensorf_chair_VM_4/imgs_test_all", help="Rendered Views 所在的文件夹")      
    parser.add_argument("--save_dir", default="/home/guest/LZY/conerf/TensoRF-main_tensor/log/chair_final", help="final Views 所在的文件夹")          
    return parser.parse_args()

def calc_psnr(img1, img2):
    if (img1 == img2).all():
        print(colorama.Fore.RED+ f"PSNR Warning: The input images are exactly the same. Returning 100." + colorama.Style.RESET_ALL)
        return 100.0
    else:
        return skimage.metrics.peak_signal_noise_ratio(img1, img2)

def calc_ssim(img1, img2):
    return skimage.metrics.structural_similarity(img1, img2, channel_axis=2)

def calc_lpips(img1, img2):
    def normalize_negative_one(img):
        normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
        return 2*normalized_input - 1
    
    img1 = normalize_negative_one(img1)
    img2 = normalize_negative_one(img2)
    img1 = torch.tensor(img1).permute([2,0,1]).unsqueeze(0)
    img2 = torch.tensor(img2).permute([2,0,1]).unsqueeze(0)
    lpips_score = loss_fn(img1, img2)
    return lpips_score.item()

def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

args = parse_args()

gt_dir = [f for f in os.listdir(args.gt) if f.endswith('.png')]

gt_names = sorted(gt_dir)

r1 = [f for f in os.listdir(args.r_1) if f.endswith('.png')]
r2 = [f for f in os.listdir(args.r_2) if f.endswith('.png')]
r3 = [f for f in os.listdir(args.r_3) if f.endswith('.png')]
r4 = [f for f in os.listdir(args.r_4) if f.endswith('.png')]

render_names_1 = sorted(r1)
render_names_2 = sorted(r2)
render_names_3 = sorted(r3)
render_names_4 = sorted(r4)

assert len(gt_names) == len(render_names_1)

image_count = len(gt_names)
gt_images = []
for gt_name in gt_names:
    img =read_image(os.path.join(args.gt, gt_name))
    gt_images.append(img)

render_images_1 = []
render_images_2 = []
render_images_3 = []
render_images_4 = []


for render_name_1 in render_names_1:
    img_1 = read_image(os.path.join(args.r_1, render_name_1))
    render_images_1.append(img_1)

for render_name_2 in render_names_2:
    img_2 = read_image(os.path.join(args.r_2, render_name_2))
    render_images_2.append(img_2)

for render_name_3 in render_names_3:
    img_3 = read_image(os.path.join(args.r_3, render_name_3))
    render_images_3.append(img_3)

for render_name_4 in render_names_4:
    img_4 = read_image(os.path.join(args.r_4, render_name_4))
    render_images_4.append(img_4)

psnr_logger = MetricLogger()
ssim_logger = MetricLogger()
lpips_logger = MetricLogger()

os.makedirs(args.save_dir, exist_ok=True)

for idx, (gt, img1, img2, img3, img4) in tqdm(enumerate(zip(gt_images, render_images_1, render_images_2, render_images_3, render_images_4))):
    # print(f"Processing {gt_names[idx]} and {render_names[idx]}")
    psnr_logger.log(calc_psnr(gt, (img1+img2+img3+img4)/4.))
    ssim_logger.log(calc_ssim(gt, (img1+img2+img3+img4)/4.))
    lpips_logger.log(calc_lpips(gt, (img1+img2+img3+img4)/4.))
    save_path = args.save_dir + '/' + render_names_1[idx]
    save_img = ((img1+img2+img3+img4)/4.)*255.0
    save_img = save_img[:, :, [2,1,0]]
    cv2.imwrite(save_path, save_img.astype(np.uint8))

print(f"Avg PSNR: {psnr_logger.avg()}")
print(f"Avg SSIM: {ssim_logger.avg()}")
print(f"Avg LPIPS: {lpips_logger.avg()}")

