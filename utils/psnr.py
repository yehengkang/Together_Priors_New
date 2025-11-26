import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import math

def compute_psnr(img1, img2):
    """
    img1, img2 传入 numpy array 格式，范围 0-255
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr


def compute_ssim(img1, img2):
    
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    # 转成 torch
    img1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0).float() / 255.0
    
    # 高斯核（11×11）
    def gaussian_kernel(window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    window_size = 11
    sigma = 1.5
    gauss = gaussian_kernel(window_size, sigma).unsqueeze(1)
    window = gauss.mm(gauss.t()).unsqueeze(0).unsqueeze(0)

    mu1 = F.conv2d(img1, window, padding=window_size//2)
    mu2 = F.conv2d(img2, window, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu12

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim.mean().item()

