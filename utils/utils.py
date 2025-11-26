import cv2
import numpy as np
from PIL import Image
import torch
import random

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def resize_image_new(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image, nw, nh, dx, dy


def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def restore_to_original_size(restored, 
                             original_size, 
                             input_size, 
                             letterbox_params):
    """
    restored: 模型输出张量 [C,H,W] 或 [1,C,H,W]
    original_size: (iw, ih)
    input_size: (w, h)
    letterbox_params: (nw, nh, dx, dy)
    """
    iw, ih = original_size
    w, h = input_size
    nw, nh, dx, dy = letterbox_params

    # -----------------------------
    # 1. 取出 Tensor，反归一化
    # -----------------------------
    if restored.ndim == 4:
        img = restored[0]
    else:
        img = restored
    img = img.detach().cpu().float()

    img = img.permute(1, 2, 0).numpy()

    # 反向 preprocess_input
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img * std + mean
    img = img * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # -----------------------------
    # 2. 去掉 padding, 恢复到 (nw, nh)
    # -----------------------------
    img = img[dy:dy+nh, dx:dx+nw, :]

    # -----------------------------
    # 3. resize 回到原始尺寸
    # -----------------------------
    img = cv2.resize(img, (ih, iw), interpolation=cv2.INTER_LINEAR)

    return img