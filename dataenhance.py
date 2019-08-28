import torch as t
from torchvision import transforms
import random
from hyper_parameters import *


def image_transform(size=img_size):
    fr_transform = transforms.Compose([
            transforms.Resize(size), #大小
            transforms.RandomCrop(cro_size), #随机裁剪
            transforms.RandomHorizontalFlip(), #水平翻转
            transforms.RandomVerticalFlip(), #垂直翻转
            transforms.RandomRotation(angle), #角度调整
            transforms.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.3, hue=0.2), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return fr_transform

def enhance_transform():
    possi = random.randint(0, 19)
    if possi % 20 == 0:#0.5%概率不发生变化,相当于数据集扩容20倍
        output_trans = transforms.Compose([
                transforms.Resize(img_size), 
                transforms.CenterCrop(cro_size), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        return output_trans
    else:
        possi2 = random.randint(0, 3)
        if possi2%5==0:#20%概率发生不同比例放缩
            return image_transform(224)
        elif possi2%5==1:
            return image_transform(245)#面积放大1.2倍
        elif possi2%5==2:
            return image_transform(274)#面积放大1.5倍
        elif possi2%5==3:
            return image_transform(316)#面积放大2倍


def transform_standard():
    output_trans = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.CenterCrop(cro_size), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return output_trans
