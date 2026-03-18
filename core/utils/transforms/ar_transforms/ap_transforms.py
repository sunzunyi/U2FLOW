import numpy as np
import torch
from torchvision import transforms as tf
from PIL import ImageFilter


def get_ap_transforms(cfg): # cj=True, cj_bri=0.5, cj_con=0.0, cj_hue=0.0, cj_sat=0.0, gamma=False, gblur=True
    transforms = [ToPILImage()]
    if cfg.cj:
        transforms.append(ColorJitter(brightness=cfg.cj_bri,
                                      contrast=cfg.cj_con,
                                      saturation=cfg.cj_sat,
                                      hue=cfg.cj_hue))
    if cfg.gblur:
        transforms.append(RandomGaussianBlur(0.5, 3))

    # transforms.append(ToTensor())
    # if cfg.gamma:
    #     transforms.append(RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True))

    transforms.append(RandomEraser(p=0.5, min_size=50, max_size=100, max_operations=2, fill_type='mean'))  # Add random erase to the last frame
    transforms.append(ToTensor())

    return tf.Compose(transforms)


# from https://github.com/visinf/irr/blob/master/datasets/transforms.py
class ToPILImage(tf.ToPILImage):
    def __call__(self, imgs):
        return [super(ToPILImage, self).__call__(im) for im in imgs]


class ColorJitter(tf.ColorJitter):
    def __call__(self, imgs):
        # transform = self.get_params(self.brightness, self.contrast,
        #                             self.saturation, self.hue)
        return [super(ColorJitter, self).__call__(im)  for im in imgs]


class ToTensor(tf.ToTensor):
    def __call__(self, imgs):
        return [super(ToTensor, self).__call__(im) for im in imgs]


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, imgs):
        gamma = self.get_params(self._min_gamma, self._max_gamma)
        return [self.adjust_gamma(im, gamma, self._clip_image) for im in imgs]


class RandomGaussianBlur():
    def __init__(self, p, max_k_sz):
        self.p = p
        self.max_k_sz = max_k_sz

    def __call__(self, imgs):
        if np.random.random() < self.p:
            radius = np.random.uniform(0, self.max_k_sz)
            imgs = [im.filter(ImageFilter.GaussianBlur(radius)) for im in imgs]
        return imgs


import numpy as np
from PIL import Image
import random

class RandomEraser:
    def __init__(self, p=0.5, min_size=50, max_size=100, max_operations=3, fill_type='mean'):
        self.p = p
        self.min_size = min_size
        self.max_size = max_size
        self.max_operations = max_operations
        self.fill_type = fill_type

    def __call__(self, imgs):
        if np.random.random() >= self.p:
            return imgs
            
        num_operations = random.randint(1, self.max_operations)
        img = imgs[-1]  # We only erase the last frame
        
        for _ in range(num_operations):
            width, height = img.size
            
            erase_width = random.randint(self.min_size, min(self.max_size, width))
            erase_height = random.randint(self.min_size, min(self.max_size, height))
            x = random.randint(0, width - erase_width)
            y = random.randint(0, height - erase_height)
            
            mask = Image.new('RGB', (erase_width, erase_height))
            
            if self.fill_type == 'mean':
                region = img.crop((x, y, x + erase_width, y + erase_height))
                region_np = np.array(region)
                mean_color = tuple(np.mean(region_np.reshape(-1, 3), axis=0).astype(int))
                mask = Image.new('RGB', (erase_width, erase_height), mean_color)

            else:  # 'constant'
                mask = Image.new('RGB', (erase_width, erase_height), (128, 128, 128))
            
            img.paste(mask, (x, y))

        imgs[-1] = img
        return imgs