import torch
import numpy as np
import torch.nn.functional as F

# from utils.semantics_utils import num_class
import random

class ObjectCache_sem:
    def __init__(self, cache_size=500):
        self.cache_size = cache_size
        self._obj_mask_cache = None
        self._img_cache = None
        self._motion_cache = None
        self.count = 0

    # We initialize the cache when we receive the first push sample so that it adapts to the current img size automatically
    def init_cache(self, img_size):
        self._obj_mask_cache = torch.zeros(
            (self.cache_size, 1, *img_size), dtype=torch.float32
        )
        self._img_cache = torch.zeros(
            (self.cache_size, 3, *img_size), dtype=torch.float32
        )
        self._motion_cache = torch.zeros((self.cache_size, 2), dtype=torch.float32)
        return

    def pop(self, B=8, with_aug=True):  # we do not remove objects after popping
        if self.count < self.cache_size:  # do not use it before it is full
            return None

        idx = np.random.choice(self.cache_size, B, replace=False)
        obj_mask = self._obj_mask_cache[idx]
        img = self._img_cache[idx]
        motion = self._motion_cache[idx]

        if with_aug:
            rand_scale = (
                torch.rand(B) * 0.7 + 0.8
            )  # randomly rescale motion by 0.8-1.5 times
            rand_scale *= (-1) ** (
                torch.rand(B) > 0.5
            ).float()  # randomly reverse motion
            motion = motion * rand_scale[:, None]

            flip_flag = torch.rand(B) > 0.5  # randomly horitontal-flip obj mask
            img[flip_flag] = img[flip_flag].flip(dims=[3])
            obj_mask[flip_flag] = obj_mask[flip_flag].flip(dims=[3])
            motion[flip_flag, 0] *= -1

        return obj_mask, img, motion

    def push(self, obj_mask, img, motion):
        """
        obj_mask: [B, 1, H, W]
        img: [B, 3, H, W]
        motion: [B, 2]
        """

        if self._obj_mask_cache is None:
            self.init_cache(img_size=img.shape[-2:])

        B = obj_mask.shape[0]

        if self.count <= self.cache_size - B:  # many spaces
            self._obj_mask_cache[self.count : (self.count + B)] = obj_mask
            self._img_cache[self.count : (self.count + B)] = img
            self._motion_cache[self.count : (self.count + B)] = motion
            self.count += B
            return

        elif self.count < self.cache_size:  # partial space
            space = self.cache_size - self.count
            self._obj_mask_cache[self.count :] = obj_mask[:space]
            self._img_cache[self.count :] = img[:space]
            self._motion_cache[self.count :] = motion[:space]

            overwrite_idx = np.random.choice(self.count, B - space, replace=False)
            self._obj_mask_cache[overwrite_idx] = obj_mask[space:]
            self._img_cache[overwrite_idx] = img[space:]
            self._motion_cache[overwrite_idx] = motion[space:]
            self.count += space
            return

        else:  # no spaces; random overwrite
            overwrite_idx = np.random.choice(self.cache_size, B, replace=False)
            self._obj_mask_cache[overwrite_idx] = obj_mask
            self._img_cache[overwrite_idx] = img
            self._motion_cache[overwrite_idx] = motion
            return

from utils.warp_utils import flow_warp
def add_fake_object_sem(input_dict):
    # prepare input
    img1_ot = input_dict["img1_tgt"]
    img2_ot = input_dict["img2_tgt"]
    flow_ot = input_dict["flow_tgt"]
    noc_ot = input_dict["noc_tgt"]

    img = input_dict["img_src"]
    obj_mask = input_dict["obj_mask"]
    motion = input_dict["motion"][:, :, None, None]

    obj_mask_total = input_dict['obj_mask_total']

    b, _, h, w = img1_ot.shape

    # add object to frame 1
    img1_ot = obj_mask * img + (1 - obj_mask) * img1_ot

    # add object to frame 2
    new_obj_mask = flow_warp(obj_mask, -motion.repeat(1, 1, h, w), pad="zeros")
    new_img = flow_warp(img, -motion.repeat(1, 1, h, w), pad="border")
    img2_ot = new_obj_mask * new_img + (1 - new_obj_mask) * img2_ot

    # change flow
    flow_ot = obj_mask * motion + (1 - obj_mask) * flow_ot
    noc_ot = torch.max(noc_ot, obj_mask)  # where we are confident about flow_ot
    obj_mask_total = torch.max(obj_mask_total, obj_mask)

    return img1_ot, img2_ot, flow_ot, noc_ot, obj_mask_total