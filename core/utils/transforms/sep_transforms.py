import numpy as np
import torch
import cv2

from sklearn.preprocessing import OneHotEncoder
# from ..semantics_utils import num_class

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, imgs_sems_tuple):
        imgs, full_segs, sems = imgs_sems_tuple
        if sems is not None:
            imgs = [torch.from_numpy(img.transpose((2, 0, 1)).copy()).float() for img in imgs]
            sems = [torch.from_numpy(sem.transpose((2, 0, 1)).copy()).float() for sem in sems]
            if full_segs is not None:
                full_segs = [torch.from_numpy(fs.transpose((2, 0, 1)).copy()).float() for fs in full_segs]            
            return imgs, full_segs, sems
        imgs = [torch.from_numpy(img.transpose((2, 0, 1)).copy()).float() for img in imgs]

        return imgs, full_segs, sems

    
class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, imgs_sems_tuple):
        imgs, full_segs, sems = imgs_sems_tuple
        if sems is not None:
            imgs = [cv2.resize(img, (self.new_w, self.new_h)) for img in imgs]
            new_key_objs = []

            for key_obj in sems:
                if key_obj.shape[-1] == 0:  ## no key obj found
                    new_key_obj = np.zeros((self.new_h, self.new_w, 0), dtype=np.uint8)
                elif key_obj.shape[-1] == 1:
                    new_key_obj = cv2.resize(
                        key_obj,
                        (self.new_w, self.new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )[:, :, None]
                else:
                    new_key_obj = cv2.resize(
                        key_obj,
                        (self.new_w, self.new_h),
                        interpolation=cv2.INTER_NEAREST,
                    )

                new_key_objs.append(new_key_obj)
            if full_segs is not None:
                full_segs = [cv2.resize(fs, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST)[:, :, None] for fs in full_segs]
            return imgs, full_segs, new_key_objs
        
        imgs = [cv2.resize(img, (self.new_w, self.new_h)) for img in imgs]
        
        return imgs, full_segs, sems # , sems

class OneHotSemantics(object):
    def __init__(self):
        self.enc = OneHotEncoder(sparse=False)
        self.enc.fit(np.array(range(num_class))[:, None])

    def __call__(self, imgs_sems_tuple):
        imgs, sems = imgs_sems_tuple
        if sems is not None:
            sems = [self.enc.transform(sem.reshape((-1, 1))).reshape((*sem.shape, -1)) for sem in sems]
            return imgs, sems
    
        # sems = [self.enc.transform(sem.reshape((-1, 1))).reshape((*sem.shape, -1)) for sem in sems]
        
        return imgs, sems