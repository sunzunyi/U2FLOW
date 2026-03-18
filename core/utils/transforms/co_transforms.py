import numbers
import random
import numpy as np
# from scipy.misc import imresize
# from skimage.transform import resize as imresize
import scipy.ndimage as ndimage


def get_co_transforms(aug_args):
    transforms = []
    # for cityscapes, where we crop to remove the car logo
    # if 'crop_bottom' in aug_args and aug_args.crop_bottom > 0:   #  "crop_bottom": 0.25,
    if hasattr(aug_args, 'crop_bottom') and aug_args.crop_bottom > 0:
        transforms.append(CropBottom(aug_args.crop_bottom))  

    if hasattr(aug_args, 'crop') and aug_args.crop:
        transforms.append(RandomCrop(aug_args.para_crop)) 
    if aug_args.hflip:
        transforms.append(RandomHorizontalFlip())
    if aug_args.vflip:
        transforms.append(RandomVerticalFlip())
    if aug_args.swap:
        transforms.append(RandomSwap())
    if hasattr(aug_args, 'fw_crop') and aug_args.fw_crop :
        transforms.append(Zoom(370, 1224))  # upflow
        transforms.append(RandomCrop_fw((256, 832)))    #   256, 832   296, 696

        # transforms.append(RandomCrop(aug_args.para_crop))
        # transforms.append(RandomCrop_fw(aug_args.fw_crop.size))  #  kitti 296×696   sintel 368×496


    return Compose(transforms, aug_args)


class Compose(object):
    def __init__(self, co_transforms, aug_args):
        self.co_transforms = co_transforms
        if hasattr(aug_args, 'fw_crop') and aug_args.fw_crop:
            self.fw_crop = True
        else:
            self.fw_crop = False

    def __call__(self, input, full_segs = None, semantics = None, target = None): # , semantics, target
        for t in self.co_transforms:
            input, full_segs, semantics, target = t(input, full_segs, semantics, target)
        return input, full_segs, semantics, target

    
class CropBottom(object):
    def __init__(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def __call__(self, inputs, full_segs = None, semantics = None, target = None):
        h, w, _ = inputs[0].shape
        new_h = int(h * (1 - self.crop_ratio))

        inputs = [img[:new_h] for img in inputs]
        if full_segs is not None:
            full_segs = [fs[:new_h] for fs in full_segs]
        if semantics is not None:
            semantics = [sem[:new_h] for sem in semantics]
        # if 'mask' in target:
        #     target['mask'] = target['mask'][:new_h]
        # if 'flow' in target:
        #     target['flow'] = target['flow'][:new_h]
        return inputs, full_segs, semantics, target
    
import cv2
class Zoom(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, inputs, full_segs = None, semantics = None, target = None):
        imgs = inputs[0]
        sems = semantics
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
            return imgs, full_segs, new_key_objs, target
        
        imgs = [cv2.resize(img, (self.new_w, self.new_h)) for img in imgs]
        # sems = [cv2.resize(sem, (self.new_w, self.new_h), interpolation=cv2.INTER_NEAREST) for sem in sems]
        
        return imgs, full_segs, sems, target
    
class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, full_segs = None, semantics = None, target = None):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs, full_segs, semantics, target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = [img[y1: y1 + th, x1: x1 + tw] for img in inputs]
        if semantics is not None:
            semantics = [sem[y1: y1 + th, x1: x1 + tw] for sem in semantics]
        if full_segs is not None:
            full_segs = [fs[y1: y1 + th, x1: x1 + tw] for fs in full_segs]
        
        return inputs, full_segs, semantics, target

class RandomCrop_fw(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.crop_size = (int(size), int(size))
        else:
            self.crop_size = size
        self.rho = 8
    def __call__(self, inputs, full_segs, semantics, target):
        height, width = inputs[0].shape[:2]
        patch_size_h, patch_size_w = self.crop_size
        x = np.random.randint(self.rho, width - self.rho - patch_size_w)
        # print(self.rho, height - self.rho - patch_size_h)
        y = np.random.randint(self.rho, height - self.rho - patch_size_h)
        start = np.array([x, y])
        target = np.expand_dims(np.expand_dims(start, 1), 2)  # (a = 2, 1, 1)
        inputs_crop = [im[y: y + patch_size_h, x: x + patch_size_w, :] for im in inputs]
        if semantics is not None:
            semantics = [sem[y: y + patch_size_h, x: x + patch_size_w, :] for sem in semantics]
        if full_segs is not None:
            full_segs = [fs[y: y + patch_size_h, x: x + patch_size_w, :] for fs in full_segs]
        
        return [inputs_crop, inputs], full_segs, semantics, target

class RandomSwap(object):
    def __call__(self, inputs, full_segs = None, semantics = None, target = None):
        n = len(inputs)

        if semantics is None:
            if random.random() < 0.5: # 0.5
                inputs = inputs[::-1]

            # if 'mask' in target:
            #     target['mask'] = target['mask'][::-1]
            # if 'flow' in target:
            #     raise NotImplementedError("swap cannot apply to flow")
        return [inputs, 0] , full_segs, semantics, target


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, full_segs = None, semantics = None, target = None): 
        if random.random() < 0.5:
            inputs = [np.fliplr(im) for im in inputs]
            if semantics is not None:
                semantics = [np.fliplr(sem) for sem in semantics]
            if full_segs is not None:
                full_segs = [np.fliplr(fs) for fs in full_segs]
            # if 'mask' in target:
            #     target['mask'] = [np.copy(np.fliplr(mask)) for mask in target['mask']]
            # if 'flow' in target:
            #     for i, flo in enumerate(target['flow']):
            #         flo = np.copy(np.fliplr(flo))
            #         flo[:, :, 0] *= -1
            #         target['flow'][i] = flo
        return inputs, full_segs, semantics, target
    

class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, inputs, full_segs = None, semantics=None, target=None):
        if random.random() < 0.1:
            inputs = [np.flipud(im) for im in inputs]
            if semantics is not None:
                semantics = [np.flipud(sem) for sem in semantics]
            if full_segs is not None:
                full_segs = [np.flipud(fs) for fs in full_segs]
            # if 'mask' in target:
            #     target['mask'] = [np.copy(np.flipud(mask)) for mask in target['mask']]
            # if 'flow' in target:
            #     for i, flo in enumerate(target['flow']):
            #         flo = np.copy(np.flipud(flo))
            #         flo[:, :, 1] *= -1
            #         target['flow'][i] = flo
        return inputs, full_segs, semantics, target