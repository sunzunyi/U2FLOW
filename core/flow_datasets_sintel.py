import time
import imageio.v2 as imageio
import numpy as np
import random
# from path import Path
from abc import abstractmethod, ABCMeta
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from .utils.transforms import sep_transforms
import os
from glob import glob
import os.path as osp
import copy
from torchvision import transforms
from torch.utils.data import ConcatDataset
from .utils.transforms.co_transforms import get_co_transforms
from .utils.transforms.ar_transforms.ap_transforms import get_ap_transforms
from .utils import frame_utils
import matplotlib.pyplot as plt

class ImgSeqDataset(Dataset):
    def __init__(self, input_transform=None, co_transform=None, ap_transform=None):
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.image_list = []
        self.sem_list = []
        self.full_seg_list = []
        self.has_sem = False
        self.start_sem_loss = False
        self.flow_list = []
        self.flow_occ = []
        self.extra_info = []
        self.split = ''
        self.is_test = False
        self.init_seed = False
        input_transform_eval = transforms.Compose([
            sep_transforms.ArrayToTensor()
        ])
        self.input_transform_eval = copy.deepcopy(input_transform_eval)
        self.input_transform_eval.transforms.insert(0, sep_transforms.Zoom(448, 1024))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.is_test: # benchmark
            images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
            h_info, w_info, c_info = images[0].shape     
            images, _, _ = self.input_transform_eval((images, None, None)) 
            data_info = {}
            data_info['h_info'] = h_info
            data_info['w_info'] = w_info
            data_info['extra_info'] = self.extra_info[idx]

            return torch.stack(images), data_info
            
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
        
        if self.start_sem_loss:
            if random.random() < 0.5: # 0.5 swap
                images = images[::-1]
                sem = np.load(self.sem_list[idx][-1]) / 255.0 # h w n 
                semantics = [sem] # h w 1
                full_segs = [imageio.imread(path)[:, :, None] for path in self.full_seg_list[idx][::-1]] # h w 1
            else:
                sem = np.load(self.sem_list[idx][0]) / 255.0
                semantics = [sem] # h w 1
                full_segs = [imageio.imread(path)[:, :, None] for path in self.full_seg_list[idx]] # h w 1

            if self.co_transform.fw_crop:
                images_set, full_segs, semantics, start = self.co_transform(images, full_segs, semantics)
                images = images_set[0]
                images_origin = images_set[1]
                if len(images)>2:
                    images_origin = [images_origin[1], images_origin[2]]
                else:
                    images_origin = [images_origin[0], images_origin[1]]
            else:
                images, full_segs, semantics, _= self.co_transform(images, full_segs, semantics)   # [inputs_crop, inputs_origin]
                images = images[0]
            if self.co_transform.fw_crop:
                images.extend(images_origin)
                images_set = images
                images_set, full_segs, semantics = self.input_transform((images_set, full_segs, semantics))
                images = images_set[:-2]
                images_origin = images_set[-2:]
            else:
                images, full_segs, semantics = self.input_transform((images, full_segs, semantics))
            semantics_roi = torch.full((1, *images[0].shape[-2:]), np.nan, dtype=torch.float32)
            if semantics[0].shape[0] != 0:
                valid_key_obj = semantics[0].mean(dim=(1, 2)) >= 0.005
                if valid_key_obj.sum() != 0:
                    idx = np.random.choice(np.where(valid_key_obj)[0])
                    semantics_roi = semantics[0][[idx], :, :]  # 1 H W

            images_ph = self.ap_transform([img.clone() for img in images])
            if self.co_transform.fw_crop:
                return torch.stack(images), torch.stack(images_ph), torch.stack(images_origin), start, semantics_roi, torch.stack(full_segs)
            else:
                return torch.stack(images), torch.stack(images_ph), 0, 0, semantics_roi, torch.stack(full_segs)

        if self.co_transform is not None:
            if self.co_transform.fw_crop:
                images_set, _, _, start= self.co_transform(images)
                images = images_set[0]
                images_origin = images_set[1]
                if len(images)>2:
                    images_origin = [images_origin[1], images_origin[2]]
                else:
                    images_origin = [images_origin[0], images_origin[1]]
            else:
                images, _, _, _= self.co_transform(images)   # [inputs_crop, inputs_origin]
                images = images[0]

        if self.input_transform is not None:
            if self.co_transform.fw_crop:
                images.extend(images_origin)
                images_set = images
                images_set, _, _ = self.input_transform((images_set, None, None))
                images = images_set[:-2]
                images_origin = images_set[-2:]
            else:
                images, _, _ = self.input_transform((images, None, None))
        else:
            images, _, _ = self.input_transform_eval((images, None, None))      
        if self.ap_transform is not None:
            images_ph = self.ap_transform([img.clone() for img in images])

        if  self.co_transform is None:
            flow = frame_utils.read_gen(self.flow_list[idx])
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            flowocc = (imageio.imread( self.flow_occ[idx]).astype(np.float32)[:, :, None]/ 255.0)
            flowocc = torch.from_numpy(flowocc).permute(2, 0, 1).float()
            return torch.stack(images), flow, flowocc, self.extra_info[idx]

        if self.co_transform.fw_crop:
            return torch.stack(images), torch.stack(images_ph), torch.stack(images_origin), start, 0, 0
        else:
            return torch.stack(images), torch.stack(images_ph), 0, 0, 0, 0


class SintelRawFile(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None):
        # self.sp_file = sp_file
        super(SintelRawFile, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        self.image_list = []
        root='your_folder/Sintel_scene/scene'
        for scene in os.listdir(root): # 0 1 2 3 4 5 6 7 ~ 162
            image_list = sorted(glob(osp.join(root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                # self.extra_info += [ (scene, i) ] # scene and frame_id

class MpiSintel_Submission(ImgSeqDataset):
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None,   split='test', dstype='clean', three_frame=False):
        super(MpiSintel_Submission, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        root = 'your_folder/Sintel'
        image_root = osp.join(root, split, dstype)
        self.is_test = True

        for scene in os.listdir(image_root):# benchmark
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            if three_frame:
                for i in range(len(image_list)-1):
                    if i == 0:
                        self.image_list.append([image_list[0], image_list[0], image_list[i+1]])  
                        self.extra_info += [[image_list[0], False]] # scene and frame_id
                    else: #  2 ~~
                        self.image_list.append([image_list[i-1], image_list[i], image_list[i+1]])
                        self.extra_info += [[image_list[i], True]] # scene and frame_id
            else:
                for i in range(len(image_list)-1):
                    self.image_list.append([image_list[i], image_list[i+1]])
                    self.extra_info += [image_list[i]] # scene and frame_id


class MpiSintel(ImgSeqDataset): 
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None,         
                dataset_type="clean",
                split="training",
                subsplit="trainval",
                three_frame=False):
        super(MpiSintel, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform)
        self.training_scene = ['alley_1', 'ambush_4', 'ambush_6', 'ambush_7', 'bamboo_2',
                               'bandage_2', 'cave_2', 'market_2', 'market_5', 'shaman_2',
                               'sleeping_2', 'temple_3']  # Unofficial train-val split
        
        self.root = 'your_folder/' + split
        sam_dir = 'your_folder' + split
        full_seg_dir = 'your_folder' + split

        self.dataset_type = dataset_type

        # self.split = split
        self.subsplit = subsplit
        
        self.image_list = []
        self.flow_list = []
        self.flow_occ = []
        self.extra_info = []
        # flow_root = osp.join(self.root, 'flow')
        image_root = osp.join(self.root, dataset_type)

        for scene in sorted(os.listdir(image_root)):
            if self.subsplit == "trainval":
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list)-1):
                    path_split = image_list[i].split("/")
                    flow_dir = os.path.join(
                        "/".join(path_split[:-3]),
                        "flow",
                        scene,
                        path_split[-1][:-4] + ".flo",
                    )
                    occ_mask_dir = os.path.join(
                        "/".join(path_split[:-3]),
                        "occlusions",
                        scene,
                        path_split[-1],
                    )
                    if three_frame:
                        if i == 0:
                            self.image_list.append([image_list[0], image_list[0], image_list[i+1]])  
                        else: #  2 ~~
                            self.image_list.append([image_list[i-1], image_list[i], image_list[i+1]])
                    
                    else:
                        self.image_list.append([image_list[i], image_list[i+1]])
                    self.flow_list.append(flow_dir)
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.flow_occ.append(occ_mask_dir)

                    file_path_nex0_sem = image_list[i].replace(self.root, sam_dir).replace('.png', '.npy')
                    file_path_nex1_sem = image_list[i+1].replace(self.root, sam_dir).replace('.png', '.npy')
                    self.sem_list.append([file_path_nex0_sem, file_path_nex1_sem])

                    file_path_nex0_full_seg = image_list[i].replace(self.root, full_seg_dir)
                    file_path_nex1_full_seg = image_list[i+1].replace(self.root, full_seg_dir)
                    self.full_seg_list.append([file_path_nex0_full_seg, file_path_nex1_full_seg])  

            elif self.subsplit == "train" and scene in self.training_scene:
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list)-1):
                    path_split = image_list[i].split("/")
                    flow_dir = os.path.join(
                        "/".join(path_split[:-3]),
                        "flow",
                        scene,
                        path_split[-1][:-4] + ".flo",
                    )
                    occ_mask_dir = os.path.join(
                        "/".join(path_split[:-3]),
                        "occlusions",
                        scene,
                        path_split[-1],
                    )
                    if three_frame:
                        if i == 0:
                            self.image_list.append([image_list[0], image_list[0], image_list[i+1]])  
                        else: #  2 ~~
                            self.image_list.append([image_list[i-1], image_list[i], image_list[i+1]])
                    
                    else:
                        self.image_list.append([ image_list[i], image_list[i+1]])
                    self.flow_list.append(flow_dir)
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.flow_occ.append(occ_mask_dir)

            elif self.subsplit == "val" and scene not in self.training_scene:   
                image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
                for i in range(len(image_list)-1):
                    path_split = image_list[i].split("/")
                    flow_dir = os.path.join(
                        "/".join(path_split[:-3]),
                        "flow",
                        scene,
                        path_split[-1][:-4] + ".flo",
                    )
                    occ_mask_dir = os.path.join(
                        "/".join(path_split[:-3]),
                        "occlusions",
                        scene,
                        path_split[-1],
                    )
                    if three_frame:
                        if i == 0:
                            self.image_list.append([image_list[0], image_list[0], image_list[i+1]])  
                        else: #  2 ~~
                            self.image_list.append([image_list[i-1], image_list[i], image_list[i+1]])
                    
                    else:
                        self.image_list.append([ image_list[i], image_list[i+1]])
                    self.flow_list.append(flow_dir)
                    self.extra_info += [ (scene, i) ] # scene and frame_id
                    self.flow_occ.append(occ_mask_dir)




class Aug_cfg:
    def __init__(self, crop, vflip, hflip, swap, crop_bottom=-1):
        self.vflip = vflip
        self.hflip = hflip
        self.swap = swap
        # "crop_bottom": 0.25
        self.crop_bottom = crop_bottom
        self.crop = crop
        self.para_crop = [384, 832]
class AP_cfg:
    def __init__(self, cj, cj_bri, cj_con, cj_hue, cj_sat, gamma, gblur):
        self.cj = cj
        self.cj_bri = cj_bri
        self.cj_con = cj_con
        self.cj_hue = cj_hue
        self.cj_sat = cj_sat
        self.gamma = gamma
        self.gblur = gblur

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    input_transform = transforms.Compose([
        sep_transforms.ArrayToTensor()
    ])
    aug_cfg = Aug_cfg(crop=True, vflip = True, hflip=True, swap=True)
    if hasattr(args, 'crop_bottom') and args.crop_bottom > 0:
        aug_cfg.crop_bottom = args.crop_bottom
    co_transform = get_co_transforms(aug_cfg)
    train_input_transform = copy.deepcopy(input_transform)
    ap_cfg = AP_cfg(cj=True, cj_bri=0.5, cj_con=0.5, cj_hue=0.0, cj_sat=0.5, gamma=True, gblur=True)
    ap_transform = get_ap_transforms(ap_cfg)

    if args.stage == 'Sintel_Raw':
        sintel_raw = SintelRawFile(
            input_transform=train_input_transform,    
            ap_transform=ap_transform,  
            co_transform=co_transform 
            )
        train_dataset = sintel_raw
    elif args.stage == 'Sintel_2stage':
        sintel_complete_1 = MpiSintel(
            input_transform=train_input_transform,    
            ap_transform=ap_transform,  
            co_transform=co_transform,    
            dataset_type="clean",
            split="training",
            subsplit=args.subsplit,  # "trainval"
            ) 
        sintel_complete_2 = MpiSintel(
            input_transform=train_input_transform,    
            ap_transform=ap_transform,  
            co_transform=co_transform,    
            dataset_type="final",
            split="training",
            subsplit=args.subsplit,  # "trainval"
            ) 
        sintel_complete = ConcatDataset([sintel_complete_1, sintel_complete_2])
        sintel_complete.name = 'sintel_complete'
        train_dataset = sintel_complete
    else:
        raise ValueError(f"args.stage = {args.stage} is not a valid stage!")


    if args.distributed:
        sampler_train = data.DistributedSampler(train_dataset, shuffle=True)
    else:
        sampler_train = None
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler_train,
        pin_memory=True, shuffle=(sampler_train is None), num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    if args.distributed:
        return train_loader, sampler_train
    
    return train_loader
