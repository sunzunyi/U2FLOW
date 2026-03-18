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
# input_transform_eval = transforms.Compose([
#     # sep_transforms.OneHotSemantics(),
#     sep_transforms.ArrayToTensor()
# ])
# input_transform_eval = copy.deepcopy(input_transform_eval)
# # input_transform_eval.transforms.insert(0, sep_transforms.Zoom(256, 832))  #   kitt 
# input_transform_eval.transforms.insert(0, sep_transforms.Zoom(256, 704))   #  cityscapes

class ImgSeqDataset(Dataset):
    def __init__(self, input_transform=None, co_transform=None, ap_transform=None, test_shape = [256, 832] ):   #  test_shape = [256, 832]   smurf[488, 1144]
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.image_list = []
        self.sem_list = []
        self.full_seg_list = []
        self.has_sem = False
        self.start_sem_loss = False
        self.flow_list = []
        self.flow_noc = []
        self.extra_info = []
        self.split = ''
        self.is_test = False
        self.init_seed = False
        input_transform_eval = transforms.Compose([
            sep_transforms.ArrayToTensor()
        ])
        self.input_transform_eval = copy.deepcopy(input_transform_eval)
        self.input_transform_eval.transforms.insert(0, sep_transforms.Zoom(*test_shape))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.is_test: # benchmark
            images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
            h_info, w_info, c_info = images[0].shape     
            images, _, _ = self.input_transform_eval((images, None, None)) 
            if self.extra_info == []:
                extra_info = self.image_list[idx]
            else:
                extra_info = self.extra_info[idx]
            return torch.stack(images), extra_info, h_info, w_info
            
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        images = [imageio.imread(path).astype(np.float32) / 255. for path in self.image_list[idx]]
        
        if self.start_sem_loss and self.has_sem:
            # semantics = None
            if random.random() < 0.5: # 0.5 swap
                images = images[::-1]
                sem = np.load(self.sem_list[idx][-1]) / 255.0
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

        if self.co_transform is not None:  # training
            # In unsupervised learning, there is no need to change target with image
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

        if  self.co_transform is None:   # Test
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[idx])
            flow = np.array(flow).astype(np.float32)
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()
            valid = torch.from_numpy(valid).float()
            flownoc, nocmask = frame_utils.readFlowKITTI(self.flow_noc[idx])
            flownoc = np.array(flownoc).astype(np.float32) 
            flownoc = torch.from_numpy(flownoc).permute(2, 0, 1).float()
            nocmask = torch.from_numpy(nocmask).float()

            return torch.stack(images), flow, valid, flownoc, nocmask
        if self.co_transform.fw_crop:
            return torch.stack(images), torch.stack(images_ph), torch.stack(images_origin), start, 0, 0
        else:
            return torch.stack(images), torch.stack(images_ph), 0, 0, 0, 0


class KITTIRawFile(ImgSeqDataset):
    def __init__(self, ap_transform=None,
                 input_transform=None, co_transform=None, test_shape = [256, 832]):
        super(KITTIRawFile, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        root = "/your_folder/kitti_raw/"
        self.image_list = []
        try:
            with open('core/kitti_train_2f_sv.txt', 'r') as file:
                for line in file:
                    img_paths = line.strip().split()
                    img0, img1, img2, img3 = img_paths
                    self.image_list.extend([
                        [osp.join(root, img0.strip()).replace('.png', '.jpg'),
                            osp.join(root, img1.strip()).replace('.png', '.jpg')],
                        [osp.join(root, img2.strip()).replace('.png', '.jpg'),
                            osp.join(root, img3.strip()).replace('.png', '.jpg')]
                    ])
        except FileNotFoundError:
            print("Error: The file 'core/kitti_train_2f_sv.txt' was not found.")

class KITTI2012(ImgSeqDataset):
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None,  split='training', test_shape = [256, 832], three_frame=False):
        super(KITTI2012, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        root='/your_folder/data_stereo_flow_multiview/'

        self.image_list = []
        self.flow_list = []
        self.flow_noc = []
        self.extra_info = []
        if split == 'training':
            for idx_list in range(194):        #                                                    10 11
                if three_frame:
                    self.image_list.append([(root+"training/image_2/000{:03}_{:02}.png".format(idx_list, i + 9)) for i in range(3)])
                else:
                    self.image_list.append([(root+"training/image_2/000{:03}_{:02}.png".format(idx_list, i + 10)) for i in range(2)])
                # #                                                                                      9 10 11 12
                self.flow_list.append("/your_folder/kitti2012_stereo_flow/data_stereo_flow/training/flow_occ/000{:03}_10.png".format(idx_list))
                self.flow_noc.append("/your_folder/kitti2012_stereo_flow/data_stereo_flow/training/flow_noc/000{:03}_10.png".format(idx_list))
    
        elif split == 'testing':  #  benchmark
            self.is_test = True
            for idx_list in range(195):        #                                                    10 11
                if three_frame:
                    self.image_list.append([(root+"testing/image_2/000{:03}_{:02}.png".format(idx_list, i + 9)) for i in range(3)])
                    self.extra_info.append("000{:03}_10.png".format(idx_list))
                else:
                    self.image_list.append([(root+"testing/image_2/000{:03}_{:02}.png".format(idx_list, i + 10)) for i in range(2)])
                    self.extra_info.append("000{:03}_10.png".format(idx_list))


# train
class KITTI2012_MV_Sem(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training'):
        super(KITTI2012_MV_Sem, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        root='your_folder/data_stereo_flow_multiview/'
        sam_dir = "your_folder/kitti2012_mv_sam1_key_objects/"
        full_seg_dir = "your_folder/kitti2012_mv_sam1_fullseg/"

        self.split = split
        self.sem_list = []
        self.has_sem = True
        self.full_seg_list = []
        for sub_dir in ['testing', 'training']:
            img_l_dir, img_r_dir = 'image_2', 'image_3'
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 1):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])
                    if id_ != id_nex1 - 1 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        self.image_list.append([file_path, file_path_nex1])

                        file_path_nex0_sem = file_path.replace(root, sam_dir).replace('.png', '.npy')
                        file_path_nex1_sem = file_path_nex1.replace(root, sam_dir).replace('.png', '.npy')
                        self.sem_list.append([file_path_nex0_sem, file_path_nex1_sem])

                        file_path_nex0_full_seg = file_path.replace(root, full_seg_dir)
                        file_path_nex1_full_seg = file_path_nex1.replace(root, full_seg_dir)
                        self.full_seg_list.append([file_path_nex0_full_seg, file_path_nex1_full_seg])



class KITTI2015(ImgSeqDataset):
    def __init__(self, ap_transform=None, input_transform=None, co_transform=None, test_shape = [256, 832], split='training', three_frame=False):
        super(KITTI2015, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        root = 'your_folder/kitti2015/'
        root_gt = 'your_folder/kitti2015/'

        self.image_list = []
        self.flow_list = []
        self.flow_noc = []
        self.extra_info = []
        if split == 'training':
            for idx_list in range(200):        #                                                                   10 11
                if three_frame:
                    self.image_list.append([(root+"multi_view/training/image_2/000{:03}_{:02}.png".format(idx_list, i + 9)) for i in range(3)])
                else:
                    self.image_list.append([(root+"multi_view/training/image_2/000{:03}_{:02}.png".format(idx_list, i + 10)) for i in range(2)])
                self.flow_list.append(root_gt+"training/flow_occ/000{:03}_10.png".format(idx_list))
                self.flow_noc.append(root_gt+"training/flow_noc/000{:03}_10.png".format(idx_list))

        elif split == 'testing':  #   benchmark
            self.is_test = True
            for idx_list in range(200):        #                                                            10 11
                if three_frame:
                    self.image_list.append([(root+"multi_view/testing/image_2/000{:03}_{:02}.png".format(idx_list, i + 9)) for i in range(3)])
                else:
                    self.image_list.append([(root+"multi_view/testing/image_2/000{:03}_{:02}.png".format(idx_list, i + 10)) for i in range(2)])
                self.extra_info.append("000{:03}_10.png".format(idx_list))

class KITTI2015_MV_Sem(ImgSeqDataset):
    def __init__(self, ap_transform, input_transform, co_transform, test_shape = [256, 832], split='training'):
        super(KITTI2015_MV_Sem, self).__init__(input_transform=input_transform,
                                           co_transform=co_transform,
                                           ap_transform=ap_transform, test_shape = test_shape)
        self.split = split
        root = 'your_folder/multi_view'
        sam_dir = "your_folder/kitti2015_mv_sam1_key_objects/"
        full_seg_dir = "your_folder/kitti2015_mv_sam1_fullseg/"

        self.sem_list = []
        self.has_sem = True
        self.full_seg_list = []
        for sub_dir in ['testing', 'training/Clean']:
            img_l_dir, img_r_dir = 'image_2', 'image_3'
            for img_list in [img_l_dir, img_r_dir]:
                img_dir = os.path.join(root, sub_dir, img_list)
                file_ls = os.listdir(img_dir)
                file_ls.sort()
                for ind in range(len(file_ls) - 1):
                    name = file_ls[ind]
                    nex_name1 = file_ls[ind + 1]
                    id_ = int(name[-6:-4])  # 000{:03}_{:02}.png
                    id_nex1 = int(nex_name1[-6:-4])
                    if id_ != id_nex1 - 1 or  \
                        12 >= id_ >= 9 or 12 >= id_nex1 >= 9:
                        pass
                    else:
                        file_path = os.path.join(img_dir, name)
                        file_path_nex1 = os.path.join(img_dir, nex_name1)
                        self.image_list.append([file_path, file_path_nex1])

                        file_path_nex0_sem = file_path.replace(root, sam_dir).replace('.png', '.npy')
                        file_path_nex1_sem = file_path_nex1.replace(root, sam_dir).replace('.png', '.npy')
                        self.sem_list.append([file_path_nex0_sem, file_path_nex1_sem, ])

                        file_path_nex0_full_seg = file_path.replace(root, full_seg_dir)
                        file_path_nex1_full_seg = file_path_nex1.replace(root, full_seg_dir)
                        self.full_seg_list.append([file_path_nex0_full_seg, file_path_nex1_full_seg])  

class Aug_cfg:
    def __init__(self, crop, vflip, hflip, swap, crop_bottom=-1, fw_crop=False):
        self.crop = crop
        self.vflip = vflip
        self.hflip = hflip
        self.swap = swap
        # "crop_bottom": 0.25
        self.crop_bottom = crop_bottom
        self.fw_crop = fw_crop
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
        # sep_transforms.OneHotSemantics(),
        sep_transforms.ArrayToTensor()
    ])
    aug_cfg = Aug_cfg(crop=False, vflip = True, hflip=True, swap=True)
    if hasattr(args, 'crop_bottom') and args.crop_bottom > 0:
        aug_cfg.crop_bottom = args.crop_bottom
    if hasattr(args, 'fw_crop') and args.fw_crop:
        aug_cfg.fw_crop = args.fw_crop
    co_transform = get_co_transforms(aug_cfg)
    train_input_transform = copy.deepcopy(input_transform)
    if not hasattr(aug_cfg, 'fw_crop') or not aug_cfg.fw_crop :
        train_input_transform.transforms.insert(0, sep_transforms.Zoom(*args.image_size))
    ap_cfg = AP_cfg(cj=True, cj_bri=0.5, cj_con=0.0, cj_hue=0.0, cj_sat=0.0, gamma=False, gblur=True)
    ap_transform = get_ap_transforms(ap_cfg)

    if args.stage == 'KITTI_Raw':
        kitti_raw = KITTIRawFile(
            input_transform=train_input_transform,    
            ap_transform=ap_transform,  
            co_transform=co_transform 
            )
        train_dataset = kitti_raw
    elif args.stage == 'MV_2stage_Sem':
        kitti2015_MV = KITTI2015_MV_Sem(
            input_transform=train_input_transform,    
            ap_transform=ap_transform,  
            co_transform=co_transform 
            )
        kitti2012_MV = KITTI2012_MV_Sem(
            input_transform=train_input_transform,    
            ap_transform=ap_transform,  
            co_transform=co_transform 
            )    
        train_set_2 = ConcatDataset([kitti2015_MV, kitti2012_MV])
        train_set_2.name = 'kitti-mv'
        train_dataset = train_set_2
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
