import sys
sys.path.append('core')

import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from core.utils.misc import process_cfg
from core.utils import flow_viz
from core import flow_datasets, flow_datasets_sintel
from core.Networks import build_network
from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
import itertools
from tqdm import tqdm
import cv2
from core.utils.flow_utils import flow_to_image, resize_flow, writeFlowKITTI

from glob import glob

# sample
# scene index : Used to save uncertainty maps for specified frames
extra_info_sample = (('0003', '0017'), 
                     ('0019', '0055'),
                     ('0028', '0018'),
                     ('0029', '0068'),
                     ('0031', '0032'),
                     ('0034', '0024'),
                     ('0035', '0089'),
                     ('0040', '0055'),
                     ('0042', '0060'),
                     ('0046', '0047'))


class Spring_submission():
    def __init__(self, root='datasets/spring'):
        super(Spring_submission, self).__init__()

        root = '/ric_nas/flow/Spring/spring/test'
        if not os.path.exists(root):
            raise ValueError(f"Spring train directory does not exist: {root}")

        self.image_list = []
        self.extra_info_list = []

        for scene in sorted(os.listdir(root)):
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(root, scene, f"frame_{cam}", '*.png')))
                len_image = len(images)
                # forward
                for i in range(len_image - 1):
                    # for scene_index in extra_info_sample:
                    #     if cam == 'right':
                    #         continue
                    #     if scene != scene_index[0]:
                    #         continue
                    #     if i+1 != int(scene_index[1]):
                    #         continue
                        self.image_list.append((images[i], images[i + 1]))
                        self.extra_info_list.append((i+1, scene, cam, 'FW'))
                # backward
                for i in range(len_image - 1, 0, -1):
                    self.image_list.append((images[i], images[i - 1]))
                    self.extra_info_list.append((i+1, scene, cam, 'BW'))
                
        print(self.image_list[:2])
        print(self.extra_info_list[:2])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import h5py
def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]

def writeFlo5File(flow, filename):
    """write optical flow to file
        flow: optical flow with shape height x width x 2. Invalid values should be represented as np.nan
        filepath: file path where to write the flow
        """
    if not filename:
        raise ValueError("writeFlowFile: empty filepath")

    if len(flow.shape) != 3 or flow.shape[2] != 2:
        raise IOError(f"writeFlowFile {filename}: expected shape height x width x 2 but received {flow.shape}")

    if flow.shape[0] > flow.shape[1]:
        print(
            f"write flo file {filename}: Warning: Are you writing an upright image? Expected shape height x width x 2, got {flow.shape}")

    with h5py.File(filename, "w") as f:
        f.create_dataset("flow", data=flow, compression="gzip", compression_opts=5)


autocast = torch.cuda.amp.autocast
print(os.getcwd())
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    
    args = parser.parse_args()

    if args.config:
        import importlib
        cfg_module = importlib.import_module(f"configs.{args.config}")
        get_cfg = cfg_module.get_cfg

    cfg = get_cfg()
    cfg[cfg.network].iters = 10 # 10 iterations were inadvertently used during submission; using the default value of 12 should yield better results
    model = torch.nn.DataParallel(build_network(cfg))
    model.module.occ_from_back = False
    path_ck = 'checkpoints/sintel.pth'
    model.load_state_dict(torch.load(path_ck),strict=True)

    model.cuda()
    model.eval()

    print(cfg.network)
    print("Parameter Count: %d" % count_parameters(model))
    
    val_dataset = Spring_submission()

    for i_batch , _ in tqdm(enumerate(val_dataset.image_list), total=len(val_dataset.image_list)):
        frame0 = cv2.imread(val_dataset.image_list[i_batch][0])
        frame1 = cv2.imread(val_dataset.image_list[i_batch][1])
        frmae0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        frmae1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img0 = torch.from_numpy(frmae0).permute(2,0,1).float().cuda() / 255.0
        img1 = torch.from_numpy(frmae1).permute(2,0,1).float().cuda() / 255.0
        img0 = img0.unsqueeze(0)
        img1 = img1.unsqueeze(0)
        img0 = F.interpolate(img0, size=(448, 1024), mode='bilinear', align_corners=False)
        img1 = F.interpolate(img1, size=(448, 1024), mode='bilinear', align_corners=False)
        images = torch.stack([img0, img1], dim=1)
        images = images.cuda()  # [None]
        
        with torch.no_grad():
            flows = model(images)
        flow_pred = flows['flows_f12'][-1] 
        h, w = frame0.shape[:2]
        flow_pred_up = resize_flow(flow_pred, (h, w)).squeeze(0).detach().cpu()
        flow_pred_up = flow_pred_up.permute(1, 2, 0).numpy()  # HWC
 
        frame, sequence, cam, dir = val_dataset.extra_info_list[i_batch]
        
        output_dir = os.path.join('spring', 'spring_submissio_448x1024', sequence, f'flow_{dir}_{cam}')
        output_file = os.path.join(output_dir, f'flow_{dir}_{cam}_%04d.flo5' % frame)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        writeFlo5File(flow_pred_up, output_file)


        # uncertainty_map_batch = flows['uncertainty_f12'][-1]
        # uncertainty_map_batch = torch.exp(uncertainty_map_batch).clamp(min=1e-3, max=1e4)
        # unc_map = uncertainty_map_batch
        # unc_up = torch.nn.functional.interpolate(unc_map, size=(h, w), mode='area')
        # unc_up = unc_up.squeeze().cpu().numpy()
        # variance_log = np.log1p(unc_up)  # log(1+x)，Avoid log (0) issues
        # variance_log_min = np.min(variance_log)
        # variance_log_max = np.max(variance_log)
        # if variance_log_max - variance_log_min < 1e-5:
        #     variance_log_norm = np.zeros_like(variance_log, dtype=np.uint8)
        # else:   
        #     variance_log_norm = ((variance_log - variance_log_min) / (variance_log_max - variance_log_min) * 255).astype(np.uint8) 
        # variance_log_norm_heatmap = cv2.applyColorMap(variance_log_norm, cv2.COLORMAP_JET)
        # variance_log_norm_heatmap = variance_log_norm_heatmap[:,:, ::-1]  # BGR to RGB
        
        # uncertainty_dir = os.path.join('spring', 'spring_submissio_448x1024', sequence, f'uncertainty_{dir}_{cam}')
        # if not os.path.exists(uncertainty_dir):
        #     os.makedirs(uncertainty_dir)
        # uncertainty_file = os.path.join(uncertainty_dir, f'uncertainty_{dir}_{cam}_%04d.png' % frame)
        # cv2.imwrite(uncertainty_file, variance_log_norm_heatmap[:, :, [2,1,0]])  # RGB to BGR for saving
        
        
        
        #########################
        # useage example:
        # python -m spring.evaluate_spring_submission --config S_M