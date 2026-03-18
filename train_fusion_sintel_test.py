import sys
sys.path.append('core')
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from typing import Tuple, Dict
import argparse
import os
from core.Networks import build_network
from core import flow_datasets, flow_datasets_sintel
from core.flow_datasets import KITTI2015_MV, KITTI2012_MV
from tqdm import tqdm
from core.utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward
from core.utils import frame_utils
from core.utils.flow_utils import flow_to_image, resize_flow, writeFlowKITTI
import numpy as np
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import cv2


class TinyModel(nn.Module):
    """Tiny model that should resemble the motion model.
    """
    def __init__(self):
        super(TinyModel, self).__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels=2+2, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1, bias=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self._layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def init_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = x * (-1.0)
        x = self._layers(x)
        return x


def _positions_center_origin(height: int, width: int):
    """Returns image coordinates where the origin is at the image center.
    """
    h = torch.linspace(0.0, height - 1, steps=height, dtype=torch.float32)
    h = h / (height - 1) * 2 - 1
    w = torch.linspace(0.0, width - 1, steps=width, dtype=torch.float32)
    w = w / (width - 1) * 2 - 1
    h_grid, w_grid = torch.meshgrid(h, w, indexing='ij')
    return torch.stack([h_grid, w_grid], dim=-1)

coords = None
iters = 2700
thr = 45.0
lr = 0.01
def train_and_run_tiny_model(
    flow_forward,
    flow_backward,
    mask_forward,
    mask_backward,
    uncertainty_forward,
    uncertainty_backward,
    iterations: int = iters,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size = flow_backward.shape[0]
    height = flow_backward.shape[2]
    width = flow_backward.shape[3]
    global coords, thr, lr, iters
    if coords is None or coords.shape[1] != height or coords.shape[2] != width:
        coords = _positions_center_origin(height, width).to(device)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        coords = coords.permute(0, 3, 1, 2)
    net_input = torch.cat([flow_backward, coords], dim=1)

    # # Compute valid mask based on occlusion masks
    # valid_mask = mask_forward * mask_backward

    # # Compute valid mask based on uncertainty
    sigma2_f = uncertainty_forward.exp().clamp(min=1e-3, max=200)
    sigma2_b = uncertainty_backward.exp().clamp(min=1e-3, max=200)
    valid_mask = (sigma2_f < thr) & (sigma2_b < thr) 
    valid_mask = valid_mask.float()
    mask_forward = (sigma2_f < thr).float()
    mask_backward = (sigma2_b < thr).float()
    ################################################

    if valid_mask.sum() < 1:
        return flow_forward, mask_forward

    model = TinyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    model.train()
    for step in range(iterations):
        optimizer.zero_grad()
        pred = model(net_input)
        error = torch.sqrt(torch.sum((pred - flow_forward) ** 2, dim=1, keepdim=True))
        loss = (error * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        loss.backward()
        optimizer.step()

        if (step + 1) % (iterations // 20) == 0:
            lr_schedule.step()

    model.eval()
    with torch.no_grad():
        predicted_flow_forward = model(net_input)  # [B,2,H,W]

    mask_backward_no_forward = (1 - mask_forward) * mask_backward
    nan_mask = torch.isnan(predicted_flow_forward).any(dim=1, keepdim=True).float()
    mask_backward_no_forward = mask_backward_no_forward * (1 - nan_mask)

    predicted_flow_forward = torch.nan_to_num(predicted_flow_forward, nan=0.0)

    fused_flow = flow_forward * (1-mask_backward_no_forward) + predicted_flow_forward * mask_backward_no_forward
    fused_mask = torch.clamp(mask_forward + mask_backward, min=0, max=1)

    if torch.isnan(predicted_flow_forward).any() or torch.isinf(predicted_flow_forward).any():
        print("predicted_flow_forward contains NaN or Inf values at val_id:")
        print('fused_flow has nan: ', torch.isnan(fused_flow).any().item(), ' has inf : ', torch.isinf(fused_flow).any().item())

    def check_tensor_nan_inf(name, t):
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"{name}: nan={torch.isnan(t).any().item()}, inf={torch.isinf(t).any().item()}, "
                f"min={t.min().item():.3e}, max={t.max().item():.3e}")

    check_tensor_nan_inf("flow_forward", flow_forward)
    check_tensor_nan_inf("predicted_flow_forward", predicted_flow_forward)
    check_tensor_nan_inf("mask_forward", mask_forward)
    check_tensor_nan_inf("mask_backward", mask_backward)
    check_tensor_nan_inf("mask_backward_no_forward", mask_backward_no_forward)

    return fused_flow, fused_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="S_M", help="config file for training") # stage
    args, unknown_args = parser.parse_known_args()

    if args.config:
        import importlib
        cfg_module = importlib.import_module(f"configs.{args.config}")
        get_cfg = cfg_module.get_cfg

    cfg = get_cfg()
    cfg.mixed_precision = True
    cfg[cfg.network].mixed_precision = True
    model_2f = torch.nn.DataParallel(build_network(cfg))
    
    path_ck = 'checkpoints/sintel.pth'
    model_2f.load_state_dict(torch.load(path_ck), strict=True)       #   cfg.model

    model_2f.cuda()
    model_2f.eval()

    dstype = 'final'   # 'clean' or 'final'
    val_dataset = flow_datasets_sintel.MpiSintel_Submission(dstype = dstype,   # 'clean' or 'final'
        three_frame=True,
        )
    output_local_dir = 'logs/sintel_submission/sintel_submission_3frames'
    ds_dir_local = os.path.join(output_local_dir, dstype)
    os.makedirs(ds_dir_local, exist_ok=True)

    for val_id in tqdm(range(len(val_dataset))): # len(val_dataset)    
        images, extra_info = val_dataset[val_id]

        scene, frame_id = extra_info["extra_info"][0].split("/")[-2:]
        filename = frame_id[:5] + frame_id[6:10] + ".flo"
        output_file_local = os.path.join(ds_dir_local, scene, filename)
        if os.path.exists(output_file_local):
            print("Flow file already exists, skiping:", output_file_local)
            continue
        images = images[None].cuda()
        image_12 = images[:, 1:, ...]
        with torch.no_grad():
            flows_12 = model_2f(image_12)
        flow12_f = flows_12['flows_f12'][-1]
        flow21_b = flows_12['flows_b21'][-1]
        uncertainty_f12 = flows_12['uncertainty_f12'][-1]
        vis_mask_f = 1 - get_occu_mask_bidirection(flow12_f, flow21_b)

        image_01 = images[:, :-1, ...]
        with torch.no_grad():
            flows_01 = model_2f(image_01)
        flow01_f = flows_01['flows_f12'][-1]
        flow10_b = flows_01['flows_b21'][-1]
        uncertainty_b10 = flows_01['uncertainty_b21'][-1]
        vis_mask_b = 1 - get_occu_mask_bidirection(flow10_b, flow01_f)

        h, w = extra_info["h_info"], extra_info["w_info"]
        if extra_info["extra_info"][1]:
            fused_flow, fused_mask = train_and_run_tiny_model(flow12_f, flow10_b, vis_mask_f, vis_mask_b, uncertainty_f12, uncertainty_b10)
        else:
            print("Directly use original flow for scene:", extra_info["extra_info"][0])
            fused_flow = flow12_f
        flow_pred_up = resize_flow(fused_flow, (h, w))[0].detach().cpu()   #  flow12_f
        # nan inf
        if torch.isnan(flow_pred_up).any() or torch.isinf(flow_pred_up).any():
            print("Flow contains NaN or Inf values at val_id:", val_id)
            print('inf : ', torch.isinf(flow_pred_up).any().item(), ' has nan : ', torch.isnan(flow_pred_up).any().item())

        scene, frame_id = extra_info["extra_info"][0].split("/")[-2:]
        filename = frame_id[:5] + frame_id[6:10] + ".flo"
        output_file_local = os.path.join(ds_dir_local, scene, filename)

        # wrtie to local and then move to manifold
        flow_pred_up = flow_pred_up.detach().cpu().numpy().transpose([1,2,0])  # H,W,2
        frame_utils.writeFlowSintel(output_file_local, flow_pred_up)

# CUDA_VISIBLE_DEVICES=0 python -u train_fusion_sintel_test.py --config S_M

