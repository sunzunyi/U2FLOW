import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from .loss_blocks import SSIM, smooth_grad_1st, smooth_grad_2nd, TernaryLoss, smooth_homography
from ..utils.warp_utils import flow_warp
from ..utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward


class unFlowLoss_Raft(nn.modules.Module):
    def __init__(self, cfg):
        super(unFlowLoss_Raft, self).__init__()
        self.cfg = cfg
        self.alpha = 0.1
            
    def loss_photomatric(self, im1_scaled, im1_recons, vis_mask1, level):
        loss = []
        if self.cfg.w_l1 > 0:
            loss += [self.cfg.w_l1 * (im1_scaled - im1_recons).abs() * vis_mask1]

        if self.cfg.w_ssim > 0:
            loss += [self.cfg.w_ssim * SSIM(im1_recons * vis_mask1,
                                            im1_scaled * vis_mask1)]

        if self.cfg.w_ternary > 0:
            loss += [self.cfg.w_ternary * TernaryLoss(im1_recons * vis_mask1,
                                                      im1_scaled * vis_mask1)]
        return sum([l.mean() for l in loss]) / (vis_mask1.mean() + 1e-6)

    def loss_smooth(self, flow, im1_scaled):
        if 'smooth_2nd' in self.cfg and self.cfg.smooth_2nd:
            func_smooth = smooth_grad_2nd
        else:
            func_smooth = smooth_grad_1st
            
        loss = []
        loss += [func_smooth(flow, im1_scaled, edge='image', alpha=self.cfg.edge_aware_alpha)]
        return sum([l.mean() for l in loss])

    def loss_smooth_homography(self, flow, full_seg, occ_mask, img, uncertainty=None):
        loss = smooth_homography(
            flow,
            full_seg=full_seg,
            occ_mask=occ_mask,
            img = img,
            ransac_threshold=self.cfg.ransac_threshold,
            uncertainty=uncertainty,
            hm_no_unc=self.cfg.hm_no_unc,
        )
        return loss
    
    def forward(self, pyramid_flows, im1_origin, im2_origin, res_dict, full_seg1, full_seg2, occ_aware=True, im1_origin_fw = None, im2_origin_fw = None, crop_start = None):
        """

        :param output: Multi-scale forward/backward flows n * [B x 4 x h x w]
        :param target: image pairs Nx6xHxW
        :return:
        """
        DEVICE = pyramid_flows[0].device
        
        top_flow = pyramid_flows[-1]
        total_iters = len(pyramid_flows)
        # if self.cfg.occ_from_back:
        #     self.vis_mask1 = 1 - get_occu_mask_backward(top_flow[:, 2:], th=0.2)
        #     self.vis_mask2 = 1 - get_occu_mask_backward(top_flow[:, :2], th=0.2)
        # else:
        #     self.vis_mask1 = 1 - get_occu_mask_bidirection(top_flow[:, :2], top_flow[:, 2:])
        #     self.vis_mask2 = 1 - get_occu_mask_bidirection(top_flow[:, 2:], top_flow[:, :2])
   
        scale = min(*top_flow.shape[-2:])

        # compute losses at each level
        pyramid_warp_losses = []
        pyramid_smooth_losses = []
        zero_loss = torch.tensor(0, dtype=torch.float32, device=DEVICE)
        
        for i, flow in enumerate(pyramid_flows):
            if self.cfg.occ_from_back:
                self.vis_mask1 = 1 - get_occu_mask_backward(flow[:, 2:], th=0.2)
                self.vis_mask2 = 1 - get_occu_mask_backward(flow[:, :2], th=0.2)
            else:
                self.vis_mask1 = 1 - get_occu_mask_bidirection(flow[:, :2], flow[:, 2:])
                self.vis_mask2 = 1 - get_occu_mask_bidirection(flow[:, 2:], flow[:, :2])
            # photometric loss
            im1_recons = flow_warp(im2_origin, flow[:, :2], pad=self.cfg.warp_pad)
            im2_recons = flow_warp(im1_origin, flow[:, 2:], pad=self.cfg.warp_pad)
            
            loss_warp = self.loss_photomatric(im1_origin, im1_recons, self.vis_mask1, i)
            if self.cfg.with_bk:
                loss_warp_b = self.loss_photomatric(im2_origin, im2_recons, self.vis_mask2, i)  
                loss_warp += loss_warp_b 
                loss_warp /= 2.

            pyramid_warp_losses.append(loss_warp)
            # smoothness loss
            if self.cfg.w_smooth > 0:
                if hasattr(self.cfg, 'smooth_type') and self.cfg.smooth_type == "homography" and i == total_iters-1:
                    loss_smooth = self.loss_smooth_homography(
                        flow[:, :2],
                        full_seg=full_seg1,
                        occ_mask=1 - self.vis_mask1,
                        img = im1_origin,
                        uncertainty=res_dict['uncertainty_f12'][i] if 'uncertainty_f12' in res_dict else None,
                    )
                    if self.cfg.with_bk:
                        loss_smooth += self.loss_smooth_homography(
                            flow[:, 2:],
                            full_seg=full_seg2,
                            occ_mask=1 - self.vis_mask2,
                            img = im2_origin,
                            uncertainty=res_dict['uncertainty_b21'][i] if 'uncertainty_b21' in res_dict else None,
                        )
                        loss_smooth /= 2.0

                else:    
                    loss_smooth = self.loss_smooth(flow[:, :2]/ scale, im1_origin)
                    if self.cfg.with_bk:
                        loss_smooth += self.loss_smooth(flow[:, 2:]/ scale, im2_origin)
                        loss_smooth /= 2.

                pyramid_smooth_losses.append(loss_smooth)
            
            else:
                pyramid_smooth_losses.append(zero_loss)
                
        
        # aggregate losses  
        n_prediction = len(pyramid_flows)
        for ii in range(len(pyramid_warp_losses)):
            i_weight = 0.8**(n_prediction - ii - 1) 
            pyramid_warp_losses[ii] *= i_weight
            pyramid_smooth_losses[ii] *= i_weight

        l_ph = sum(pyramid_warp_losses)
        l_sm = sum(pyramid_smooth_losses)

        total_loss = l_ph +  self.cfg.w_smooth * l_sm
        
        return total_loss, l_ph, l_sm, pyramid_flows[0].abs().mean()


import math
def sequence_uncertainty_loss(flow_preds, uncertainty_preds, flow_gt, valid, gamma=0.8, decouple=True):
    n_predictions = len(flow_preds)
    uncertainty_loss = 0.0

    valid = (valid >= 0.5).float()
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_alpha = uncertainty_preds[i]
        unc_loss_comp = torch.abs(flow_preds[i] - flow_gt).sum(dim=1, keepdim=True)  # L1 norm per-pixel
        if decouple:
            unc_loss_comp = unc_loss_comp.detach()
        i_loss = math.sqrt(2.0) * torch.exp(-0.5 * i_alpha) * unc_loss_comp + 0.5 * i_alpha
        uncertainty_loss += i_weight * (valid[:, None] * i_loss).sum() / (valid.sum() + 1e-7)

    return uncertainty_loss

def sequence_loss_raft(flow_preds, flow_gt, valid, gamma=0.8):

    n_predictions = len(flow_preds)
    total_loss = 0.0

    valid = (valid >= 0.5).float()
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        loss_fun = torch.nn.SmoothL1Loss(reduction='none')
        flow_loss = (flow_preds[i] - flow_gt).abs()
        loss_i = loss_fun(flow_loss, torch.zeros_like(flow_loss))
        total_loss += (valid * loss_i).sum() / (valid.sum() + 1e-7) * i_weight
        
    return total_loss

