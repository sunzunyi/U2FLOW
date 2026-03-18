import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.Networks.extractor import BasicEncoder
from core.Networks.corr import CorrBlock
from core.utils.utils import coords_grid

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
        

from core.Networks.update import BasicUpdateBlock_U, conv
class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        self.only3frames = False

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=args.dropout)
        self.update_block = BasicUpdateBlock_U(self.args, hidden_dim=hdim)
        self.curr_epoch = -1

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

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_flow(self, pred, mask, scale_factor=8):
        N, dim, H, W = pred.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_pred = F.unfold(scale_factor*pred, [3,3], padding=1)
        up_pred = up_pred.view(N, dim, 9, 1, 1, H, W)
        up_pred = torch.sum(mask * up_pred, dim=2)
        up_pred = up_pred.permute(0, 1, 4, 2, 5, 3)
        return up_pred.reshape(N, dim, 8*H, 8*W)
        
    def forward_2_frames(self, img0, img1):
        iters = self.args.iters
        hdim = self.hidden_dim
        cdim = self.context_dim
        B,  _, H, W = img0.shape
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap0, fmap1 = self.fnet([img0, img1], frames=2)
            
        fmap0 = fmap0.float()
        fmap1 = fmap1.float()
        b, c, h, w = fmap0.shape

        corr_fn_forward = CorrBlock(fmap0, fmap1, radius=self.args.corr_radius)
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(img0)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net_f = torch.tanh(net)
            inp_f = torch.relu(inp)
            
        coords0_f = coords_grid(B, H//8, W//8).to(img0.device)
        coords1_f = coords_grid(B, H//8, W//8).to(img0.device)

        flows_f12 = []
        uncertainty_list = []
        for itr in range(iters):
            coords1_f = coords1_f.detach()
            corr_f = corr_fn_forward(coords1_f) # index correlation volume
            flow_f = coords1_f - coords0_f

            with autocast(enabled=self.args.mixed_precision):
                net_f, up_mask_f, flow_f_d, uncertainty = self.update_block(net_f, inp_f, corr_f, flow_f)

            coords1_f = coords1_f + flow_f_d
            
            flow_f_up = self.upsample_flow(coords1_f-coords0_f, up_mask_f)
            uncertainty_up = self.upsample_flow(uncertainty, up_mask_f, scale_factor=1)
            uncertainty_up[~torch.isfinite(uncertainty_up)] = 35
            
            flows_f12.append(flow_f_up)
            uncertainty_list.append(uncertainty_up)
          
        return flows_f12, uncertainty_list

    def forward(self, imgs, only_fw = False):
        res_dict = {}
        if only_fw:
            flows_f12, uncertainty_f12 = self.forward_2_frames(imgs[:, 0, ...], imgs[:, 1, ...])
            res_dict['flows_f12'] = flows_f12
            res_dict['uncertainty_f12'] = uncertainty_f12
            return res_dict

        flows_f12, uncertainty_f12 = self.forward_2_frames(imgs[:, 0, ...], imgs[:, 1, ...])
        flows_b21, uncertainty_b21 = self.forward_2_frames(imgs[:, 1, ...], imgs[:, 0, ...])
        res_dict['flows_f12'] = flows_f12
        res_dict['flows_b21'] = flows_b21
        res_dict['uncertainty_f12'] = uncertainty_f12
        res_dict['uncertainty_b21'] = uncertainty_b21

        return res_dict
