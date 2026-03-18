from yacs.config import CfgNode as CN
import torch
import os
_CN = CN()
if ('RANK' in os.environ and 'WORLD_SIZE' in os.environ) or ('SLURM_PROCID' in os.environ):
    distributed = True
    num_gpus = torch.cuda.device_count()
else:
    distributed = False

_CN.name = 'KITTI_MV'
_CN.suffix =''
_CN.gpus = [7,5]
_CN.total_batch_size = 4
if distributed:
    _CN.batch_size = int(_CN.total_batch_size/num_gpus)
else:
    _CN.batch_size = _CN.total_batch_size
_CN.sum_freq = 1000
_CN.val_freq = 5000
_CN.image_size = [256, 832]
_CN.stage = 'MV_2stage_Sem' 
_CN.validation = ['kitti2015']
_CN.fw_crop =   False
_CN.critical_params = ['w_ar_u', 'w_sem']

_CN.network = 'raft_2f_u'
_CN.mixed_precision = True

_CN.restore_ckpt = None
_CN.resume = None

_CN.raft_2f_u = CN()
_CN.raft_2f_u.mixed_precision = _CN.mixed_precision
_CN.raft_2f_u.iters = 12
_CN.raft_2f_u.critical_params = []
### TRAINER
_CN.trainer = CN()
_CN.trainer.w_ar_u = 0.005
_CN.trainer.w_ar_u_decoupled = True
_CN.trainer.w_sem = 0.05

_CN.trainer.optimizer = 'adam'
# # # for optimizer
_CN.trainer.bias_decay = 0
_CN.trainer.weight_decay = 1e-06
_CN.trainer.momentum = 0.9
_CN.trainer.beta = 0.999
_CN.trainer.canonical_lr = 0.0002 #  25e-5
# #
_CN.trainer.scheduler = 'OneCycleLR'
# # # for OneCycleLR
_CN.trainer.max_lr = 0.00025
_CN.trainer.pct_start = 0.05
_CN.trainer.cycle_momentum = False
_CN.trainer.anneal_strategy = 'linear'
# #
_CN.trainer.clip =  1.0 #  1.0
_CN.trainer.epoch_size = 1000  #   1000
_CN.trainer.num_steps = _CN.trainer.epoch_size * 100 * (4/_CN.total_batch_size)

_CN.trainer.run_atst = False
_CN.trainer.run_st = False
_CN.trainer.w_ar = 0.02   #  0.02

_CN.stage2 = CN()
_CN.stage2.epoch = 0

_CN.stage2.loss = CN()
_CN.stage2.loss.occ_from_back = False
_CN.stage2.loss.w_l1 = 0.0
_CN.stage2.loss.w_ssim = 0.0
_CN.stage2.loss.w_ternary = 1.0

_CN.stage2.trainer = CN()
_CN.stage2.trainer.run_atst = True
_CN.stage2.trainer.run_st = True

_CN.stage3 = CN()
_CN.stage3.epoch = 50
_CN.stage3.trainer = CN()

_CN.stage3.trainer.run_sem_aug = True
_CN.stage3.loss = CN()
_CN.stage3.loss.w_smooth = 0.1  #  0.1


_CN.loss = CN()
_CN.loss.edge_aware_alpha = 10
_CN.loss.occ_from_back = True
_CN.loss.smooth_2nd = True
_CN.loss.type = 'unFlowLoss_Raft'
_CN.loss.w_l1 = 0.15
_CN.loss.w_smooth = 0.0
_CN.loss.smooth_type = 'homography'
_CN.loss.ransac_threshold = 0.5   # sintel 3
_CN.loss.hm_no_unc = False
_CN.loss.w_ssim = 0.85
_CN.loss.w_ternary = 0.0
_CN.loss.warp_pad = 'border'
_CN.loss.with_bk = True

_CN.st_cfg = CN()
_CN.st_cfg.add_noise = True
_CN.st_cfg.hflip = True
_CN.st_cfg.rotate = [-0.01, 0.01, -0.01, 0.01]
_CN.st_cfg.squeeze = [1.0, 1.0, 1.0, 1.0]
_CN.st_cfg.trans = [0.04, 0.005]
_CN.st_cfg.vflip = False
_CN.st_cfg.zoom = [1.0, 1.4, 0.99, 1.01]


def get_cfg():
    return _CN.clone()
