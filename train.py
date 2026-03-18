from __future__ import print_function, division
import sys
# sys.path.append('core')
import matplotlib
matplotlib.use('Agg')

import argparse
import os
import cv2
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from core import optimizer
import core.flow_datasets as datasets
from core.losses.get_loss import get_loss
from core.losses.flow_loss import sequence_uncertainty_loss, sequence_loss_raft
from core.optimizer import fetch_optimizer
from core.utils.misc import process_cfg
import logging
from core.utils.logger import Logger, save_ckpt, setup_logger, AverageMeter, format_moving_averages_as_progress_dict
import evaluate
from core.Networks import build_network
import pytz
from core.utils.transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from copy import deepcopy
from core.utils.transforms.obj_cache_sem import ObjectCache_sem, add_fake_object_sem 
from core.utils.warp_utils import flow_warp
torch.set_num_threads(4)
cv2.setNumThreads(0)
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove 'module.'
        else:
            new_state_dict[k] = v
    return new_state_dict

def init_distributed_mode(args):
    args.is_main_process = True
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        if args.rank == 0:
            args.is_main_process = True
        else:
            args.is_main_process = False
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)

def train(cfg):

    init_distributed_mode(cfg)
    if cfg.is_main_process:
        process_cfg(cfg)
    logger = Logger(cfg, is_main_process=cfg.is_main_process)

    loss_func = get_loss(cfg.loss)
    sp_transform = RandomAffineFlow(cfg.st_cfg, addnoise=cfg.st_cfg.add_noise).cuda()

    model = build_network(cfg)
    model.init_weights()
    model.train()

    if cfg.restore_ckpt is not None:
        logger.logger.info("[Loading ckpt from {}]".format(cfg.restore_ckpt))
        ckpt = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = remove_module_prefix(ckpt)
        model.load_state_dict(ckpt, strict=True)
        del ckpt

    current_step = 0
    best_val_epe = 1e10
    best_val_step = 0
    if cfg.resume is not None:
        ckpt_resume = cfg.resume
        state_resume = cfg.resume.replace('.pth', '.state')
        if os.path.exists(state_resume):
            ckpt = torch.load(ckpt_resume, map_location='cpu')
            ckpt = remove_module_prefix(ckpt)
            state = torch.load(state_resume, map_location='cpu')
            logger.logger.info("Loading ckpt & state from:\n{}\n{}".format(ckpt_resume, state_resume))
            model.load_state_dict(ckpt, strict=True)
            del ckpt

            current_step = state["iter"]
            best_val_epe = state["best_val_epe"]
            best_val_step = state["best_val_step"]
            logger.logger.info("current_step : {:6d} \nlearning_rate : {:10.7f}".format(current_step, state["learning_rate"]))
            logger.logger.info("best_val_epe : {:10.7f} , best_val_step : {:6d}".format(state["best_val_epe"], state["best_val_step"]))
        else:
            raise FileNotFoundError("Wrong resume value.")

    if cfg.distributed:
        model = model.to('cuda')
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            # find_unused_parameters=False,
            find_unused_parameters=True,
            broadcast_buffers=False
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in cfg.gpus])
        model = nn.DataParallel(model)
        model = model.cuda()

    logger.logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.distributed:
        train_loader, sampler_train = datasets.fetch_dataloader(cfg)
    else:
        train_loader = datasets.fetch_dataloader(cfg)

    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)
    if cfg.resume is not None and 'state' in locals():
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        del state

    logger.logger.info("total_step : {:6d}".format(int(cfg.trainer.num_steps)))
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger.model = model
    logger.scheduler = scheduler
    logger.total_steps = current_step
    should_keep_training = True

    timing_meter_names = [
        "1_data_loading",
        "2_main_forward",
        "3_main_loss",
        "4_atst",
        "5_vel_sem_aug",
        "6_bakward_update",
    ]
    timing_meters = AverageMeter(i=len(timing_meter_names))
    obj_cache = None
    while should_keep_training:
        if 'stage2' in cfg and (logger.total_steps/(cfg.trainer.epoch_size * (4/cfg.total_batch_size))) >= cfg.stage2.epoch:
            loss_func.cfg.update(cfg.stage2.loss)
            print(cfg.stage2.trainer)
            cfg.trainer.update(cfg.stage2.trainer)
            logger.logger.info(f"*************  stage2 start_run_atst  total_steps:{logger.total_steps}  epoch:{cfg.stage2.epoch}  ***************")
            cfg.pop('stage2')
        if 'stage3' in cfg and (logger.total_steps/(cfg.trainer.epoch_size * (4/cfg.total_batch_size))) >= cfg.stage3.epoch:
            print(cfg.stage3.trainer)
            cfg.trainer.update(cfg.stage3.trainer)
            loss_func.cfg.update(cfg.stage3.loss)
            if 'run_sem_aug' in cfg.trainer and cfg.trainer.run_sem_aug:
                obj_cache = None
                obj_cache = ObjectCache_sem(cache_size=100)
                if 'ConcatDataset' in str(train_loader.dataset.__class__):
                    for ds in train_loader.dataset.datasets:
                        ds.start_sem_loss = True
                else:
                    train_loader.dataset.start_sem_loss = True
            logger.logger.info(f"*************  stage3 start_chain_velocity  total_steps:{logger.total_steps}  epoch:{cfg.stage3.epoch}  ***************")
            cfg.pop('stage3')
            
        last_time = time.time()

        if cfg.distributed:
            sampler_train.set_epoch(logger.total_steps // cfg.trainer.epoch_size)  #  epoch_size  = len(train_loader)
        if cfg.is_main_process:
            progress = tqdm(total=cfg.trainer.epoch_size, desc="Train")
        moving_averages_dict = None
        for i_batch, data_blob in enumerate(train_loader):
            # # if i_batch_start > 0:
            # #     i_batch_start -= 1
            # #     continue
            # if i_batch >= cfg.trainer.epoch_size:
            #     break
            metrics = {}
            batch_timing = []
            tic = time.time()
            optimizer.zero_grad()
            images_, images_ph_, images_origin, crop_start  = [x.cuda() for x in data_blob[:4]]   #  images_origin   4  5
            semantics_roi = data_blob[4].cuda()
            full_segs_ = data_blob[5].cuda()
            images = images_[:, -2:, ... ]
            images_ph = images_ph_[:, -2:, ...]
            if full_segs_.dim() != 5:  
                full_segs_ = torch.zeros_like(images_ph)
            full_segs = full_segs_[:, -2:, ...]
            # timing: 1_data_loading
            batch_timing.append(time.time() - last_time)
            last_time = time.time()
            # run 1st pass
            res_dict = model(images) # B, N, _, H, W = images.shape
            flows_f12, flows_b21 = res_dict['flows_f12'], res_dict['flows_b21']
            # timing: 2_main_forward
            batch_timing.append(time.time() - last_time)
            last_time = time.time()
            flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in zip(flows_f12, flows_b21)]
            loss, l_ph, l_sm, flow_mean = loss_func(flows, images[:, 0, ... ], images[:, 1, ... ], res_dict, 
                                                                            full_segs[:, 0, ...], full_segs[:, 1, ...])
            metrics['p'] = l_ph.item()
            metrics['s'] = l_sm.item()
            # timing: 3_main_loss
            batch_timing.append(time.time() - last_time)
            last_time = time.time()
            ## ARFlow appearance/spatial transform
            noc_ori_12 = loss_func.vis_mask1.detach()
            flow_ori_12 = flows_f12[-1].detach()
            img1, img2 = images_ph[:, 0, ...], images_ph[:, 1, ...]
            confidence_max = 0
            confidence_mean = 0
            if cfg.trainer.run_atst:
                if 'uncertainty_f12' in res_dict:
                    uncertainty_f12_ori = res_dict['uncertainty_f12'][-1].detach().float()
                else:
                    uncertainty_f12_ori = torch.zeros_like(flow_ori_12[:, :1, ...])
                s = {'imgs': [img1, img2], 'flows_f': [flow_ori_12], 'masks_f': [noc_ori_12], 'uncer': [uncertainty_f12_ori]}
                st_res = sp_transform(deepcopy(s)) if cfg.trainer.run_st else deepcopy(s)
                flow_t, noc_t = st_res['flows_f'], st_res['masks_f']
                unc12_t = st_res['uncer'][0]
                # run another pass
                flow_ar = model(torch.stack(st_res['imgs'], dim=1), only_fw=True)
                flow_t_pred = flow_ar['flows_f12']
                uncertainty_f12 = flow_ar['uncertainty_f12']
                confidence_max = (uncertainty_f12[-1].exp()).max().item()
                confidence_mean = (uncertainty_f12[-1].exp()).mean().item()

                l_atst_unc = sequence_uncertainty_loss(flow_t_pred, uncertainty_f12, flow_t[0], noc_t[0], decouple=cfg.trainer.w_ar_u_decoupled)  # sequence_uncertainty_loss
                metrics['atst_u'] = l_atst_unc.item()
                if cfg.trainer.w_ar_u_decoupled:
                    loss += cfg.trainer.w_ar_u * l_atst_unc
                else:
                    loss += cfg.trainer.w_ar * l_atst_unc
                    
                if cfg.trainer.w_ar_u_decoupled:
                    l_atst = sequence_loss_raft(flow_t_pred, flow_t[0], noc_t[0])      
                    metrics['atst'] = l_atst.item()
                    loss += cfg.trainer.w_ar * l_atst
                    
            if 'atst' not in metrics:
                metrics['atst'] = 0.0
            if 'atst_u' not in metrics:
                metrics['atst_u'] = 0.0
            # timing: 4_atst
            batch_timing.append(time.time() - last_time)
            last_time = time.time()
            
            if hasattr(cfg.trainer, 'run_sem_aug') and cfg.trainer.run_sem_aug: # 
                img1_ot = img1
                img2_ot = img2
                B = img1.shape[0]
                flow_ot_12 = flow_ori_12.clone()
                noc_ot_12 = noc_ori_12.clone()
                obj_mask_total = torch.zeros_like(noc_ot_12)
                if obj_cache.count == obj_cache.cache_size:
                    all_obj_mask, all_img_src, all_mean_flow = obj_cache.pop(B*3, with_aug=True)
                    for _round in range(3):
                        obj_mask = all_obj_mask[_round * B : (_round + 1) * B]
                        img_src = all_img_src[_round * B : (_round + 1) * B]
                        mean_flow = all_mean_flow[_round * B : (_round + 1) * B]

                        input_dict = {
                            "img1_tgt": img1_ot,
                            "img2_tgt": img2_ot,
                            "flow_tgt": flow_ot_12,
                            "noc_tgt": noc_ot_12,
                            "img_src": img_src.cuda(),
                            "obj_mask": obj_mask.cuda(),
                            "motion": mean_flow.cuda(),
                            "obj_mask_total": obj_mask_total,
                        }

                        img1_ot, img2_ot, flow_ot_12, noc_ot_12, obj_mask_total = add_fake_object_sem(input_dict)
        
                    # run another pass
                    flow_sam_pred = model(torch.stack([img1_ot, img2_ot], dim=1), only_fw=True)
                    flow_ot_pred = flow_sam_pred['flows_f12']
                    if hasattr(cfg.trainer, 'w_sem') and cfg.trainer.w_sem > 0:
                        if 'uncertainty_f12' in flow_sam_pred:
                            uncertainty_f12 = flow_sam_pred['uncertainty_f12']
                            uncertainty_f12_ori = res_dict['uncertainty_f12'][-1].detach().float()
                            l_sem = sequence_loss_raft(flow_ot_pred, flow_ot_12, noc_ot_12)
                                   
                            metrics['a_sem'] = l_sem.item()
                            loss += cfg.trainer.w_sem * l_sem 
                    
                # push current object mask into cache for future use
                valid_idx = ~torch.isnan(semantics_roi[:, 0, 0, 0])  #  b 1 h w
                if valid_idx.sum() > 0: # at least one valid object
                        obj_mask = semantics_roi[valid_idx] #   B(valid) 1 H W
                        img_ph = images_ph[valid_idx][:, 0, ... ]   # B(valid) 3 H W
                        # img = images[valid_idx][:, 0, ... ]

                        mean_flow = (flow_ori_12[valid_idx] * obj_mask).mean(dim=[2, 3] ) / obj_mask.mean(dim=[2, 3])
                        obj_cache.push(obj_mask.cpu(), img_ph.cpu(), mean_flow.cpu())


            if 'a_sem' not in metrics:
                metrics['a_sem'] = 0.0

            metrics['t_loss'] = loss.item()
            # timing: 5_vel_sem_aug
            batch_timing.append(time.time() - last_time)
            last_time = time.time()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            # timing: 6_bakward_update
            batch_timing.append(time.time() - last_time)
            
            if cfg.is_main_process:
                if moving_averages_dict is None:
                    moving_averages_dict = {
                        key: AverageMeter() for key in metrics.keys()
                    }
                for key, loss in metrics.items():
                    # if math.isnan(loss): # if torch.isnan(loss):
                    #     continue
                    if key not in moving_averages_dict:
                        moving_averages_dict[key] = AverageMeter()
                    moving_averages_dict[key].update(loss)

                # view statistics in progress bar
                progress_stats = format_moving_averages_as_progress_dict(
                    moving_averages_dict=moving_averages_dict,
                    moving_averages_postfix="")    # "_ema"
            
                progress.set_postfix(progress_stats)
                progress.update(1)

            timing_meters.update(batch_timing)
            # metrics.update(output)
            toc = time.time()
            metrics['time'] = toc - tic
            logger.push(metrics)           
            if logger.total_steps % cfg.sum_freq == 0 and cfg.is_main_process:
                progress.close()
                logger._print_training_status()
                logger.logger.info('confidence_max : %.3f ,  mean: %.3f' % (confidence_max, confidence_mean))
                save_ckpt(logger.total_steps , scheduler, optimizer, model, cfg, [best_val_epe, best_val_step], latest=True) 
                # for v, name in zip(timing_meters.avg, timing_meter_names):
                #     logger.logger.info("timing_batch_avg/%s: %s, total steps: %s", name, v, logger.total_steps)
                timing_meters.reset(i=len(timing_meter_names))
                if hasattr(cfg, 'downtime') and cfg.downtime is not None: # 
                    if logger.get_timestamp( ) >= cfg.downtime: # '09:30'
                        logger.logger.info("Arrival stop time (downtime: %s)"% (cfg.downtime))
                        should_keep_training = False
                        break

            if logger.total_steps % cfg.val_freq == 0 and cfg.is_main_process:
                if hasattr(model, 'require_forward_param_sync'):
                    save_flag = model.require_forward_param_sync
                    model.require_forward_param_sync = False

                PATH = '%s/%d_%s.pth' % (cfg.log_dir, logger.total_steps, cfg.name)
                # torch.save(model.state_dict(), PATH)
                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == 'kitti2015': # sintel_train
                        if hasattr(cfg, 'test_shape'):
                            test_shape = cfg.test_shape
                        else:
                            test_shape = [256, 832]   #   [256, 832]   488, 1144
                        results.update(evaluate.validate_kitti_2015(model = model.module, test_shape=test_shape, logger = logger))
                epeval = 1e10
                for key in results:
                    if 'epe' in key:
                        epeval = results[key]
                        break
                ############# if new best #############
                if epeval <= best_val_epe:
                    best_val_epe = epeval
                    best_val_step = logger.total_steps
                    save_ckpt(logger.total_steps, scheduler, optimizer, model, cfg, [best_val_epe, best_val_step], latest=False)

                # save_ckpt(logger.total_steps , scheduler, optimizer, model, cfg, [best_val_epe, best_val_step], latest=True)    
                logger.logger.info(
                "Validation EPE: %.3f, fl: %.3f, current best EPE: %.3f(step: %s), noc: %.3f, occ: %.3f"
                                    % (epeval, results['fl'], best_val_epe, best_val_step, results['noc'], results['occ']))
                logger.write_dict(results)
                model.train()
                if hasattr(model, 'require_forward_param_sync'):
                    model.require_forward_param_sync = save_flag
                    
            if logger.total_steps >= cfg.trainer.num_steps:
                should_keep_training = False
                break

            if i_batch >= cfg.trainer.epoch_size-1:
                break

            last_time = time.time()

    logger.close()
    if cfg.is_main_process:
        PATH = cfg.log_dir + '/final'
        torch.save(model.state_dict(), PATH)

    # return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="determines which dataset to use for training") # stage

    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()

    import importlib
    cfg_module = importlib.import_module(f"configs.{args.config}")
    get_cfg = cfg_module.get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    

    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.manual_seed(1234)
    # torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(1234)
    random.seed(1234)
    train(cfg)
