import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.misc import process_cfg
from core.utils import flow_viz
from core import flow_datasets, flow_datasets_sintel

from core.Networks import build_network

from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
import itertools
from tqdm import tqdm
import torch.utils.data as data

import cv2
from scipy.stats import spearmanr
from core.utils.warp_utils import get_occu_mask_bidirection, get_occu_mask_backward

def save_imgs_with_unc(img_name, network, imgs, flows_fw, flow_gt, unc_up, save_dir):
        # imgs N, _, H, W  flow_predictions N-1, 2, H, W  flow_gts N-1, 2, H, W
    imgs = imgs.clone().detach().permute(0, 2, 3, 1).cpu().numpy() * 255.
    flows_fw = flows_fw.clone().detach().permute(1, 2, 0).cpu().numpy()
    # map flow to rgb image
    flows_fw = flow_viz.flow_to_image(flows_fw)

    flow_gt = flow_gt.clone().detach().permute(1, 2, 0).cpu().numpy()
    flow_gt = flow_viz.flow_to_image(flow_gt)

    unc_up = unc_up.squeeze().cpu().numpy()
    # Apply a log compression to σ² (to address contrast issues caused by extreme values).
    variance_log = np.log1p(unc_up)
    variance_log_min = np.min(variance_log)
    variance_log_max = np.max(variance_log)
    if variance_log_max - variance_log_min < 1e-5:
        variance_log_norm = np.zeros_like(variance_log, dtype=np.uint8)
    else:   
        variance_log_norm = ((variance_log - variance_log_min) / (variance_log_max - variance_log_min) * 255).astype(np.uint8) 
    variance_log_norm_heatmap = cv2.applyColorMap(variance_log_norm, cv2.COLORMAP_JET)
    variance_log_norm_heatmap = variance_log_norm_heatmap[:,:, ::-1]  # BGR to RGB


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save uncertainty map
    filename = f"{save_dir}/{img_name}_unc.png"
    cv2.imwrite(filename, variance_log_norm_heatmap[:, :, [2,1,0]])

    # Save flow prediction map
    filename = f"{save_dir}/{img_name}_flow.png"
    cv2.imwrite(filename, flows_fw[:, :, [2,1,0]])

    # Save flow GT map
    filename = f"{save_dir}/{img_name}_flow_gt.png"
    cv2.imwrite(filename, flow_gt[:, :, [2,1,0]])

@torch.no_grad()
def create_sintel_submission(model):
    """Create submission for the Sintel leaderboard"""
    # start inference
    model.eval()
    output_local_dir = 'logs/sintel_submission/sintel_submission_2frames'
    for dstype in ["final", "clean"]:
        ds_dir_local = os.path.join(output_local_dir, dstype)
        os.makedirs(ds_dir_local, exist_ok=True)

        dataset = flow_datasets_sintel.MpiSintel_Submission(dstype=dstype)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, pin_memory=True, shuffle=False
        )

        for data in tqdm(data_loader):
            images, data_info = data
            images = images.cuda()  # [None]
            flows = model(images)
            flow_pred = flows['flows_f12'][-1]

            for i in range(images.shape[0]): # B

                h, w = data_info["h_info"][i], data_info["w_info"][i]
                h, w = h.item(), w.item()
                flow_pred_up = resize_flow(flow_pred[i : (i + 1)], (h, w))

                scene, frame_id = data_info["extra_info"][i].split("/")[-2:]
                filename = frame_id[:5] + frame_id[6:10] + ".flo"
                output_file_local = os.path.join(ds_dir_local, scene, filename)

                # wrtie to local and then move to manifold
                flow_pred_up = flow_pred_up.detach().cpu().numpy().transpose([0, 2, 3, 1])
                frame_utils.writeFlowSintel(output_file_local, flow_pred_up[0])

                # # Save uncertainty map
                # uncertainty_map_batch = flows['uncertainty_f12'][-1]
                # uncertainty_map_batch = torch.exp(uncertainty_map_batch).clamp(min=1e-3, max=200)   # 1e4
                # unc_map = uncertainty_map_batch[i:i+1]  # [1, hf, wf]
                # unc_up = torch.nn.functional.interpolate(unc_map, size=(h, w), mode='area')  # [1,H,W] -> [1,H,W] then .cpu()
                # img_name = filename.replace('.flo', '.png')

                # unc_up = unc_up.squeeze().cpu().numpy()
                # # Apply a log compression to σ² (to address contrast issues caused by extreme values).
                # variance_log = np.log1p(unc_up)
                # variance_log_min = np.min(variance_log)
                # variance_log_max = np.max(variance_log)
                # if variance_log_max - variance_log_min < 1e-5:
                #     variance_log_norm = np.zeros_like(variance_log, dtype=np.uint8)
                # else:   
                #     variance_log_norm = ((variance_log - variance_log_min) / (variance_log_max - variance_log_min) * 255).astype(np.uint8) 
                # variance_log_norm_heatmap = cv2.applyColorMap(variance_log_norm, cv2.COLORMAP_JET)
                # variance_log_norm_heatmap = variance_log_norm_heatmap[:,:, ::-1]  # BGR to RGB


                # vis_unc = os.path.join(output_local_dir, 'vis_unc', dstype, scene)
                # if not os.path.exists(vis_unc):
                #     os.makedirs(vis_unc)
                # # save
                # filename = f"{vis_unc}/{img_name}"
                # cv2.imwrite(filename, variance_log_norm_heatmap[:, :, [2,1,0]])

    print("Completed!")
    return


@torch.no_grad()
def validate_sintel_batch(model, logger = None):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}

    for dstype in ['final', "clean"]: # dstype in ['final', "clean"]:
        val_dataset = flow_datasets_sintel.MpiSintel(dataset_type = dstype,
            split="training",
            subsplit= "trainval"  # "trainval"   "val"    "train"
            )
        val_loader = data.DataLoader(val_dataset, batch_size=8, pin_memory=True, shuffle=False, num_workers=4, drop_last=False)  # *2
        epe_list = []
        noc_epe_error_list, epe_error_occ_list= [], []
        for i_step, data_batch in tqdm(enumerate(val_loader), total=len(val_loader), desc='Validate'):
            images, flow_gt, flowocc_mask, extra_info = data_batch
            images = images.cuda()  # [None]
            flows = model(images)
            flow_pred = flows['flows_f12'][-1] 
            _, h, w = flow_gt[0].size()
            flow_pred_up = resize_flow(flow_pred, (h, w)).detach().cpu()
            for i_batch in range(images.size(0)):

                epe = torch.sum((flow_gt[i_batch] - flow_pred_up[i_batch])**2, dim=0).sqrt()
                epe_list.append(epe.view(-1).numpy())

                pep_error_occ= torch.sum((flow_pred_up[i_batch] - flow_gt[i_batch])**2, dim=0).sqrt().view(-1)
                occ_erea_mask = flowocc_mask[i_batch].view(-1) >= 0.5 
                if (occ_erea_mask == 0).all():
                    epe_error_occ_list.append(0)
                else :
                    epe_error_occ_list.append(pep_error_occ[occ_erea_mask].mean().item())

                noc_epe_error_= torch.sum((flow_gt[i_batch] - flow_pred_up[i_batch])**2, dim=0).sqrt().view(-1)
                val_noc = 1 - flowocc_mask[i_batch]
                val_noc = val_noc.view(-1) >= 0.5
                noc_epe_error_list.append(noc_epe_error_[val_noc].mean().item())

                if 0:
                    uncertainty_map_batch = flows['uncertainty_f12'][-1]
                    uncertainty_map_batch = torch.exp(uncertainty_map_batch).clamp(min=1e-3, max=1e4)
                    unc_map = uncertainty_map_batch[i_batch:i_batch+1]  # [1, hf, wf]
                    unc_up = torch.nn.functional.interpolate(unc_map, size=(h, w), mode='area')  # [1,H,W] -> [1,H,W] then .cpu()
                    img_name = f"frame_{int(extra_info[1][i_batch]):04d}"
                    save_dir = f"./logs/evaluate_sintel_training/{dstype}/{extra_info[0][i_batch]}/"
                    save_imgs_with_unc(img_name, 'sintel', images[i_batch, :, ... ], flow_pred_up[i_batch], flow_gt[i_batch], unc_up[0], save_dir)

                # Save the error map in Sintel style.
                if 0:
                    epe_map = epe.numpy()  # [H, W]
                    # get scene name and frame index
                    scene_name = extra_info[0][i_batch]
                    frame_idx = int(extra_info[1][i_batch])
                    img_name = f"frame_{frame_idx:04d}"

                    # build save dir
                    save_dir = f"./logs/evaluate_sintel_training/{dstype}/{scene_name}/"
                    os.makedirs(save_dir, exist_ok=True)
                
                    # Apply a log compression to σ² (to address contrast issues caused by extreme values).
                    variance_epe = np.log1p(epe_map)
                    variance_epe_min = np.min(variance_epe)
                    variance_epe_max = np.max(variance_epe)
                    if variance_epe_max - variance_epe_min < 1e-5:
                        variance_epe_norm = np.zeros_like(variance_epe, dtype=np.uint8)
                    else:   
                        variance_epe_norm = ((variance_epe - variance_epe_min) / (variance_epe_max - variance_epe_min) * 255).astype(np.uint8) 
                        
                    epe_gray = variance_epe_norm
                    cv2.imwrite(os.path.join(save_dir, f"{img_name}_epe_gray.png"), epe_gray)

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = 0 # np.mean(epe_all<1)
        px3 = 0 # np.mean(epe_all<3)
        px5 = 0 # np.mean(epe_all<5)

        results[dstype + '_epe'] = np.mean(epe_list)


        noc_epe_error_list = np.array(noc_epe_error_list)
        epe_error_occ_list = np.array(epe_error_occ_list)
        noc_epe = np.mean(noc_epe_error_list)
        occ_epe = np.mean(epe_error_occ_list)
        results[dstype + '_noc'] = noc_epe
        results[dstype + '_occ'] = occ_epe

        if logger is not None:
            logger.logger.info(
                "Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, NOC_EPE: %f, OCC_EPE: %f" % (
                    dstype, epe, px1, px3, px5, noc_epe, occ_epe))
        else:
            print(
                "Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f, NOC_EPE: %f, OCC_EPE: %f" % (
                    dstype, epe, px1, px3, px5, noc_epe, occ_epe))


    return {'epe': results['final_epe'], 'fl': 0, 'noc':results['final_noc'], 'occ':results['final_occ'] }


@torch.no_grad()
def validate_sintel_unc_per_image(model,
                                  test_batch_size=8,
                                  intervals=50,
                                  save_dir="results_sintel_unc_per_image",
                                  logger=None,
                                  network='U2Flow'):
    """
    Per-image uncertainty evaluation for Sintel (final, clean, and combined).
    Returns a dictionary with metrics for 'final', 'clean', and 'combined'.

    Requirements/assumptions about model output:
        flows_dict = model(images)
        flows_dict['flows_f12'] -> list of predicted flows (last is final)
        flows_dict['uncertainty_f12'] -> list of log-uncertainty maps (last corresponds to last flow)
    """

    model.eval()
    dstypes = ['final', 'clean']
    results = {}
    os.makedirs(save_dir, exist_ok=True)

    # accumulate for combined
    combined_per_image_metrics = []
    combined_epe_list = []
    combined_noc_epe_list = []
    combined_occ_epe_list = []
    combined_batch_timing = []

    for dstype in dstypes:
        # prepare dataset + loader (same as your existing code)
        val_dataset = flow_datasets_sintel.MpiSintel(dataset_type=dstype,
                                                    split="training",
                                                    subsplit="trainval")
        val_loader = data.DataLoader(val_dataset,
                                     batch_size=test_batch_size,
                                     pin_memory=True,
                                     shuffle=False,
                                     num_workers=4,
                                     drop_last=False)

        per_image_metrics = []   # will store dicts per image
        epe_list = []            # per-image mean EPE to compute dataset EPE average
        noc_epe_error_list = []
        occ_epe_error_list = []
        batch_timing = []

        print(f"Start validating Sintel ({dstype}) per-image uncertainty...")

        # iterate batches
        for i_step, data_batch in tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Validate-{dstype}'):
            images, flow_gt_batch, flowocc_mask_batch, extra_info = data_batch
            images = images.cuda()
            # model forward: expect dict with flows and uncertainty
            last_time = time.time()
            flows_dict = model(images)
            batch_timing.append(time.time() - last_time)

            # predicted flow (last)
            flow_pred_batch = flows_dict['flows_f12'][-1]  # shape [B, 2, Hf, Wf]
            # predicted uncertainty (last) -- assume log-uncertainty like KITTI code
            if 'uncertainty_f12' in flows_dict and flows_dict['uncertainty_f12'] is not None:
                uncertainty_map_batch = flows_dict['uncertainty_f12'][-1]
                uncertainty_map_batch = torch.exp(uncertainty_map_batch).clamp(min=1e-3, max=1e4)
            else:
                raise RuntimeError("Model did not return 'uncertainty_f12'. Need uncertainty map for per-image metrics.")

            # # Compute the uncertainty using forward-backward consistency.
            # flow12_f = flows_dict['flows_f12'][-1]
            # flow21_b = flows_dict['flows_b21'][-1]
            # uncertainty_map_batch = get_occu_mask_bidirection(flow12_f, flow21_b, return_error=True)

            # # Compute the uncertainty using a second-order smoothness constraint.
            # uncertainty_map_batch = estimate_uncertainty(resize_flow(flow_pred_batch, flow_gt_batch.shape[2:]))
                
            # loop images in batch to compute per-image metrics
            for b in range(images.size(0)):
                flow_gt = flow_gt_batch[b]                 # [2, H, W] (cpu tensor)
                flowocc_mask = flowocc_mask_batch[b]       # [H, W] (0/1, 1 = occluded)
                _, h, w = flow_gt.size()

                # resize prediction and uncertainty to ground-truth resolution
                flow_pred = flow_pred_batch[b:b+1]  # keep batch dim for resize helper
                flow_pred_up = resize_flow(flow_pred, (h, w))[0].detach().cpu()  # [2,H,W] cpu

                unc_map = uncertainty_map_batch[b:b+1]  # [1, hf, wf]
                unc_up = torch.nn.functional.interpolate(unc_map, size=(h, w), mode='bilinear', align_corners=False)[0].detach().cpu()  # [1,H,W] -> [1,H,W] then .cpu()
                # flatten
                epe_map = torch.sqrt(torch.sum((flow_gt - flow_pred_up) ** 2, dim=0))  # [H,W]
                epe_flat = epe_map.view(-1).numpy()
                unc_flat = unc_up.view(-1).numpy()

                # Remove NaNs or infs if any
                valid_mask = np.isfinite(epe_flat) & np.isfinite(unc_flat)
                if not np.any(valid_mask):
                    # skip this image if nothing valid
                    continue
                epe_flat = epe_flat[valid_mask]
                unc_flat = unc_flat[valid_mask]

                # store per-image mean epe (for dataset-level EPE)
                epe_mean_img = float(np.mean(epe_flat))
                epe_list.append(epe_mean_img)

                # NOC (non-occluded) EPE
                val_noc = (1 - flowocc_mask).view(-1).numpy().astype(bool)
                val_noc = val_noc[valid_mask]  # align with flattened valid_mask
                if np.any(val_noc):
                    noc_epe = np.mean(epe_flat[val_noc])
                else:
                    noc_epe = 0.0
                noc_epe_error_list.append(noc_epe)

                # OCC (occluded) EPE
                val_occ = flowocc_mask.view(-1).numpy().astype(bool)
                val_occ = val_occ[valid_mask]
                if np.any(val_occ):
                    occ_epe = np.mean(epe_flat[val_occ])
                else:
                    occ_epe = 0.0
                occ_epe_error_list.append(occ_epe)

                # --- per-image uncertainty metrics (AUSE & Spearman corr) ---
                uncert = -unc_flat
                true_uncert = -epe_flat

                quants = [100. / intervals * t for t in range(intervals)]
                plotx = [1. / intervals * t for t in range(intervals + 1)]

                # model sparsification curve (remove pixels with highest predicted uncertainty first)
                thresholds = [np.percentile(uncert, q) for q in quants]
                subs = [(uncert >= t) for t in thresholds]
                sparse_curve = [np.mean(epe_flat[sub]) if np.any(sub) else 0.0 for sub in subs] + [0.0]

                # oracle (optimal) curve (remove pixels with highest true error first)
                opt_thresholds = [np.percentile(true_uncert, q) for q in quants]
                opt_subs = [(true_uncert >= o) for o in opt_thresholds]
                opt_curve = [np.mean(epe_flat[sub]) if np.any(sub) else 0.0 for sub in opt_subs] + [0.0]

                max_val = np.max(opt_curve) + 1e-6
                sparse_curve = np.array(sparse_curve) / max_val
                opt_curve = np.array(opt_curve) / max_val

                ause_i = np.abs(np.trapz(sparse_curve, x=plotx) - np.trapz(opt_curve, x=plotx))
                corr_i, _ = spearmanr(unc_flat, epe_flat)

                per_image_metrics.append({
                    'ause': ause_i,
                    'corr': corr_i,
                    'model_curve': sparse_curve,
                    'oracle_curve': opt_curve,
                    'plotx': plotx
                })

        # --- aggregate dataset-level results for this dstype ---
        epe_arr = np.array(epe_list) if len(epe_list) > 0 else np.array([0.0])
        noc_arr = np.array(noc_epe_error_list) if len(noc_epe_error_list) > 0 else np.array([0.0])
        occ_arr = np.array(occ_epe_error_list) if len(occ_epe_error_list) > 0 else np.array([0.0])

        epe_mean = float(np.mean(epe_arr))
        noc_mean = float(np.mean(noc_arr))
        occ_mean = float(np.mean(occ_arr))
        avg_time = float(np.mean(batch_timing)) if len(batch_timing) > 0 else 0.0

        # per-image aggregated curves and stats
        if len(per_image_metrics) > 0:
            ause_mean = float(np.mean([m['ause'] for m in per_image_metrics]))
            corr_mean = float(np.mean([m['corr'] for m in per_image_metrics]))
            model_curve_mean = np.mean([m['model_curve'] for m in per_image_metrics], axis=0)
            oracle_curve_mean = np.mean([m['oracle_curve'] for m in per_image_metrics], axis=0)
            plotx = per_image_metrics[0]['plotx']
        else:
            ause_mean = 0.0
            corr_mean = 0.0
            model_curve_mean = np.zeros(intervals + 1)
            oracle_curve_mean = np.zeros(intervals + 1)
            plotx = [1. / intervals * t for t in range(intervals + 1)]

        # save per-dstype results
        results[dstype + '_epe'] = epe_mean
        results[dstype + '_noc'] = noc_mean
        results[dstype + '_occ'] = occ_mean
        results[dstype + '_ause_mean'] = ause_mean
        results[dstype + '_corr_mean'] = corr_mean
        results[dstype + '_model_curve_mean'] = model_curve_mean
        results[dstype + '_oracle_curve_mean'] = oracle_curve_mean
        results[dstype + '_plotx'] = plotx
        results[dstype + '_avg_time'] = avg_time

        # logging / printing
        if logger is not None:
            logger.logger.info(
                "Validation (%s) EPE: %f, NOC_EPE: %f, OCC_EPE: %f, AUSE(mean): %f, Corr(mean): %f" %
                (dstype, epe_mean, noc_mean, occ_mean, ause_mean, corr_mean)
            )
        else:
            print("Validation (%s)  EPE: %.6f  NOC_EPE: %.6f  OCC_EPE: %.6f  AUSE(mean): %.6f  Corr(mean): %.4f" %
                  (dstype, epe_mean, noc_mean, occ_mean, ause_mean, corr_mean))

        # save curves and plot
        np.save(os.path.join(save_dir, f"sintel_{dstype}_model_curve_mean.npy"), model_curve_mean)
        np.save(os.path.join(save_dir, f"sintel_{dstype}_oracle_curve_mean.npy"), oracle_curve_mean)
        np.save(os.path.join(save_dir, f"sintel_{dstype}_plotx.npy"), np.array(plotx))

        plt.figure()
        plt.plot(plotx, model_curve_mean, label='Model (avg)')
        plt.plot(plotx, oracle_curve_mean, label='Oracle (avg)')
        plt.xlabel('Removed Fraction')
        plt.ylabel('Normalized Mean EPE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title(f'Sintel-{dstype} Per-image Sparsification Curves (AUSE={ause_mean:.4f})')
        plt.savefig(os.path.join(save_dir, f'sintel_{dstype}_per_image_sparsification_curve.png'), dpi=150)
        plt.close()

        # accumulate for combined
        combined_per_image_metrics += per_image_metrics
        combined_epe_list += epe_list
        combined_noc_epe_list += noc_epe_error_list
        combined_occ_epe_list += occ_epe_error_list
        combined_batch_timing += batch_timing

    # --- combined metrics across final+clean ---
    if len(combined_epe_list) > 0:
        combined_epe_mean = float(np.mean(np.array(combined_epe_list)))
    else:
        combined_epe_mean = 0.0
    combined_noc_mean = float(np.mean(np.array(combined_noc_epe_list))) if len(combined_noc_epe_list) > 0 else 0.0
    combined_occ_mean = float(np.mean(np.array(combined_occ_epe_list))) if len(combined_occ_epe_list) > 0 else 0.0
    combined_avg_time = float(np.mean(np.array(combined_batch_timing))) if len(combined_batch_timing) > 0 else 0.0

    if len(combined_per_image_metrics) > 0:
        combined_ause_mean = float(np.mean([m['ause'] for m in combined_per_image_metrics]))
        combined_corr_mean = float(np.mean([m['corr'] for m in combined_per_image_metrics]))
        combined_model_curve_mean = np.mean([m['model_curve'] for m in combined_per_image_metrics], axis=0)
        combined_oracle_curve_mean = np.mean([m['oracle_curve'] for m in combined_per_image_metrics], axis=0)
        plotx = combined_per_image_metrics[0]['plotx']
    else:
        combined_ause_mean = 0.0
        combined_corr_mean = 0.0
        combined_model_curve_mean = np.zeros(intervals + 1)
        combined_oracle_curve_mean = np.zeros(intervals + 1)
        plotx = [1. / intervals * t for t in range(intervals + 1)]

    results['combined_epe'] = combined_epe_mean
    results['combined_noc'] = combined_noc_mean
    results['combined_occ'] = combined_occ_mean
    results['combined_ause_mean'] = combined_ause_mean
    results['combined_corr_mean'] = combined_corr_mean
    results['combined_model_curve_mean'] = combined_model_curve_mean
    results['combined_oracle_curve_mean'] = combined_oracle_curve_mean
    results['combined_plotx'] = plotx
    results['combined_avg_time'] = combined_avg_time

    # save combined curves
    np.save(os.path.join(save_dir, "sintel_combined_model_curve_mean.npy"), combined_model_curve_mean)
    np.save(os.path.join(save_dir, "sintel_combined_oracle_curve_mean.npy"), combined_oracle_curve_mean)
    np.save(os.path.join(save_dir, "sintel_combined_plotx.npy"), np.array(plotx))

    plt.figure()
    plt.plot(plotx, combined_model_curve_mean, label='Model (avg)')
    plt.plot(plotx, combined_oracle_curve_mean, label='Oracle (avg)')
    plt.xlabel('Removed Fraction')
    plt.ylabel('Normalized Mean EPE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f'Sintel Combined Per-image Sparsification Curves (AUSE={combined_ause_mean:.4f})')
    plt.savefig(os.path.join(save_dir, 'sintel_combined_per_image_sparsification_curve.png'), dpi=150)
    plt.close()

    # print summary
    print("\n========== Validation Sintel (Per-Image Uncertainty) ==========")
    for dstype in dstypes:
        print(f"{dstype.upper()} -- EPE: {results[dstype + '_epe']:.6f}, NOC: {results[dstype + '_noc']:.6f}, "
              f"OCC: {results[dstype + '_occ']:.6f}, AUSE(mean): {results[dstype + '_ause_mean']:.6f}, "
              f"Corr(mean): {results[dstype + '_corr_mean']:.4f}")
    print("----------------------------------------")
    print(f"COMBINED -- EPE: {combined_epe_mean:.6f}, NOC: {combined_noc_mean:.6f}, OCC: {combined_occ_mean:.6f}, "
          f"AUSE(mean): {combined_ause_mean:.6f}, Corr(mean): {combined_corr_mean:.4f}, AvgTime: {combined_avg_time:.4f}s/frame")
    print("===========================================================\n")

    return results

from core.utils.flow_utils import flow_to_image, resize_flow, writeFlowKITTI
@torch.no_grad()
def validate_kitti_2015(model, save_flow=False, network='U2Flow', test_shape=[256, 832], logger = None):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    val_dataset = flow_datasets.KITTI2015(test_shape = test_shape)
    
    epe_list = []
    out_list = []
    noc_epe_error_list, epe_error_occ_list= [], []
    batch_timing = []

    out_noc_list = []
    out_occ_list = []
    for val_id in tqdm(range(len(val_dataset)), desc='Validate'):
        images, flow_gt, valid_gt, flownoc, nocmask = val_dataset[val_id]

        images = images[None].cuda()
        last_time = time.time()
        flows = model(images, only_fw=True)
        flows = flows['flows_f12']
        batch_timing.append(time.time() - last_time)
        _, h, w = flow_gt.size()
        flow_pred = flows[-1]

        flow_pred_up = resize_flow(flow_pred, (h, w))[0].detach().cpu()

        epe = torch.sum((flow_pred_up - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / (mag + 1e-8)) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

        noc_epe_error_ = torch.sum((flow_pred_up - flownoc) ** 2, dim=0).sqrt().view(-1)
        val_noc = nocmask.view(-1) >= 0.5
        if torch.any(val_noc):
            noc_epe_error_list.append(noc_epe_error_[val_noc].mean().item())
        else:
            noc_epe_error_list.append(0.0)
        out_noc = ((epe > 3.0) & ((epe / (mag + 1e-8)) > 0.05)).float()
        if val_noc.sum().item() > 0:
            out_noc_list.append(out_noc[val_noc].cpu().numpy())

        pep_error_occ = torch.sum((flow_pred_up - flow_gt) ** 2, dim=0).sqrt().view(-1)
        occ_erea_mask = valid_gt - nocmask
        if (occ_erea_mask == 0).all():
            epe_error_occ_list.append(0.0)
        else:
            val_occ = occ_erea_mask.view(-1) >= 0.5
            epe_error_occ_list.append(pep_error_occ[val_occ].mean().item())
            out_occ = ((pep_error_occ > 3.0) & ((pep_error_occ / (mag + 1e-8)) > 0.05)).float()
            out_occ_list.append(out_occ[val_occ].cpu().numpy())

        if 0: # save_flow:
            flow_pred_up_save = flow_pred_up.permute(1, 2, 0).numpy()
            frame_id =  f"{val_id:06d}_10.png"
            output_flow = f'./logs/evaluate_kitti_2015_training/flow'
            os.makedirs(output_flow, exist_ok=True)
            output_filename = os.path.join(output_flow, frame_id)
            frame_utils.writeFlowKITTI(output_filename, flow_pred_up_save)

    epe_list = np.array(epe_list)
    noc_epe_error_list = np.array(noc_epe_error_list)
    epe_error_occ_list = np.array(epe_error_occ_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    noc_epe = np.mean(noc_epe_error_list)
    occ_epe = np.mean(epe_error_occ_list)
    fl = 100 * np.mean(out_list)
    avg_time = np.mean(batch_timing)

    if len(out_noc_list) > 0:
        out_noc_all = np.concatenate(out_noc_list)
        f1_noc = 100 * np.mean(out_noc_all)
    else:
        f1_noc = 0.0

    if len(out_occ_list) > 0:
        out_occ_all = np.concatenate(out_occ_list)
        f1_occ = 100 * np.mean(out_occ_all)
    else:
        f1_occ = 0.0
    if logger is not None:
        logger.logger.info("average time: %f", avg_time)
        logger.logger.info("Validation KITTI: %f, %f, noc %f, occ %f, fl-noc %f, fl-occ %f", 
                           epe, fl, noc_epe, occ_epe, f1_noc, f1_occ)
    else:
        print('average time: %f' % avg_time)
        print("Validation KITTI: %f, %f, noc %f, occ %f, fl-noc %f, fl-occ %f" %
            (epe, fl, noc_epe, occ_epe, f1_noc, f1_occ))

    return {
        'epe': epe,
        'fl': fl,
        'noc': noc_epe,
        'occ': occ_epe,
    }


def estimate_uncertainty(flow):
    """
    Estimate optical flow uncertainty using second-order smoothness constraint.
    Args:
        flow: tensor of shape [B, 2, H, W]  (optical flow)
    Returns:
        uncertainty: tensor of shape [B, 1, H, W]  (in [0, 1])
    """
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]

    dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]

    min_h = min(dxx.shape[2], dyy.shape[2])
    min_w = min(dxx.shape[3], dyy.shape[3])
    dxx = dxx[:, :, :min_h, :min_w]
    dyy = dyy[:, :, :min_h, :min_w]

    smoothness_error = torch.abs(dxx) + torch.abs(dyy)
    smoothness_error = smoothness_error.sum(dim=1, keepdim=True)  # sum over flow channels (u,v)

    uncertainty = torch.sigmoid(smoothness_error)

    return uncertainty

@torch.no_grad()
def validate_kitti_unc_per_image(model, save_flow=False, network='U2Flow',
                                      test_shape=[256, 832], save_dir="results_kitti_unc_per_image"):
    """
    Validation on KITTI (per-image uncertainty metrics)
    Adds:
      - per-image curve of "remaining uncertainty mean" after removing top-X% highest uncertainties
      - aggregate mean curve across images, saved and plotted
    """
    model.eval()
    val_dataset_15 = flow_datasets.KITTI2015(test_shape=test_shape)
    val_dataset_12 = flow_datasets.KITTI2012(test_shape=test_shape)
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_15, val_dataset_12])

    epe_list, out_list = [], []
    noc_pep_error_list, pep_error_occ_list = [], []
    batch_timing = []
    per_image_metrics = []  # store per-image AUSE/Corr/curves

    print("Start validating KITTI (per-image uncertainty)...")

    for val_id in tqdm(range(len(val_dataset)), desc='Validate'):
        images, flow_gt, valid_gt, flownoc, nocmask = val_dataset[val_id]
        images = images[None].cuda()

        last_time = time.time()
        flows_dict = model(images, only_fw=False)
        flows = flows_dict['flows_f12']
        uncertainty_map = flows_dict['uncertainty_f12'][-1]
        uncertainty_map = torch.exp(uncertainty_map).clamp(min=1e-3, max=1e4)
        batch_timing.append(time.time() - last_time)

        # # Compute the uncertainty using forward-backward consistency.
        # flow12_f = flows_dict['flows_f12'][-1]
        # flow21_b = flows_dict['flows_b21'][-1]
        # uncertainty_map = get_occu_mask_bidirection(flow12_f, flow21_b, return_error=True)

        # # Compute the uncertainty using a second-order smoothness constraint.
        # uncertainty_map = estimate_uncertainty(resize_flow(flows[-1], flow_gt.shape[1:]))

        _, h, w = flow_gt.size()
        flow_pred = flows[-1]
        flow_pred_up = resize_flow(flow_pred, (h, w))[0].detach().cpu()
        uncertainty = torch.nn.functional.interpolate(
            uncertainty_map, size=(h, w), mode='bilinear', align_corners=False
        )[0].detach().cpu()

        # compute EPE
        epe = torch.sum((flow_pred_up - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()
        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5
        epe_valid = epe[val]
        unc_valid = uncertainty.view(-1)[val]

        # Fl
        out = ((epe_valid > 3.0) & ((epe_valid / (mag[val] + 1e-8)) > 0.05)).float()
        epe_list.append(epe_valid.mean().item())
        out_list.append(out.cpu().numpy())

        # NOC/OCC
        noc_pep_error = torch.sum((flow_pred_up - flownoc) ** 2, dim=0).sqrt().view(-1)
        val_noc = nocmask.view(-1) >= 0.5
        noc_pep_error_list.append(noc_pep_error[val_noc].mean().item())

        pep_error_occ = torch.sum((flow_pred_up - flow_gt) ** 2, dim=0).sqrt().view(-1)
        occ_erea_mask = valid_gt - nocmask
        if (occ_erea_mask == 0).all():
            pep_error_occ_list.append(0)
        else:
            occ_erea_mask = occ_erea_mask.view(-1) >= 0.5
            pep_error_occ_list.append(pep_error_occ[occ_erea_mask].mean().item())

        # --- uncertainty metrics per image (existing AUSE stuff) ---
        epe_np = epe_valid.cpu().numpy()
        unc_np = unc_valid.cpu().numpy()
        uncert = -unc_np
        true_uncert = -epe_np

        intervals = 50
        quants = [100. / intervals * t for t in range(intervals)]
        plotx = [1. / intervals * t for t in range(intervals + 1)]

        thresholds = [np.percentile(uncert, q) for q in quants]
        subs = [(uncert >= t) for t in thresholds]
        sparse_curve = [np.mean(epe_np[sub]) if np.any(sub) else 0.0 for sub in subs] + [0.0]

        opt_thresholds = [np.percentile(true_uncert, q) for q in quants]
        opt_subs = [(true_uncert >= o) for o in opt_thresholds]
        opt_curve = [np.mean(epe_np[sub]) if np.any(sub) else 0.0 for sub in opt_subs] + [0.0]

        max_val = np.max(opt_curve) + 1e-6
        sparse_curve = np.array(sparse_curve) / max_val
        opt_curve = np.array(opt_curve) / max_val

        ause_i = np.abs(np.trapz(sparse_curve, x=plotx) - np.trapz(opt_curve, x=plotx))
        corr_i, _ = spearmanr(unc_np, epe_np)

        per_image_metrics.append({
            'ause': ause_i,
            'corr': corr_i,
            'model_curve': sparse_curve,
            'oracle_curve': opt_curve,
            'plotx': plotx
        })
        if np.any(np.isnan(sparse_curve)) or np.any(np.isnan(opt_curve)):
            print(f"[Warning] NaN detected in curves at image {val_id}")

        # optional save flow
        if save_flow:
            network_name = network + '_2015'
            save_imgs_with_unc(val_id, network_name, images[0, :, ...],
                                  flows['flows_fw'], flows['flows_bw'], epe_valid.mean().item())

    # --- aggregate base KITTI metrics ---
    epe_list = np.array(epe_list)
    noc_pep_error_list = np.array(noc_pep_error_list)
    pep_error_occ_list = np.array(pep_error_occ_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    noc_epe = np.mean(noc_pep_error_list)
    occ_epe = np.mean(pep_error_occ_list)
    fl = 100 * np.mean(out_list)
    avg_time = np.mean(batch_timing)

    # --- aggregate per-image uncertainty metrics ---
    ause_mean = np.mean([m['ause'] for m in per_image_metrics])
    corr_mean = np.mean([m['corr'] for m in per_image_metrics])
    model_curve_mean = np.mean([m['model_curve'] for m in per_image_metrics], axis=0)
    oracle_curve_mean = np.mean([m['oracle_curve'] for m in per_image_metrics], axis=0)
    plotx = per_image_metrics[0]['plotx']

    # print aggregated summary
    print("----------------------------------------")
    # --- print results ---
    print("\n========== Validation KITTI (Per-Image Uncertainty) ==========")
    print(f"EPE:     {epe:.4f}")
    print(f"Fl:      {fl:.2f}")
    print(f"NOC:     {noc_epe:.4f}")
    print(f"OCC:     {occ_epe:.4f}")
    print("----------------------------------------")
    print(f"AUSE(mean): {ause_mean:.6f}")
    print(f"Corr(mean): {corr_mean:.4f}")
    print(f"Avg Time:   {avg_time:.4f} s/frame")
    print("===========================================================\n")

    # --- ensure save path ---
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "kitti_model_curve_mean.npy"), model_curve_mean)
    np.save(os.path.join(save_dir, "kitti_oracle_curve_mean.npy"), oracle_curve_mean)
    np.save(os.path.join(save_dir, "kitti_plotx.npy"), np.array(plotx))
    
    # --- plot sparsification curves (EPE) ---
    plt.figure()
    plt.plot(plotx, model_curve_mean, label='Model (avg)')
    plt.plot(plotx, oracle_curve_mean, label='Oracle (avg)')
    plt.xlabel('Removed Fraction')
    plt.ylabel('Normalized Mean EPE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.title(f'KITTI Per-image Sparsification Curves (AUSE={ause_mean:.4f})')
    plt.savefig(os.path.join(save_dir, 'kitti_per_image_sparsification_curve.png'), dpi=150)
    plt.close()

    return {
        'epe': epe,
        'fl': fl,
        'noc': noc_epe,
        'occ': occ_epe,
        'ause_mean': ause_mean,
        'corr_mean': corr_mean,
        'model_curve_mean': model_curve_mean,
        'oracle_curve_mean': oracle_curve_mean,
        'plotx': plotx,
        'avg_time': avg_time
    }

@torch.no_grad()
def validate_kitti_2012(model, save_flow=False, network='U2Flow', test_shape = [256, 832]):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}

    val_dataset = flow_datasets.KITTI2012(test_shape = test_shape)
    
    epe_list = []
    out_list = []
    noc_epe_error_list, epe_error_occ_list= [], []
    for val_id in tqdm(range(len(val_dataset)), desc='Validate'): # len(val_dataset)
                    
        images, flow_gt, valid_gt, flownoc, nocmask = val_dataset[val_id]

        images = images[None].cuda()

        flows = model(images, only_fw=True)
        flow_pred = flows['flows_f12'][-1]

        _, h, w = flow_gt.size()
        flow_pred_up = resize_flow(flow_pred, (h, w))[0].detach().cpu()

        epe = torch.sum((flow_pred_up - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        # out_list.append(out[val].mean().item())

        noc_epe_error_= torch.sum((flow_pred_up - flownoc)**2, dim=0).sqrt().view(-1)
        val_noc = nocmask.view(-1) >= 0.5
        noc_epe_error_list.append(noc_epe_error_[val_noc].mean().item())

        pep_error_occ= torch.sum((flow_pred_up - flow_gt)**2, dim=0).sqrt().view(-1)
        occ_erea_mask = valid_gt   - nocmask 
        if (occ_erea_mask == 0).all():
            epe_error_occ_list.append(0)
        else :
            occ_erea_mask = occ_erea_mask.view(-1) >= 0.5
            epe_error_occ_list.append(pep_error_occ[occ_erea_mask].mean().item())


        if save_flow:
            network = network + '_2015'
            save_imgs_with_unc(val_id, network, images[0, :, ... ], flows['flows_fw'], flows['flows_bw'], epe[val].mean().item())

    epe_list = np.array(epe_list)
    noc_epe_error_list = np.array(noc_epe_error_list)
    epe_error_occ_list = np.array(epe_error_occ_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    noc_epe = np.mean(noc_epe_error_list)
    occ_epe = np.mean(epe_error_occ_list)
    fl = 100 * np.mean(out_list)

    print("Validation 2012 KITTI: %f, %f, noc %f, occ %f" % (epe, fl, noc_epe, occ_epe))
    return {'2012_epe': epe, 'fl': fl, 'noc':noc_epe, 'occ':occ_epe }

@torch.no_grad()
def create_kitti_submission(model, output_path):
    """ Peform validation using the Sintel (train) split """

    model.eval()
    results = {}
    output_path = os.path.join(output_path, 'kitti2015')
    val_dataset = flow_datasets.KITTI2015(split='testing')

    # output_path = os.path.join(output_path, 'kitti2012')
    # val_dataset = flow_datasets.KITTI2012(split='testing')
    

    output_flow = os.path.join(output_path, 'flow')
    if not os.path.exists(output_flow):
        os.makedirs(output_flow)
    vis_kitti = os.path.join(output_path, 'vis_kitti')
    if not os.path.exists(vis_kitti):
        os.makedirs(vis_kitti)
    for val_id in tqdm(range(len(val_dataset)), desc='Validate'):
                    
        images, frame_id, h_info, w_info = val_dataset[val_id]

        images = images[None].cuda()
        flows_dict = model(images, only_fw=True)
        flows = flows_dict['flows_f12']

        flow_pred = flows[-1]
        flow_pred_up = resize_flow(flow_pred, (h_info, w_info))[0].permute(1, 2, 0).cpu().numpy()
        
        output_filename = os.path.join(output_flow, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow_pred_up)

        flow_img = flow_viz.flow_to_image(flow_pred_up)
        image = Image.fromarray(flow_img)


        vis_kitti_ = vis_kitti + f'/{frame_id}'  # f'/{frame_id}.png'
        image.save(vis_kitti_)


        # unc_map = flows_dict['uncertainty_f12'][-1]
        # unc_map = torch.exp(unc_map).clamp(min=1e-3, max=1000)
        # unc_up = torch.nn.functional.interpolate(unc_map, size=(h_info, w_info), mode='area')
        # unc_up = unc_up.squeeze().cpu().numpy()
        # variance_log = np.log1p(unc_up)
        # variance_log_min = np.min(variance_log)
        # variance_log_max = np.max(variance_log)
        # if variance_log_max - variance_log_min < 1e-5:
        #     variance_log_norm = np.zeros_like(variance_log, dtype=np.uint8)
        # else:   
        #     variance_log_norm = ((variance_log - variance_log_min) / (variance_log_max - variance_log_min) * 255).astype(np.uint8) 
        # variance_log_norm_heatmap = cv2.applyColorMap(variance_log_norm, cv2.COLORMAP_JET)
        # variance_log_norm_heatmap = variance_log_norm_heatmap[:,:, ::-1]  # BGR to RGB


        # vis_unc = os.path.join(output_path, 'vis_unc')
        # if not os.path.exists(vis_unc):
        #     os.makedirs(vis_unc)
        # filename = f"{vis_unc}/{frame_id}"
        # cv2.imwrite(filename, variance_log_norm_heatmap[:, :, [2,1,0]])

    return 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

autocast = torch.cuda.amp.autocast
print(os.getcwd())
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')

    args = parser.parse_args()
    if args.config:
        import importlib
        cfg_module = importlib.import_module(f"configs.{args.config}")
        get_cfg = cfg_module.get_cfg

    cfg = get_cfg()
    cfg.mixed_precision = True
    cfg[cfg.network].mixed_precision = True
    cfg.raft_2f_u.iters = 12
    model = torch.nn.DataParallel(build_network(cfg))
    
    path_ck = 'checkpoints/kitti.pth'
    # path_ck = 'checkpoints/sintel.pth'


    model.load_state_dict(torch.load(path_ck),strict=True)       #   cfg.model
    model.cuda()
    model.eval()

    print(cfg.network)
    print("Parameter Count: %d" % count_parameters(model))
    with torch.no_grad():
        
        validate_kitti_2015(model = model.module, save_flow = False, network =  cfg.network, test_shape = [256, 832])
        
        # validate_kitti_unc_per_image(model = model.module, save_flow = False, network =  cfg.network, test_shape = [256, 832])

        # validate_kitti_2012(model = model.module, save_flow = False, network =  cfg.network, test_shape = [256, 832])




        # validate_sintel_batch(model.module)

        # validate_sintel_unc_per_image(model = model.module)



        # create_kitti_submission(model.module, output_path="logs/evaluate/kitti_submission/2frames/")

        # create_sintel_submission(model = model.module)
        