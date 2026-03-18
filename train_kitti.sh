#!/bin/bash

gpu_id='6,7'

n_per_n=$(echo $gpu_id | awk -F',' '{print NF}')
hostport=$((15500 + RANDOM % 500))

export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES=$gpu_id OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=$n_per_n --rdzv_endpoint=localhost:$hostport train.py \
    --config=KITTI_Raw
