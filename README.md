
# U²Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation (CVPR 2026)

## ⚙️ Requirements

### 📦 Environment

This code has been tested under **Python 3.7.16**, **PyTorch 1.13.1**, **Torchvision 0.14.1**, and **CUDA 11.3** on Ubuntu 18.04. The environment can be set up using the following commands:

```shell
# Install python packages
pip install -r requirements.txt

# Install the correct PyTorch version
# Please install the versions compatible with your CUDA setup (e.g., cu116)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 📁 Datasets

- **KITTI**: The following sets are required.
    - **KITTI-raw**: Follow the instructions on the official KITTI raw data website to download.
    - **KITTI-2015**: From the official KITTI 2015 benchmark website, download **"stereo 2015/flow 2015/scene flow 2015 data set (2 GB)"** and **"multi-view extension (20 frames per scene) (14 GB)"**.
    - **KITTI-2012**: From the official KITTI 2012 benchmark website, download **"stereo/optical flow data set (2 GB)"** and **"multi-view extension (20 frames per scene, all cameras) (17 GB)"**.
    
- **Sintel**: Download the required data from the official Sintel website. For Sintel Raw, please refer to the ARFlow project repository.


Modify the file paths in `core/flow_datasets.py` and `core/flow_datasets_sintel.py` according to the unzip path.

### 🧩 Segment Anything Model

For semantic segmentation and homography smoothness loss, we follow the official [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) repository to generate SAM masks for all samples. Please refer to the [UnSAMFlow](https://github.com/facebookresearch/UnSAMFlow) project repository for generating full segmentation and finding key objects from SAM masks. This process is also implemented in `UnSAMFlow/sam_inference.py`.

## 🚀 How to Train

### Training on KITTI

The training process for KITTI datasets is divided into two stages:

1. **Stage 1: Pre-training on KITTI Raw**
   - Update the configuration argument in `train_kitti.sh` to `--config=KITTI_Raw`.
   - Execute the training script:
     ```shell
     bash train_kitti.sh
     ```

2. **Stage 2: Fine-tuning on KITTI Multi-View**
   - Provide the path of your Stage 1 pre-trained model checkpoint in the `restore_ckpt` parameter within `configs/KITTI_MV.py`.
   - Update the configuration argument in `train_kitti.sh` to `--config=KITTI_MV`.
   - Execute the training script:
     ```shell
     bash train_kitti.sh
     ```

### Training on Sintel

The training pipeline for Sintel follows a similar two-stage approach:

1. **Stage 1: Pre-training on Sintel Raw**
   - Update the configuration argument in `train_sintel.sh` to `--config=S_R`.
   - Run the training script:
     ```shell
     bash train_sintel.sh
     ```

2. **Stage 2: Fine-tuning on Sintel Final/Clean**
   - Provide the Stage 1 model checkpoint path in `configs/S_M.py`.
   - Change the configuration argument in `train_sintel.sh` to `--config=S_M`.
   - Run the training script:
     ```shell
     bash train_sintel.sh
     ```

---

## 🧪 How to Test

Pre-trained checkpoints are available at:  
https://drive.google.com/drive/folders/1zPanXhaVFGwd-et8d9u_EuE58KstVIq9?usp=sharing  

Please place them in the `checkpoints/` directory (e.g., `kitti.pth` and `sintel.pth`).

To evaluate the model, run `evaluate.py` with the desired configuration. Make sure to update the checkpoint path in `evaluate.py` accordingly.

**For KITTI evaluation:**
```shell
python -u evaluate.py --config KITTI_MV
```

**For Sintel evaluation:**
```shell
python -u evaluate.py --config S_M
```

## 💡 Code Credits

This project builds upon the following excellent works:

- **ARFlow**  
  Liang Liu *et al.*, *"Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation"*  
  https://github.com/lliuz/ARFlow

- **RAFT**  
  Zachary Teed and Jia Deng, *"RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"*  
  https://github.com/princeton-vl/RAFT

- **UnSAMFlow**  
  Shuai Yuan *et al.*, *"UnSAMFlow: Unsupervised Optical Flow Guided by Segment Anything Model"*  
  https://github.com/facebookresearch/UnSAMFlow

- **SMURF**  
  Austin Stone *et al.*, *"SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping"*  
  https://github.com/google-research/google-research/tree/master/smurf