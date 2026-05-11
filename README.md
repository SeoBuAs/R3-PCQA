<div align="center">

<h1>R3-PCQA: Ray-Reprojection-Reinforcement for No-Reference 3D Point Cloud Quality Assessment</h1>

[![Conference](https://img.shields.io/badge/CVPR%202026-Poster-1b6ec2.svg?style=for-the-badge)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg?style=for-the-badge)](LICENSE)

<sub>📄 <b>Official PyTorch implementation</b> of our CVPR 2026 (Poster) paper.</sub>

<br/>

<b>Junhyuk Seo<sup>&ast;</sup> &nbsp;·&nbsp; Sanghyuk Seo<sup>&ast;</sup> &nbsp;·&nbsp; Dawoon Kim &nbsp;·&nbsp; Heeseok Oh<sup>&dagger;</sup></b>

<sub>Hansung University</sub>

<sub><sup>&ast;</sup>Equal contribution &nbsp;·&nbsp; <sup>&dagger;</sup>Corresponding author</sub>

<sub>`{withop9974, aissh, 2071290, ohhs}@hansung.ac.kr`</sub>

</div>

<!-- TODO: add arXiv / project page badges once links are available. -->
<!-- (Optional) Add a teaser image: <p align="center"><img src="assets/teaser.png" width="85%"/></p> -->

---

## 📢 News / Updates

- **[2026-02]** 🎉 R3-PCQA has been accepted to **CVPR 2026** as a **Poster**!
- **[2026-05]** Initial code released.

<!-- 정확한 월이 다르면 위 날짜만 수정해 주세요. 추후 항목(arXiv, pretrained weights 등)은 같은 형식으로 추가하면 됩니다. -->

## 📝 Abstract

Prevailing no-reference 3D point cloud quality assessment methods predominantly treat 2D projections and 3D point clouds as independent modalities and rely on simplistic feature fusion, thereby neglecting fundamental mechanisms underlying human 3D perception. To address this limitation, we introduce **R3-PCQA** (Ray-Reprojection-Reinforcement 3D Point Cloud Quality Assessor), a novel and principled framework that explicitly encodes perceptual priors into the assessment pipeline: A geometric-aware ray-based reprojection pipeline simulates viewpoint-dependent observation of 3D structure. A reinforcement-learning-based quality-salient subcloud selector adaptively attends to perceptually informative regions. The global view attention module aggregates local quality responses across viewpoints, forming a unified representation that facilitates reliable cross-view inference. Extensive experiments demonstrate that R3-PCQA achieves state-of-the-art performance on **SJTU-PCQA**, **WPC**, and **WPC2.0**.

## 🔧 Method Overview

<p align="center">
  <img src="assets/pipeline.png" alt="R3-PCQA pipeline overview" width="90%"/>
</p>

R3-PCQA consists of three key components:

1. **Ray-based Reprojection** — A geometric-aware pipeline that simulates viewpoint-dependent observation of 3D structure, bridging 2D projections and the underlying 3D geometry.
2. **RL-based Quality-Salient Subcloud Selector** — A reinforcement-learning agent that adaptively attends to perceptually informative regions of the point cloud.
3. **Global View Attention** — Aggregates local quality responses across viewpoints into a unified representation for reliable cross-view inference.

## Project Structure

```
R3-PCQA/
├── train.py
├── inference.py
├── src/
│   ├── model.py
│   ├── data_loader.py
│   ├── pixel_coordinate_utils.py
│   ├── trainer_utils.py
│   └── preprocessing/
│       ├── projection.py
│       └── reprojection.py
├── data_csv/
│   ├── SJTU_MOS.csv
│   ├── WPC_MOS.csv
│   └── WPC2.0_MOS.csv
├── assets/
│   └── pipeline.png
├── LICENSE
└── README.md
```

## Data Preprocessing

### 1. Projection

```bash
cd src/preprocessing
python projection.py \
    --input /path/to/ply/files \
    --output /path/to/projections \
    --width 1080 \
    --height 1080 \
    --depth_scale 30
```

### 2. Reprojection

```bash
python reprojection.py \
    --projection_dir /path/to/projections \
    --ply_dir /path/to/ply/files \
    --coord_output /path/to/pixel_coordinates \
    --patch_output /path/to/3d_patches
```

## Training

```bash
python train.py \
    --data_path /workspace/dataset/WPC_MOS_no_100.csv \
    --kmeans_patches_dir /path/to/3d_patches \
    --pixel_coords_dir /path/to/pixel_coordinates \
    --projection_dir /path/to/projections \
    --num_epochs 40 \
    --warmup_epochs 20 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --lambda_view 1.0 \
    --lambda_policy 1.0 \
    --cuda_device 0
```

## Inference

```bash
python inference.py \
    --model_path ./endtoend_results/experiment_xxx/fold_1/fold_1_best_plcc_model.pth \
    --data_path /workspace/dataset/WPC_MOS_no_100.csv \
    --kmeans_patches_dir /path/to/3d_patches \
    --pixel_coords_dir /path/to/pixel_coordinates \
    --projection_dir /path/to/projections \
    --batch_size 4 \
    --cuda_device 0
```

## Requirements

- PyTorch
- NumPy
- Pandas
- scikit-learn
- scipy
- tqdm
- wandb (optional)
- torchvision
- PIL
- open3d **(== 0.19.0)**
- opencv-python
