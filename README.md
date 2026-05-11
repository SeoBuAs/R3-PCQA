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
  <img src="assets/pipeline.png" alt="R3-PCQA architecture overview" width="90%"/>
</p>

R3-PCQA emulates three HVS priors — **viewpoint-dependent processing**, **selective attention**, and **multiview integration** — through a geometric-aware preprocessing pipeline and three architectural components, trained with a two-stage curriculum.

### Preprocessing · Geometric-aware Reprojection
The input point cloud is rendered onto **V = 20** uniformly distributed viewpoints centered on the faces of a regular icosahedron. At each view, **N = 9** candidate pixels are obtained by K-means on valid-depth pixels, back-projected to camera rays, and matched to the nearest visible 3D point. A KNN expansion (M = 8192 points) around each seed yields the pixel-to-subcloud correspondences `(p_{v,n}, X_{v,n})` that link 2D coordinates to view-visible 3D regions — establishing precise 2D–3D spatial alignment.

### 1. Local View Encoder
At each viewpoint, a 2D encoder `E_rgb` captures fine-grained **texture** distortions from `I_v`, while a 3D encoder `E_pc` captures coarse **geometric** distortions from the QSS-selected subcloud `X_{v,n*}`. The two features are fused into a coupled geometry–texture representation `F_v`, and concatenated with an auxiliary local-quality prediction `ŷ_v^local` to form the **local view token** `z_v`. Parameters are shared across views to learn generalizable, view-agnostic representations.

### 2. Quality-Salient Subcloud Selector (QSS)
To our knowledge the **first reinforcement-learning module for PCQA**. QSS frames per-view subcloud selection as a **contextual bandit** problem: a self-attention policy network observes 11×11 local contexts `H_v` cropped from an intermediate feature map of `E_rgb`, and outputs the temperature-softmax policy `π_θ(a_{v,n} | H_v)` over the N candidate subclouds. The policy is trained with **REINFORCE**, mimicking HVS selective attention to perceptually decisive regions while reducing compute by evaluating only one subcloud per view.

### 3. Global View Attention
A **local-to-global aggregator** over the V local view tokens. The mean-pooled token `g` queries a multi-head attention over `Z = [z_1, …, z_V]^T`, producing per-view attention weights `{α_v}` and a refined global context `g̃`. The global regressor maps `g̃ ⊕ g` to the final quality prediction `ŷ`, enabling adaptive multiview integration.

### Training · Two-stage Curriculum Learning
1. **Warm-up phase** — QSS is deactivated and one of `{X_{v,n}}_{n=1}^N` is sampled randomly so the local view encoder learns generalized representations: `L_warm-up = L_global + L_local`.
2. **Joint phase** — QSS is activated and trained jointly with the encoders: `L_joint = L_global + L_local + L_policy`. The REINFORCE reward `r_b = exp(−s · |ŷ_b − y_b| / σ)` (with `s = 100`, `σ = 15`) is **redistributed across views by the attention weights** `α_{b,v}`, providing per-view credit assignment that steers each viewpoint's policy toward subclouds most informative for the global score.

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
