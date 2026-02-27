# R3-PCQA

## Project Structure

```
.R3-PCQA/
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
└── data_csv/
    ├── SJTU_MOS.csv
    ├── WPC_MOS.csv
    └── WPC2.0_MOS.csv
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
- open3d
- opencv-python
