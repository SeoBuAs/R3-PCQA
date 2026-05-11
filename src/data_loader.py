import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def xyzrgb_normalize(point):
    point_normalized = point.copy()
    point_normalized[:, 0:3] = point_normalized[:, 0:3] - np.mean(point_normalized[:, 0:3], axis=0)
    point_normalized[:, 3:6] = point_normalized[:, 3:6] - np.mean(point_normalized[:, 3:6], axis=0)
    return point_normalized

def remove_ply_extension(filename):
    base = os.path.basename(filename)
    if base.endswith("'"):
        base = base[:-1]
    if base.lower().endswith('.ply'):
        base = base[:-4]
    return base

class UnifiedDataset(Dataset):
    def __init__(self, df, transform_rgb=None, num_views=20, verbose=False, 
                 kmeans_patches_dir=None, patch_size=8192, num_patches_per_view=9, 
                 pattern='train', stage='stage1'):
        self.df = df.reset_index(drop=True)
        self.transform_rgb = transform_rgb
        self.num_views = num_views
        self.verbose = verbose
        self.kmeans_patches_dir = kmeans_patches_dir
        self.patch_size = patch_size
        self.num_patches_per_view = num_patches_per_view
        self.pattern = pattern
        self.stage = stage
        
        self.rgb_suffixes = [f'_view_{i}_rgb.png' for i in range(1, num_views+1)]
        self.view_names = [f"view_{i}" for i in range(1, num_views + 1)]
        
    def __len__(self):
        return len(self.df)
    
    def clean_filename(self, filename):
        clean_name = filename.rstrip("'").rstrip('"')
        if clean_name.endswith('.ply'):
            clean_name = clean_name[:-4]
        return clean_name
    
    def load_kmeans_patch(self, base_name, view_name, patch_idx):
        patch_filename = f"{base_name}_{view_name}_3d_patch_{patch_idx:02d}.npy"
        patch_path = os.path.join(self.kmeans_patches_dir, patch_filename)
        
        if not os.path.exists(patch_path):
            raise FileNotFoundError(f"Patch file not found: {patch_path}")
        
        patch_data = np.load(patch_path)
        
        if patch_data.shape[0] == 0 or patch_data.shape[1] != 6:
            raise ValueError(f"Invalid patch shape: {patch_path}, shape: {patch_data.shape}")
        
        if patch_data.shape[0] < self.patch_size:
            padding = np.zeros((self.patch_size - patch_data.shape[0], 6), dtype=np.float32)
            patch_data = np.concatenate([patch_data, padding], axis=0)
        elif patch_data.shape[0] > self.patch_size:
            patch_data = patch_data[:self.patch_size]
        
        patch_normalized = xyzrgb_normalize(patch_data.astype(np.float32))
        
        return patch_normalized
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base_path = row['Ply_images']
        ply_name = row['Ply_name']
        
        clean_base = self.clean_filename(ply_name)
        
        if self.verbose:
            print(f"Processing: {clean_base}")
        
        if self.pattern == 'test':
            seed = idx
        else:
            seed = None
        
        rng = np.random.default_rng(seed)
        
        rgb_images = []
        for rgb_suffix in self.rgb_suffixes:
            rgb_path = base_path + rgb_suffix
            
            if not os.path.exists(rgb_path):
                raise FileNotFoundError(f"RGB image file not found: {rgb_path}")
            
            rgb_image = Image.open(rgb_path).convert("RGB")
            
            if self.transform_rgb:
                rgb_tensor = self.transform_rgb(rgb_image)
            else:
                rgb_tensor = transforms.ToTensor()(rgb_image)
            
            rgb_images.append(rgb_tensor)
        
        rgb_tensor = torch.stack(rgb_images)
        
        point_patches = []
        
        if self.kmeans_patches_dir:
            for view_idx, view_name in enumerate(self.view_names):
                if self.stage == 'stage1':
                    patch_idx = rng.integers(0, self.num_patches_per_view)
                    patch_data = self.load_kmeans_patch(clean_base, view_name, patch_idx)
                    
                    if self.pattern == 'train':
                        rng.shuffle(patch_data)
                    point_patches.append(torch.tensor(patch_data, dtype=torch.float32))
                        
                else:
                    view_patches = []
                    for patch_idx in range(self.num_patches_per_view):
                        patch_data = self.load_kmeans_patch(clean_base, view_name, patch_idx)
                        
                        if self.pattern == 'train':
                            rng.shuffle(patch_data)
                        view_patches.append(torch.tensor(patch_data, dtype=torch.float32))
                    
                    view_patches_tensor = torch.stack(view_patches)
                    point_patches.append(view_patches_tensor)
                        
        else:
            if self.stage == 'stage1':
                point_patches = [torch.zeros(self.patch_size, 6).float() for _ in range(self.num_views)]
            else:
                for _ in range(self.num_views):
                    view_patches = [torch.zeros(self.patch_size, 6).float() for _ in range(self.num_patches_per_view)]
                    view_patches_tensor = torch.stack(view_patches)
                    point_patches.append(view_patches_tensor)
            
            if self.verbose:
                print("K-means patches directory not set")
        
        if self.stage == 'stage1':
            point_tensor = torch.stack(point_patches).permute(0, 2, 1)
        else:
            point_tensor = torch.stack(point_patches).permute(0, 1, 3, 2)
        
        target = torch.tensor(row['MOS'] / 100.0, dtype=torch.float)
        
        return {
            'rgb': rgb_tensor,
            'point': point_tensor,
            'target': target,
            'path': base_path,
            'df': row
        }

def custom_collate_fn(batch):
    rgb_batch = torch.stack([item['rgb'] for item in batch])
    point_batch = torch.stack([item['point'] for item in batch])
    target_batch = torch.stack([item['target'] for item in batch])
    
    df_batch = [item['df'] for item in batch]
    
    path_batch = [item['path'] for item in batch]
    
    return {
        'rgb': rgb_batch,
        'point': point_batch,
        'target': target_batch,
        'df': df_batch,
        'path': path_batch
    }

def load_data(data_path, kmeans_patches_dir, num_views=20, patch_size=8192, 
              num_patches_per_view=9, batch_size=4, random_seed=42, stage='stage1',
              custom_train_df=None, custom_val_df=None, projection_dir='/workspace/dataset/WPC_Projection/'):
    
    df = pd.read_csv(data_path)
    df['Ply_name'] = df['Ply_name'].apply(remove_ply_extension) + '.ply'
    df['Ply_images'] = projection_dir + df['Ply_name'].apply(remove_ply_extension)
    
    if custom_train_df is not None and custom_val_df is not None:
        df_train = custom_train_df.copy()
        df_val = custom_val_df.copy()
        print(f"Using custom train/val split for 5Fold cross-validation")
    else:
        selected_objects = ['cauliflower', 'banana', 'mushroom', 'pineapple']
        df_train = df[~df['Content'].isin(selected_objects)].reset_index(drop=True)
        df_val = df[df['Content'].isin(selected_objects)].reset_index(drop=True)
        print(f"Using default train/val split")
    
    df_train['Ply_name'] = df_train['Ply_name'].apply(remove_ply_extension) + '.ply'
    df_train['Ply_images'] = projection_dir + df_train['Ply_name'].apply(remove_ply_extension)
    
    df_val['Ply_name'] = df_val['Ply_name'].apply(remove_ply_extension) + '.ply'
    df_val['Ply_images'] = projection_dir + df_val['Ply_name'].apply(remove_ply_extension)
    
    transform_rgb = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = UnifiedDataset(
        df=df_train,
        transform_rgb=transform_rgb,
        num_views=num_views,
        verbose=False,
        kmeans_patches_dir=kmeans_patches_dir,
        patch_size=patch_size,
        num_patches_per_view=num_patches_per_view,
        pattern='train',
        stage=stage
    )
    
    test_dataset = UnifiedDataset(
        df=df_val,
        transform_rgb=transform_rgb,
        num_views=num_views,
        verbose=False,
        kmeans_patches_dir=kmeans_patches_dir,
        patch_size=patch_size,
        num_patches_per_view=num_patches_per_view,
        pattern='test',
        stage=stage
    )
    
    num_workers = 8
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    
    print(f"{stage.upper()} Dataset created successfully!")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Using {num_workers} CPU workers for data loading")
    
    for batch in train_loader:
        print(f"RGB shape: {batch['rgb'].shape}")
        print(f"Point shape: {batch['point'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"DF type: {type(batch['df'])}")
        print(f"DF length: {len(batch['df'])}")
        break
    
    return train_loader, test_loader

def load_data_stage1_efficient(data_path, kmeans_patches_dir, num_views=20, patch_size=8192, 
                              num_patches_per_view=9, batch_size=4, random_seed=42):
    return load_data(data_path, kmeans_patches_dir, num_views, patch_size, 
                    num_patches_per_view, batch_size, random_seed, 'stage1')

class OptimizedMultiViewKMeansDataset(UnifiedDataset):
    def __init__(self, *args, **kwargs):
        kwargs['stage'] = 'stage2'
        super().__init__(*args, **kwargs) 
