import numpy as np
import torch
import torch.nn.functional as F
import os
import glob

def normalize_pixel_coordinates(pixel_coords, original_size=1080, target_size=112):
    scale_factor = target_size / original_size
    normalized_coords = pixel_coords * scale_factor
    return normalized_coords.astype(np.int64)

def crop_feature_maps_at_coordinates(feature_maps, pixel_coords, crop_size=11):
    batch_size, channels, height, width = feature_maps.shape
    device = feature_maps.device
    
    if pixel_coords.dim() == 3:
        num_views = 1
        pixel_coords = pixel_coords.unsqueeze(1)
    else:
        num_views = pixel_coords.shape[1]
    
    half_crop = crop_size // 2
    
    cropped_patches = []
    
    for b in range(batch_size):
        batch_patches = []
        for v in range(num_views):
            view_patches = []
            for p in range(9):
                center_x, center_y = pixel_coords[b, v, p, 0], pixel_coords[b, v, p, 1]
                
                start_x = max(0, center_x - half_crop)
                end_x = min(width, center_x + half_crop + 1)
                start_y = max(0, center_y - half_crop)
                end_y = min(height, center_y + half_crop + 1)
                
                patch = feature_maps[b, :, start_y:end_y, start_x:end_x]
                
                if patch.shape[1] < crop_size or patch.shape[2] < crop_size:
                    pad_h = max(0, crop_size - patch.shape[1])
                    pad_w = max(0, crop_size - patch.shape[2])
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                patch = patch[:, :crop_size, :crop_size]
                view_patches.append(patch)
            
            view_patches = torch.stack(view_patches, dim=0)
            batch_patches.append(view_patches)
        
        if num_views == 1:
            batch_patches = batch_patches[0]
        else:
            batch_patches = torch.stack(batch_patches, dim=0)
        
        cropped_patches.append(batch_patches)
    
    return torch.stack(cropped_patches, dim=0)

def load_pixel_coordinates(base_name, view_name, pixel_coords_dir):
    filename = f"{base_name}_{view_name}_pixel_coords.npy"
    file_path = os.path.join(pixel_coords_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Pixel coordinate file not found: {filename}")
    
    pixel_coords = np.load(file_path)
    return pixel_coords

def prepare_pixel_coordinates_batch(df_batch, pixel_coords_dir, num_views=20):
    batch_size = len(df_batch)
    batch_pixel_coords = []
    
    for i in range(batch_size):
        if isinstance(df_batch, list):
            row = df_batch[i]
        else:
            row = df_batch.iloc[i]
        
        base_name = row['Ply_name'].replace('.ply', '')
        view_coords = []
        
        for view_idx in range(1, num_views + 1):
            view_name = f"view_{view_idx}"
            pixel_coords = load_pixel_coordinates(base_name, view_name, pixel_coords_dir)
            normalized_coords = normalize_pixel_coordinates(pixel_coords)
            view_coords.append(normalized_coords)
        
        view_coords = np.stack(view_coords, axis=0)
        batch_pixel_coords.append(view_coords)
    
    return np.stack(batch_pixel_coords, axis=0)
