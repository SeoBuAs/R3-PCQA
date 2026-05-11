import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import R3_PCQA
from src.data_loader import UnifiedDataset, custom_collate_fn, remove_ply_extension
from src.pixel_coordinate_utils import prepare_pixel_coordinates_batch

def predict(args):
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing model...")
    model = R3_PCQA(n_views=args.n_views, feature_dim=args.feature_dim, num_patches=args.num_patches_per_view)
    model = model.to(device)

    print(f"Loading model weights from {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights file not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model weights loaded successfully.")

    print(f"Loading full dataset from {args.data_path}")
    full_df = pd.read_csv(args.data_path)
    full_df['Ply_name'] = full_df['Ply_name'].apply(remove_ply_extension) + '.ply'
    full_df['Ply_images'] = full_df['Ply_name'].apply(lambda x: os.path.join(args.projection_dir, remove_ply_extension(x)))

    transform_rgb = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Creating dataset object...")
    full_dataset = UnifiedDataset(
        df=full_df,
        transform_rgb=transform_rgb,
        num_views=args.n_views,
        verbose=False,
        kmeans_patches_dir=args.kmeans_patches_dir,
        patch_size=args.patch_size,
        num_patches_per_view=args.num_patches_per_view,
        pattern='test',
        stage='stage2'
    )
    
    print("Creating data loader...")
    num_workers = 4
    full_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=custom_collate_fn
    )
    print(f"Full dataset loaded. Number of batches: {len(full_loader)}")

    all_predictions = []
    all_targets = []
    all_file_names = []

    with torch.no_grad():
        pbar = tqdm(full_loader, desc="Predicting")
        for batch in pbar:
            rgb_data = batch['rgb'].to(device)
            point_data = batch['point'].to(device)
            target = batch['target'].to(device)
            
            data_dict = {'rgb': rgb_data, 'point': point_data}

            if 'df' in batch and len(batch['df']) > 0:
                pixel_coords = prepare_pixel_coordinates_batch(
                    df_batch=batch['df'],
                    pixel_coords_dir=args.pixel_coords_dir,
                    num_views=args.n_views
                )
                pixel_coords = torch.tensor(pixel_coords, dtype=torch.long).to(device)
                
                outputs = model(data_dict, pixel_coords=pixel_coords, is_random=False)
                predictions = outputs['final_prediction'].squeeze().cpu().numpy()
                
                if predictions.ndim == 0:
                    predictions = [predictions.item()]
                else:
                    predictions = predictions.tolist()

                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
                all_file_names.extend([row['Ply_name'] for row in batch['df']])

            else:
                print("Warning: 'df' not found in batch or is empty. Skipping batch.")
                continue

    results_df = pd.DataFrame({
        'Ply_name': all_file_names,
        'MOS': np.array(all_targets) * 100,
        'Y_hat_normalized': all_predictions,
        'Y_hat': np.array(all_predictions) * 100
    })

    output_dir = os.path.dirname(args.model_path)
    output_filename = os.path.join(output_dir, "predictions_full_dataset.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on the full WPC dataset.')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights (.pth file).')
    
    parser.add_argument('--data_path', type=str, 
                        default='/workspace/dataset/WPC_MOS_no_100.csv', 
                        help='Path to the CSV data file.')
    parser.add_argument('--kmeans_patches_dir', type=str, 
                        default='/workspace/dataset/WPC/unified_3d_patches_raw/', 
                        help='Directory of K-means patches.')
    parser.add_argument('--pixel_coords_dir', type=str, 
                        default='/workspace/dataset/WPC/pixel_coordinates_raw/', 
                        help='Directory of pixel coordinates.')
    parser.add_argument('--projection_dir', type=str, 
                        default='/workspace/dataset/WPC_Projection/', 
                        help='Directory of projection images.')
    parser.add_argument('--num_patches_per_view', type=int, default=9, help='Number of patches per view.')
    parser.add_argument('--patch_size', type=int, default=8192, help='Patch size.')
    
    parser.add_argument('--n_views', type=int, default=20, help='Number of views.')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension.')
    
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference.')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number.')
    
    args = parser.parse_args()
    
    if not args.model_path:
        parser.error("--model_path is required.")
        
    predict(args)
