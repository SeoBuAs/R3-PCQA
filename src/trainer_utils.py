import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os

def calculate_metrics(predictions, targets):
    targets_np = np.array(targets).flatten()
    preds_np = np.array(predictions).flatten()
    
    val_plcc, val_srcc, val_rmse = 0.0, 0.0, 0.0
    
    if len(targets_np) > 1 and len(preds_np) > 1:
        plcc_value, _ = pearsonr(targets_np, preds_np)
        val_plcc = plcc_value if not np.isnan(plcc_value) else 0.0
        
        srcc_value, _ = spearmanr(targets_np, preds_np)
        val_srcc = srcc_value if not np.isnan(srcc_value) else 0.0
        
        targets_original = targets_np * 100.0
        preds_original = preds_np * 100.0
        val_rmse = np.sqrt(np.mean((preds_original - targets_original) ** 2))
    
    return val_plcc, val_srcc, val_rmse

def save_model(model, optimizer, epoch, metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, save_path)


def log_epoch_results(epoch, train_loss, val_loss, val_plcc, val_srcc, val_rmse, 
                     stage_name, use_wandb=True, train_view_loss=None, train_final_loss=None, 
                     train_mae=None, val_view_loss=None, val_final_loss=None, 
                     val_mae=None, learning_rate=None, train_reward=None, num_epochs=30):
    print("-" * 80)
    print(f"Epoch {epoch+1}/{num_epochs} Summary:")
    
    if train_view_loss is not None and train_final_loss is not None and train_mae is not None:
        print(f"\tTrain - View: {train_view_loss:.4f}, Final: {train_final_loss:.4f}, MAE: {train_mae:.4f}")
    else:
        print(f"\tTrain - Loss: {train_loss:.4f}")
    
    if val_view_loss is not None and val_final_loss is not None and val_mae is not None:
        print(f"\tVal   - View: {val_view_loss:.4f}, Final: {val_final_loss:.4f}, MAE: {val_mae:.4f}")
    else:
        print(f"\tVal   - Loss: {val_loss:.4f}")
    
    print(f"\tVal   - PLCC: {val_plcc:.4f}, SRCC: {val_srcc:.4f}, RMSE: {val_rmse:.4f}")
    
    if learning_rate is not None:
        print(f"\tLearning Rate: {learning_rate:.2e}")
    if train_reward is not None:
        print(f"\tTrain - Reward: {train_reward:.4f}")
    
    print("-" * 80)
def validate_model(model, test_loader, criterion, device, stage_name, pixel_coords_dir=None, n_views=20, num_patches=9):
    model.eval()
    val_loss = 0
    val_view_loss = 0
    val_final_loss = 0
    val_mae = 0
    val_predictions = []
    val_targets = []
    
    val_patch_selection_counts = None
    if stage_name == 'stage2':
        val_patch_selection_counts = np.zeros(num_patches, dtype=np.int64)
    
    with torch.no_grad():
        val_pbar = tqdm(test_loader, desc=f"{stage_name} [Val]")
        for batch in val_pbar:
            rgb_data = batch['rgb'].to(device)
            point_data = batch['point'].to(device)
            target = batch['target'].to(device)
            
            data_dict = {'rgb': rgb_data, 'point': point_data}
            
            if stage_name == 'stage2':
                if batch.get('df') is not None and len(batch['df']) > 0:
                    from src.pixel_coordinate_utils import prepare_pixel_coordinates_batch
                    pixel_coords = prepare_pixel_coordinates_batch(
                        df_batch=batch['df'],
                        pixel_coords_dir=pixel_coords_dir,
                        num_views=n_views
                    )
                    pixel_coords = torch.tensor(pixel_coords, dtype=torch.long).to(device)
                    outputs = model(data_dict, pixel_coords=pixel_coords, is_random=False)
            else:
                outputs = model(data_dict, is_random=True)
            
            target = target.unsqueeze(1) if target.dim() == 1 else target
            
            final_loss = criterion(outputs['final_prediction'], target)
            val_final_loss += final_loss.item()
            
            view_loss = 0
            if 'view_predictions' in outputs:
                target = target.float().view(-1, 1)
                targets_expanded = target.unsqueeze(1).repeat(1, model.n_views, 1)
                view_loss = criterion(outputs['view_predictions'], targets_expanded)
                val_view_loss += view_loss.item()
            
            mae = torch.mean(torch.abs(outputs['final_prediction'].squeeze() - target))
            val_mae += mae.item()
            
            if stage_name == 'stage2' and 'selected_patch_idx' in outputs:
                selected_patches = outputs['selected_patch_idx'].cpu().numpy()
                for batch_patches in selected_patches:
                    for patch_idx in batch_patches:
                        val_patch_selection_counts[int(patch_idx)] += 1
            
            total_loss = final_loss + view_loss
            val_loss += total_loss.item()
            
            val_predictions.extend(outputs['final_prediction'].squeeze().cpu().numpy())
            val_targets.extend(target.cpu().numpy())
            
            val_pbar.set_postfix({
                'View': f"{val_view_loss/(val_pbar.n+1):.3f}",
                'Final': f"{val_final_loss/(val_pbar.n+1):.3f}",
                'MAE': f"{val_mae/(val_pbar.n+1):.3f}"
            })
    
    val_plcc, val_srcc, val_rmse = calculate_metrics(val_predictions, val_targets)
    avg_val_loss = val_loss / len(test_loader)
    avg_val_view_loss = val_view_loss / len(test_loader)
    avg_val_final_loss = val_final_loss / len(test_loader)
    avg_val_mae = val_mae / len(test_loader)
    
    return avg_val_loss, val_plcc, val_srcc, val_rmse, avg_val_view_loss, avg_val_final_loss, avg_val_mae, val_patch_selection_counts