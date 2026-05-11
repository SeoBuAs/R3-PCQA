import os
import sys
import argparse
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

def create_fold_logger(save_dir, fold_idx, start_time):
    fold_log_dir = os.path.join(save_dir, f'fold_{fold_idx + 1}', 'logs')
    os.makedirs(fold_log_dir, exist_ok=True)
    
    log_filename = f'fold_{fold_idx + 1}_training_log_{start_time}.txt'
    log_path = os.path.join(fold_log_dir, log_filename)
    
    return log_path

def write_fold_log(log_path, message):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

def log_fold_message(message):
    print(f"[Fold] {message}")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import R3_PCQA
from src.data_loader import load_data
from sklearn.model_selection import KFold
import pandas as pd
from src.trainer_utils import (
    save_model, 
    log_epoch_results, validate_model
)


def setup_logging(save_dir, stage_name):
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'{stage_name.lower()}_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_fold_splits(data_path, n_folds=5, random_seed=42):
    print(f"Creating {n_folds}-fold splits based on objects...")
    
    df = pd.read_csv(data_path)
    
    unique_objects = df['Content'].unique()
    print(f"Found {len(unique_objects)} unique objects: {unique_objects}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_splits = []
    for fold_idx, (train_obj_idx, val_obj_idx) in enumerate(kf.split(unique_objects)):
        train_objects = unique_objects[train_obj_idx]
        val_objects = unique_objects[val_obj_idx]
        
        train_df = df[df['Content'].isin(train_objects)].reset_index(drop=True)
        val_df = df[df['Content'].isin(val_objects)].reset_index(drop=True)
        
        fold_splits.append({
            'fold_idx': fold_idx,
            'train_df': train_df,
            'val_df': val_df,
            'train_objects': train_objects,
            'val_objects': val_objects
        })
        
        print(f"Fold {fold_idx + 1}: Train={len(train_df)} samples, Val={len(val_df)} samples")
        print(f"  Train objects: {train_objects}")
        print(f"  Val objects: {val_objects}")
    
    return fold_splits


def train_endtoend_fold(fold_info, args, save_dir, logger, fold_idx):
    train_df = fold_info['train_df']
    val_df = fold_info['val_df']
    
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_log_path = create_fold_logger(save_dir, fold_idx, start_time)
    
    log_fold_message(f"\n=== Starting Fold {fold_idx + 1} End-to-End Joint Training ===")
    
    logger.info(f"Fold {fold_idx + 1} - Creating Stage1 dataloader for warm-up...")
    train_loader_stage1, test_loader_stage1 = load_data(
        data_path=args.data_path,
        kmeans_patches_dir=args.kmeans_patches_dir,
        patch_size=args.patch_size,
        num_patches_per_view=args.num_patches_per_view,
        batch_size=args.batch_size,
        stage='stage1',
        custom_train_df=train_df,
        custom_val_df=val_df,
        projection_dir=args.projection_dir
    )
    
    train_loader = train_loader_stage1
    test_loader = test_loader_stage1
    
    log_fold_message(f"Train batches: {len(train_loader)}")
    log_fold_message(f"Val batches: {len(test_loader)}")
    
    train_batch = next(iter(train_loader))
    val_batch = next(iter(test_loader))
    
    log_fold_message(f"Train batch shapes - RGB: {train_batch['rgb'].shape}, Point: {train_batch['point'].shape}, Target: {train_batch['target'].shape}")
    log_fold_message(f"Val batch shapes - RGB: {val_batch['rgb'].shape}, Point: {val_batch['point'].shape}, Target: {val_batch['target'].shape}")
    
    write_fold_log(fold_log_path, f"Fold {fold_idx + 1} End-to-End Joint Training Started...")
    write_fold_log(fold_log_path, f"Fold {fold_idx + 1} started at: {start_time}")
    write_fold_log(fold_log_path, f"Warm-up epochs: {args.warmup_epochs}")
    write_fold_log(fold_log_path, f"Total epochs: {args.num_epochs}")
    write_fold_log(fold_log_path, f"Train batches: {len(train_loader)}")
    write_fold_log(fold_log_path, f"Val batches: {len(test_loader)}")
    write_fold_log(fold_log_path, "")
    
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    log_fold_message(f"Using device: {device}")
    write_fold_log(fold_log_path, f"Using device: {device}")
    
    model = R3_PCQA(n_views=args.n_views, feature_dim=args.feature_dim, num_patches=args.num_patches_per_view)
    model = model.to(device)
    
    if args.use_pretrained:
        logger.info("Attempting to load Stage 1 pretrained weights...")
    
    logger.info("End-to-End training mode: All parameters trainable")
    log_fold_message("End-to-End training mode: All parameters trainable")
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate
    )
    
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log_fold_message(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    log_fold_message(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    main_criterion = nn.MSELoss()
    
    scaler = GradScaler()
    
    write_fold_log(fold_log_path, f"Learning rate: {args.learning_rate}")
    write_fold_log(fold_log_path, f"Batch size: {args.batch_size}")
    write_fold_log(fold_log_path, f"Lambda View: {args.lambda_view}")
    write_fold_log(fold_log_path, f"Lambda Policy: {args.lambda_policy}")
    write_fold_log(fold_log_path, f"Training Mode: End-to-End Joint")
    write_fold_log(fold_log_path, f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    write_fold_log(fold_log_path, "")
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_val_plcc_at_best_loss = None
    best_val_srcc_at_best_loss = None
    best_val_rmse_at_best_loss = None
    
    for epoch in range(args.num_epochs):
        model.train()
        
        epoch_final_loss = 0
        epoch_view_loss = 0
        epoch_policy_loss = 0
        epoch_total_loss = 0
        epoch_mae = 0
        
        if epoch < args.warmup_epochs:
            mode = 'warmup'
            logger.info(f"\n[Epoch {epoch + 1}/{args.num_epochs}] Mode: WARM-UP (Full Point Cloud)")
            log_fold_message(f"[Epoch {epoch + 1}] Mode: WARM-UP")
            write_fold_log(fold_log_path, f"[Epoch {epoch + 1}] Mode: WARM-UP")
        else:
            mode = 'joint'
            
            if epoch == args.warmup_epochs:
                logger.info("=" * 60)
                logger.info("Starting joint phase - Switching to Stage2 dataloader...")
                logger.info("=" * 60)
                log_fold_message("Joint phase - Switching to Stage2 dataloader")
                write_fold_log(fold_log_path, "\n[Joint Phase] Switching to Stage2 dataloader")
                
                del train_loader, test_loader
                import gc
                gc.collect()
                
                train_loader, test_loader = load_data(
                    data_path=args.data_path,
                    kmeans_patches_dir=args.kmeans_patches_dir,
                    patch_size=args.patch_size,
                    num_patches_per_view=args.num_patches_per_view,
                    batch_size=args.batch_size,
                    stage='stage2',
                    custom_train_df=train_df,
                    custom_val_df=val_df,
                    projection_dir=args.projection_dir
                )
                
                logger.info(f"Stage2 dataloader created - Train: {len(train_loader)} batches, Val: {len(test_loader)} batches")
                write_fold_log(fold_log_path, f"Stage2 dataloader created - Train: {len(train_loader)} batches\n")
            
            joint_epoch = epoch - args.warmup_epochs
            total_joint_epochs = args.num_epochs - args.warmup_epochs
            temperature = max(0.5, 1.0 - (joint_epoch / total_joint_epochs) * 0.5)
            
            logger.info(f"\n[Epoch {epoch + 1}/{args.num_epochs}] Mode: JOINT (Patch Selection) | Temperature: {temperature:.3f}")
            log_fold_message(f"[Epoch {epoch + 1}] Mode: JOINT | Temperature: {temperature:.3f}")
            write_fold_log(fold_log_path, f"[Epoch {epoch + 1}] Mode: JOINT | Temperature: {temperature:.3f}")
        
        if mode == 'joint':
            patch_selection_counts = np.zeros(args.num_patches_per_view, dtype=np.int64)
        
        train_pbar = tqdm(train_loader, desc=f"EndtoEnd Fold {fold_idx + 1} [{mode.upper()}] Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            rgb_data = batch['rgb'].to(device)
            point_data = batch['point'].to(device)
            target = batch['target'].to(device)
            target = target.unsqueeze(1) if target.dim() == 1 else target
            
            with autocast('cuda'):
                if mode == 'warmup':
                    data_dict = {
                        'rgb': rgb_data,
                        'point': point_data
                    }
                    outputs = model(data_dict, is_random=True)
                    
                    final_loss = main_criterion(outputs['final_prediction'], target)
                    
                    view_predictions = outputs['view_predictions']
                    target_expanded = target.unsqueeze(1).expand_as(view_predictions)
                    view_loss = main_criterion(view_predictions, target_expanded)
                    
                    mse_loss = final_loss + args.lambda_view * view_loss
                    total_loss = mse_loss
                    policy_loss_value = 0.0
                    view_loss_value = view_loss.item()
                    final_loss_value = final_loss.item()
                
                else:
                    if batch.get('df') is not None and len(batch['df']) > 0:
                        from src.pixel_coordinate_utils import prepare_pixel_coordinates_batch
                        pixel_coords = prepare_pixel_coordinates_batch(
                            df_batch=batch['df'],
                            pixel_coords_dir=args.pixel_coords_dir,
                            num_views=args.n_views
                        )
                    else:
                        raise ValueError("Pixel coordinate data is missing.")
                    
                    pixel_coords = torch.tensor(pixel_coords, dtype=torch.long).to(device)
                    
                    data_dict = {'rgb': rgb_data, 'point': point_data}
                    outputs = model(data_dict, pixel_coords=pixel_coords, is_random=False, temperature=temperature)
                    
                    final_loss = main_criterion(outputs['final_prediction'], target)
                    
                    view_predictions = outputs['view_predictions']
                    target_expanded = target.unsqueeze(1).expand_as(view_predictions)
                    view_loss = main_criterion(view_predictions, target_expanded)
                    
                    policy_loss = model.compute_contextual_bandit_loss(outputs, target)
                    
                    mse_loss = final_loss + args.lambda_view * view_loss
                    total_loss = mse_loss + args.lambda_policy * policy_loss
                    policy_loss_value = policy_loss.item()
                    view_loss_value = view_loss.item()
                    final_loss_value = final_loss.item()
                    
                    selected_patches = outputs['selected_patch_idx'].cpu().numpy()
                    for batch_patches in selected_patches:
                        for patch_idx in batch_patches:
                            patch_selection_counts[int(patch_idx)] += 1
            
            mae = torch.mean(torch.abs(outputs['final_prediction'] - target))
            
            scaler.scale(total_loss).backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_final_loss += final_loss_value
            epoch_view_loss += view_loss_value
            epoch_policy_loss += policy_loss_value
            epoch_total_loss += total_loss.item()
            epoch_mae += mae.item()
            
            if mode == 'warmup':
                train_pbar.set_postfix({
                    'Final': f"{final_loss_value:.3f}",
                    'View': f"{view_loss_value:.3f}",
                    'MAE': f"{mae.item():.3f}"
                })
            else:
                train_pbar.set_postfix({
                    'Final': f"{final_loss_value:.3f}",
                    'View': f"{view_loss_value:.3f}",
                    'Policy': f"{policy_loss_value:.3f}",
                    'MAE': f"{mae.item():.3f}"
                })
        
        avg_final_loss = epoch_final_loss / len(train_loader)
        avg_view_loss = epoch_view_loss / len(train_loader)
        avg_policy_loss = epoch_policy_loss / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_mae = epoch_mae / len(train_loader)
        
        train_losses.append(avg_total_loss)
        
        val_stage = 'stage1' if mode == 'warmup' else 'stage2'
        val_loss, val_plcc, val_srcc, val_rmse, val_view_loss, val_final_loss, val_mae, val_patch_counts = validate_model(
            model, test_loader, main_criterion, device, val_stage,
            pixel_coords_dir=args.pixel_coords_dir, n_views=args.n_views, num_patches=args.num_patches_per_view
        )
        val_losses.append(val_loss)
        
        val_metrics = {
            'val_loss': val_loss,
            'val_plcc': val_plcc,
            'val_srcc': val_srcc,
            'val_rmse': val_rmse,
            'val_view_loss': val_view_loss,
            'val_final_loss': val_final_loss,
            'val_mae': val_mae
        }
        
        if mode == 'joint':
            total_selections = patch_selection_counts.sum()
            patch_distribution = (patch_selection_counts / total_selections * 100) if total_selections > 0 else patch_selection_counts
            
            logger.info(f"\n[Epoch {epoch + 1}] Training Patch Selection Distribution:")
            write_fold_log(fold_log_path, f"\n[Epoch {epoch + 1}] Training Patch Selection Distribution:")
            
            for patch_idx in range(args.num_patches_per_view):
                logger.info(f"  Patch {patch_idx}: {patch_selection_counts[patch_idx]:5d} times ({patch_distribution[patch_idx]:5.2f}%)")
                write_fold_log(fold_log_path, f"  Patch {patch_idx}: {patch_selection_counts[patch_idx]:5d} times ({patch_distribution[patch_idx]:5.2f}%)")
            
            logger.info(f"  Total selections: {total_selections}")
            write_fold_log(fold_log_path, f"  Total selections: {total_selections}\n")
            
            if val_patch_counts is not None:
                val_total_selections = val_patch_counts.sum()
                val_patch_distribution = (val_patch_counts / val_total_selections * 100) if val_total_selections > 0 else val_patch_counts
                
                logger.info(f"\n[Epoch {epoch + 1}] Validation Patch Selection Distribution:")
                write_fold_log(fold_log_path, f"\n[Epoch {epoch + 1}] Validation Patch Selection Distribution:")
                
                for patch_idx in range(args.num_patches_per_view):
                    logger.info(f"  Patch {patch_idx}: {val_patch_counts[patch_idx]:5d} times ({val_patch_distribution[patch_idx]:5.2f}%)")
                    write_fold_log(fold_log_path, f"  Patch {patch_idx}: {val_patch_counts[patch_idx]:5d} times ({val_patch_distribution[patch_idx]:5.2f}%)")
        
        if mode == 'joint' and val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_val_plcc_at_best_loss = val_plcc
            best_val_srcc_at_best_loss = val_metrics['val_srcc']
            best_val_rmse_at_best_loss = val_metrics['val_rmse']
            best_loss_model_path = os.path.join(save_dir, f'fold_{fold_idx + 1}', f'fold_{fold_idx + 1}_best_loss_model.pth')
            os.makedirs(os.path.dirname(best_loss_model_path), exist_ok=True)
            save_model(model, optimizer, epoch, val_metrics, best_loss_model_path)
            logger.info(f"New best Loss model saved at epoch {epoch + 1} with Loss: {best_val_loss:.4f}, PLCC: {best_val_plcc_at_best_loss:.4f}, SRCC: {best_val_srcc_at_best_loss:.4f}, RMSE: {best_val_rmse_at_best_loss:.4f}")
            write_fold_log(fold_log_path, f"New best Loss model saved at epoch {epoch + 1} with Loss: {best_val_loss:.4f}, PLCC: {best_val_plcc_at_best_loss:.4f}, SRCC: {best_val_srcc_at_best_loss:.4f}, RMSE: {best_val_rmse_at_best_loss:.4f}")
        
        log_epoch_results(
            epoch, avg_total_loss, val_metrics['val_loss'],
            val_metrics['val_plcc'], val_metrics['val_srcc'], val_metrics['val_rmse'],
            f"ENDTOEND_FOLD{fold_idx + 1}", False,
            train_view_loss=avg_view_loss, train_final_loss=avg_final_loss, train_mae=avg_mae,
            val_view_loss=val_metrics['val_view_loss'], val_final_loss=val_metrics['val_final_loss'], val_mae=val_metrics['val_mae'],
            learning_rate=optimizer.param_groups[0]['lr'], num_epochs=args.num_epochs
        )
    
    final_model_path = os.path.join(save_dir, f'fold_{fold_idx + 1}', f'fold_{fold_idx + 1}_final_model.pth')
    save_model(model, optimizer, epoch, val_metrics, final_model_path)
    
    end_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    write_fold_log(fold_log_path, f"\nFold {fold_idx + 1} End-to-End Training Completed!")
    write_fold_log(fold_log_path, f"Started at: {start_time}")
    write_fold_log(fold_log_path, f"Completed at: {end_time}")
    write_fold_log(fold_log_path, f"Best Loss: {best_val_loss:.4f}")
    write_fold_log(fold_log_path, f"PLCC (at best Loss): {best_val_plcc_at_best_loss:.4f}")
    write_fold_log(fold_log_path, f"SRCC (at best Loss): {best_val_srcc_at_best_loss:.4f}")
    write_fold_log(fold_log_path, f"RMSE (at best Loss): {best_val_rmse_at_best_loss:.4f}")
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_plcc_at_best_loss': best_val_plcc_at_best_loss,
        'best_val_srcc_at_best_loss': best_val_srcc_at_best_loss,
        'best_val_rmse_at_best_loss': best_val_rmse_at_best_loss,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }


def main():
    parser = argparse.ArgumentParser(description='End-to-End Joint Training')
    
    parser.add_argument('--data_path', type=str,
                        default='/workspace/dataset/WPC_MOS_no_100.csv',
                        help='CSV data path')
    parser.add_argument('--kmeans_patches_dir', type=str,
                        default='/workspace/dataset/WPC/unified_3d_patches_raw/',
                        help='K-means patches directory')
    parser.add_argument('--pixel_coords_dir', type=str,
                       default='/workspace/dataset/WPC/pixel_coordinates_raw/',
                       help='Pixel coordinates directory path')
    parser.add_argument('--projection_dir', type=str,
                       default='/workspace/dataset/WPC_Projection/',
                       help='Projection directory path')
    parser.add_argument('--num_patches_per_view', type=int, default=9, help='Number of patches per view')
    parser.add_argument('--patch_size', type=int, default=8192, help='Patch size')
    
    parser.add_argument('--n_views', type=int, default=20, help='Number of views')
    parser.add_argument('--feature_dim', type=int, default=128, help='Feature dimension')
    
    parser.add_argument('--num_epochs', type=int, default=40, help='Total number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Number of warm-up epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_view', type=float, default=1.0, help='View loss weight')
    parser.add_argument('--lambda_policy', type=float, default=1.0, help='Policy loss weight')
    
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number')
    parser.add_argument('--save_dir', type=str, default='./endtoend_results/', help='Results save directory')
    parser.add_argument('--run_name', type=str, default='endtoend_joint', help='Experiment name')
    parser.add_argument('--use_pretrained', action='store_true', help='Use Stage 1 pretrained weights')
    
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}"
    save_dir = os.path.join(args.save_dir, f'experiment_{run_name}')
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logging(save_dir, 'endtoend')
    logger.info("=" * 60)
    logger.info("Starting End-to-End Joint Training")
    logger.info("=" * 60)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Save directory: {save_dir}")
    logger.info(f"Warmup epochs: {args.warmup_epochs}")
    logger.info(f"Total epochs: {args.num_epochs}")
    logger.info(f"Lambda view: {args.lambda_view}")
    logger.info(f"Lambda policy: {args.lambda_policy}")
    logger.info(f"Configuration: {args}")
    
    fold_results = []
    
    logger.info("Creating fold splits...")
    fold_splits = create_fold_splits(args.data_path, n_folds=5, random_seed=42)
    
    for fold_info in fold_splits:
        fold_idx = fold_info['fold_idx']
        
        logger.info("=" * 60)
        logger.info(f"Starting Fold {fold_idx + 1}/5 training")
        logger.info("=" * 60)
        log_fold_message(f"\n=== Starting Fold {fold_idx + 1} ===")
        
        try:
            result = train_endtoend_fold(fold_info, args, save_dir, logger, fold_idx)
            
            result['fold_idx'] = fold_idx
            fold_results.append(result)
            
            logger.info("=" * 60)
            logger.info(f"Fold {fold_idx + 1} training completed!")
            logger.info(f"  Best Val Loss: {result['best_val_loss']:.4f}")
            logger.info(f"  PLCC (at best Loss): {result['best_val_plcc_at_best_loss']:.4f}")
            logger.info(f"  SRCC (at best Loss): {result['best_val_srcc_at_best_loss']:.4f}")
            logger.info(f"  RMSE (at best Loss): {result['best_val_rmse_at_best_loss']:.4f}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Fold {fold_idx + 1} failed: {e}")
            logger.error("Traceback:", exc_info=True)
            continue
    
    logger.info("=" * 60)
    logger.info("End-to-End Joint Training 5-Fold completed!")
    logger.info("=" * 60)
    
    if fold_results:
        best_losses = [r['best_val_loss'] for r in fold_results]
        plccs_at_best_loss = [r['best_val_plcc_at_best_loss'] for r in fold_results]
        sroccs_at_best_loss = [r['best_val_srcc_at_best_loss'] for r in fold_results]
        rmses_at_best_loss = [r['best_val_rmse_at_best_loss'] for r in fold_results]
        
        logger.info(f"Average results (Best Loss criterion):")
        logger.info(f"  Loss: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}")
        logger.info(f"  PLCC: {np.mean(plccs_at_best_loss):.4f} ± {np.std(plccs_at_best_loss):.4f}")
        logger.info(f"  SRCC: {np.mean(sroccs_at_best_loss):.4f} ± {np.std(sroccs_at_best_loss):.4f}")
        logger.info(f"  RMSE: {np.mean(rmses_at_best_loss):.4f} ± {np.std(rmses_at_best_loss):.4f}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"5fold_summary_{timestamp}.txt"
        summary_path = os.path.join(save_dir, summary_filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("End-to-End Joint Training 5-Fold Summary\n")
            f.write("=" * 60 + "\n")
            f.write(f"Experiment name: {run_name}\n")
            f.write(f"Warmup epochs: {args.warmup_epochs}\n")
            f.write(f"Total epochs: {args.num_epochs}\n")
            f.write(f"Lambda view: {args.lambda_view}\n")
            f.write(f"Lambda policy: {args.lambda_policy}\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("[Best Loss Criterion]\n")
            f.write("=" * 60 + "\n")
            for i, result in enumerate(fold_results):
                f.write(f"Fold {i+1}:\n")
                f.write(f"  Loss: {result['best_val_loss']:.4f}\n")
                f.write(f"  PLCC: {result['best_val_plcc_at_best_loss']:.4f}\n")
                f.write(f"  SRCC: {result['best_val_srcc_at_best_loss']:.4f}\n")
                f.write(f"  RMSE: {result['best_val_rmse_at_best_loss']:.4f}\n")
            f.write("\nSummary (Best Loss):\n")
            f.write(f"  Loss: {np.mean(best_losses):.4f} ± {np.std(best_losses):.4f}\n")
            f.write(f"  PLCC: {np.mean(plccs_at_best_loss):.4f} ± {np.std(plccs_at_best_loss):.4f}\n")
            f.write(f"  SRCC: {np.mean(sroccs_at_best_loss):.4f} ± {np.std(sroccs_at_best_loss):.4f}\n")
            f.write(f"  RMSE: {np.mean(rmses_at_best_loss):.4f} ± {np.std(rmses_at_best_loss):.4f}\n")
        
        logger.info(f"5-fold summary file saved: {summary_path}")
    
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
