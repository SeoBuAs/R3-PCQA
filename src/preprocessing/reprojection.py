import numpy as np
import open3d as o3d
import os
import pickle
import glob
import time
import argparse
from sklearn.cluster import KMeans
import cv2
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def numpy_pixel_to_ray(pixels_x, pixels_y, fx, fy, cx, cy):
    pixels_x = np.asarray(pixels_x, dtype=np.float32)
    pixels_y = np.asarray(pixels_y, dtype=np.float32)
    
    x_norm = (pixels_x - cx) / fx
    y_norm = (pixels_y - cy) / fy
    z_norm = np.ones_like(x_norm)
    
    rays_camera = np.stack([x_norm, y_norm, z_norm], axis=1)
    norms = np.linalg.norm(rays_camera, axis=1, keepdims=True)
    rays_camera = rays_camera / norms
    
    return rays_camera

def numpy_ray_point_intersection(ray_origins, ray_directions, points, threshold=1.0):
    ray_origins = np.asarray(ray_origins, dtype=np.float32)
    ray_directions = np.asarray(ray_directions, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    
    n_rays = ray_origins.shape[0]
    n_points = points.shape[0]
    
    closest_indices = np.full(n_rays, -1, dtype=np.int32)
    
    batch_size = 1000
    
    for start_idx in range(0, n_rays, batch_size):
        end_idx = min(start_idx + batch_size, n_rays)
        batch_origins = ray_origins[start_idx:end_idx]
        batch_directions = ray_directions[start_idx:end_idx]
        
        to_points = points[None, :, :] - batch_origins[:, None, :]
        proj_lengths = np.sum(to_points * batch_directions[:, None, :], axis=2)
        proj_points = batch_origins[:, None, :] + proj_lengths[:, :, None] * batch_directions[:, None, :]
        distances = np.linalg.norm(points[None, :, :] - proj_points, axis=2)
        valid_mask = (proj_lengths > 0) & (distances < threshold)
        
        for i in range(end_idx - start_idx):
            valid_points = np.where(valid_mask[i])[0]
            if len(valid_points) > 0:
                valid_proj_lengths = proj_lengths[i, valid_points]
                min_idx = np.argmin(valid_proj_lengths)
                closest_indices[start_idx + i] = valid_points[min_idx]
            else:
                front_mask = proj_lengths[i] > 0
                if np.any(front_mask):
                    front_points = np.where(front_mask)[0]
                    front_distances = distances[i, front_points]
                    min_idx = np.argmin(front_distances)
                    closest_indices[start_idx + i] = front_points[min_idx]
    
    return closest_indices

def numpy_knn_search(points, center_point, k=8192):
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(center_point.reshape(1, -1), dtype=np.float32)
    
    if len(points) <= k:
        return np.arange(len(points))
    
    distances = np.sum((points - center) ** 2, axis=1)
    
    indices = np.argpartition(distances, k)[:k]
    
    return indices

def unified_kmeans_sampling(depth_image, num_regions=9):
    mask = depth_image > 0
    valid_coords = np.column_stack(np.where(mask))
    
    if len(valid_coords) < num_regions:
        return valid_coords
    
    kmeans = KMeans(n_clusters=num_regions, random_state=42, n_init=1)
    cluster_labels = kmeans.fit_predict(valid_coords)
    
    centers_with_indices = [(i, center) for i, center in enumerate(kmeans.cluster_centers_)]
    centers_with_indices.sort(key=lambda x: (x[1][0], x[1][1]))
    
    ordered_sample_points = []
    for original_idx, center in centers_with_indices:
        cluster_mask = cluster_labels == original_idx
        if np.any(cluster_mask):
            cluster_coords = valid_coords[cluster_mask]
            distances = np.sum((cluster_coords - center) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            sample_point = cluster_coords[closest_idx]
            ordered_sample_points.append(sample_point)
    
    return np.array(ordered_sample_points)

def xyz_normalize(xyz):
    new_xyz = np.copy(xyz)
    new_xyz = new_xyz - new_xyz.min()
    scale = xyz.max() - xyz.min()
    new_xyz = new_xyz / scale 
    new_xyz = new_xyz * 1000 + 1 
    return new_xyz

def extract_pixel_coordinates_and_3d_patches(ply_file, camera_file, sample_points, points, colors, intrinsic, extrinsic, patch_size=8192, ray_threshold=6.0):
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    camera_position = -R.T @ t
    
    pixels_y = sample_points[:, 0].astype(np.float32)
    pixels_x = sample_points[:, 1].astype(np.float32)
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    rays_camera = numpy_pixel_to_ray(pixels_x, pixels_y, fx, fy, cx, cy)
    
    rays_world = np.matmul(rays_camera, R.T.T)
    rays_world = rays_world / np.linalg.norm(rays_world, axis=1, keepdims=True)
    
    ray_origins = np.tile(camera_position, (len(rays_world), 1))
    
    closest_indices = numpy_ray_point_intersection(ray_origins, rays_world, points, ray_threshold)
    
    all_patches = []
    pixel_coordinates = []
    
    for kmeans_idx in range(len(sample_points)):
        closest_idx = closest_indices[kmeans_idx]
        
        pixel_coord = sample_points[kmeans_idx]
        pixel_coordinates.append(pixel_coord)
        
        if closest_idx == -1:
            patch_data = np.zeros((patch_size, 6), dtype=np.float32)
        else:
            center_point = points[closest_idx]
            
            if len(points) <= patch_size:
                patch_indices = np.arange(len(points))
            else:
                patch_indices = numpy_knn_search(points, center_point, patch_size)
            
            points_normalized = xyz_normalize(points)
            
            patch_points_norm = points_normalized[patch_indices]
            patch_colors = colors[patch_indices] if colors is not None else None
            
            if patch_colors is not None:
                patch_colors_scaled = patch_colors * 255.0
                patch_data = np.concatenate([patch_points_norm, patch_colors_scaled], axis=1)
            else:
                patch_data = np.concatenate([patch_points_norm, 
                                           np.zeros((len(patch_points_norm), 3), dtype=np.float32)], axis=1)
            
            if patch_data.shape[0] < patch_size:
                padding = np.zeros((patch_size - patch_data.shape[0], 6), dtype=np.float32)
                patch_data = np.concatenate([patch_data, padding], axis=0)
            elif patch_data.shape[0] > patch_size:
                patch_data = patch_data[:patch_size]
        
        all_patches.append(patch_data)
    
    return np.array(all_patches), np.array(pixel_coordinates)

def process_single_file(args):
    ply_file, camera_file, depth_file, coord_output_dir, npy_output_dir = args
    
    try:
        base_name = os.path.basename(depth_file).replace('_depth.png', '')
        
        if not os.path.exists(ply_file):
            return False, f"{base_name}: PLY file not found"
        
        if not os.path.exists(camera_file):
            return False, f"{base_name}: Camera file not found"
        
        depth_image = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
        
        if depth_image is None:
            return False, f"{base_name}: Image load failed"
        
        pcd = o3d.io.read_point_cloud(ply_file)
        if not pcd.has_points():
            return False, f"{base_name}: PLY load failed"
        
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        
        camera_data = np.load(camera_file, allow_pickle=True).item()
        intrinsic = np.array(camera_data['intrinsic'], dtype=np.float32)
        extrinsic = np.array(camera_data['extrinsic'], dtype=np.float32)
        
        sample_points = unified_kmeans_sampling(depth_image, num_regions=9)
        
        if len(sample_points) == 0:
            return False, f"{base_name}: No sample points"
        
        patches_3d, pixel_coordinates = extract_pixel_coordinates_and_3d_patches(
            ply_file, camera_file, sample_points, points, colors, 
            intrinsic, extrinsic, patch_size=8192, ray_threshold=6.0
        )
        
        coord_output_path = os.path.join(coord_output_dir, f"{base_name}_pixel_coords.npy")
        np.save(coord_output_path, pixel_coordinates)
        
        coord_success = 1
        npy_success = 0
        
        for i, patch_3d in enumerate(patches_3d):
            npy_output_path = os.path.join(npy_output_dir, f"{base_name}_3d_patch_{i:02d}.npy")
            np.save(npy_output_path, patch_3d)
            npy_success += 1
        
        return True, f"{base_name}: {coord_success} coords, {npy_success} 3D patches completed"
        
    except Exception as e:
        return False, f"{base_name}: Error - {str(e)[:100]}"

def find_task_files(projection_dir, ply_dir):
    print(f"Checking directories:")
    print(f"Projection dir: {projection_dir} (exists: {os.path.exists(projection_dir)})")
    print(f"PLY dir: {ply_dir} (exists: {os.path.exists(ply_dir)})")
    
    depth_files = glob.glob(os.path.join(projection_dir, "*_depth.png"))
    print(f"\nNumber of depth files: {len(depth_files)}")
    
    if len(depth_files) > 0:
        print(f"First depth file: {depth_files[0]}")
    
    ply_files_available = glob.glob(os.path.join(ply_dir, "*.ply"))
    print(f"Number of PLY files: {len(ply_files_available)}")
    
    if len(ply_files_available) > 0:
        print(f"First PLY file: {ply_files_available[0]}")
    
    tasks = []
    matched_count = 0
    unmatched_count = 0
    
    for depth_file in depth_files[:5]:
        base_name = os.path.basename(depth_file).replace('_depth.png', '')
        print(f"\nProcessing: {base_name}")
        
        parts = base_name.split('_')
        print(f"  parts: {parts}")
        
        if len(parts) >= 2 and parts[-1].isdigit() and parts[-2] == 'view':
            object_name = '_'.join(parts[:-2])
            print(f"  view_N pattern detected, object_name: {object_name}")
        elif len(parts) >= 3:
            object_name = '_'.join(parts[:-2])
            print(f"  general pattern, object_name: {object_name}")
        else:
            print(f"  pattern matching failed")
            unmatched_count += 1
            continue
        
        ply_file = os.path.join(ply_dir, f"{object_name}.ply")
        camera_file = os.path.join(projection_dir, f"{base_name}_camera.npy")
        
        print(f"  PLY file: {ply_file} (exists: {os.path.exists(ply_file)})")
        print(f"  Camera file: {camera_file} (exists: {os.path.exists(camera_file)})")
        
        if os.path.exists(ply_file) and os.path.exists(camera_file):
            tasks.append((ply_file, camera_file, depth_file))
            matched_count += 1
            print(f"  Matching success")
        else:
            unmatched_count += 1
            print(f"  Matching failed")
    
    print(f"\nMatching results: {matched_count} success, {unmatched_count} failed")
    
    if len(depth_files) > 5:
        for depth_file in depth_files[5:]:
            base_name = os.path.basename(depth_file).replace('_depth.png', '')
            parts = base_name.split('_')
            
            if len(parts) >= 2 and parts[-1].isdigit() and parts[-2] == 'view':
                object_name = '_'.join(parts[:-2])
            elif len(parts) >= 3:
                object_name = '_'.join(parts[:-2])
            else:
                continue
            
            ply_file = os.path.join(ply_dir, f"{object_name}.ply")
            camera_file = os.path.join(projection_dir, f"{base_name}_camera.npy")
            
            if os.path.exists(ply_file) and os.path.exists(camera_file):
                tasks.append((ply_file, camera_file, depth_file))
    
    return tasks

def main():
    parser = argparse.ArgumentParser(description='Extract pixel coordinates and 3D patches from projections')
    parser.add_argument('--projection_dir', type=str, required=True,
                        help='Directory containing projection results (depth, camera files)')
    parser.add_argument('--ply_dir', type=str, required=True,
                        help='Directory containing original PLY files')
    parser.add_argument('--coord_output', type=str, required=True,
                        help='Output directory for pixel coordinates')
    parser.add_argument('--patch_output', type=str, required=True,
                        help='Output directory for 3D patches')
    
    args = parser.parse_args()
    
    os.makedirs(args.coord_output, exist_ok=True)
    os.makedirs(args.patch_output, exist_ok=True)
    
    tasks = find_task_files(args.projection_dir, args.ply_dir)
    print(f"Tasks to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("No files to process.")
        return
    
    args_list = [(ply_file, camera_file, depth_file, args.coord_output, args.patch_output) 
                 for ply_file, camera_file, depth_file in tasks]
    
    start_time = time.time()
    success_count = 0
    failed_count = 0
    
    for task_args in tqdm(args_list, desc="Processing coords and 3D patches"):
        success, message = process_single_file(task_args)
        if success:
            success_count += 1
        else:
            failed_count += 1
            print(f"  {message}")
    
    end_time = time.time()
    
    print(f"\n=== Coordinate and 3D Patch Extraction Complete ===")
    print(f"Processing time: {end_time - start_time:.2f}s")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Total coordinate files: {success_count}")
    print(f"Total 3D patches: {success_count * 9}")
    print(f"Coordinate output: {args.coord_output}")
    print(f"3D output: {args.patch_output}")

if __name__ == "__main__":
    main()
