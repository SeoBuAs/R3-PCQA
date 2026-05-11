import numpy as np
import time
import open3d as o3d
import os
from PIL import Image
import cv2
import argparse

def generate_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    return path

def get_icosahedron_viewpoints():
    phi = (1 + np.sqrt(5)) / 2
    
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    vertices = vertices / np.linalg.norm(vertices[0])
    
    faces = [
        [0, 8, 4], [0, 4, 5], [0, 5, 9], [0, 9, 1], [0, 1, 8],
        [1, 9, 7], [1, 7, 6], [1, 6, 8], [2, 3, 11], [2, 11, 5],
        [2, 5, 4], [2, 4, 10], [2, 10, 3], [3, 10, 6], [3, 6, 7],
        [3, 7, 11], [4, 8, 10], [5, 11, 9], [6, 10, 8], [7, 9, 11]
    ]
    
    face_centers = []
    for face in faces:
        center = np.mean([vertices[i] for i in face], axis=0)
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    
    views = []
    for i, center in enumerate(face_centers):
        front = -center
        
        if abs(front[0]) < abs(front[1]):
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        up = np.cross(front, temp)
        up = up / np.linalg.norm(up)
        
        right = np.cross(up, front)
        right = right / np.linalg.norm(right)
        up = np.cross(front, right)
        up = up / np.linalg.norm(up)
        
        views.append({
            "name": f"view_{i+1}",
            "front": front.tolist(),
            "up": up.tolist()
        })
    
    return views

def generate_icosahedron_views(
    obj_path,
    img_root_path,
    base_filename,
    width=1080,
    height=1080,
    depth_scale=1000.0,
    rgb_point_size=4.0,
    depth_point_size=7.0,
    zoom_level=0.5,
):
    print(f"Processing: {obj_path}")

    try:
        obj = o3d.io.read_point_cloud(obj_path)
        if not obj.has_points():
            print(f"Info: Trying to read {obj_path} as mesh (was empty as point cloud).")
            obj = o3d.io.read_triangle_mesh(obj_path, True)
            if not obj.has_vertices(): raise ValueError("Could not read as Point Cloud or Mesh.")
            obj.compute_vertex_normals()
    except Exception as e:
        print(f"Error loading {obj_path}: {e}"); return

    views = get_icosahedron_viewpoints()
    obj_center = obj.get_center()
    bounding_box = obj.get_axis_aligned_bounding_box()
    extent = bounding_box.get_max_extent()
    print(f"Using Zoom Level: {zoom_level:.2f} (Object Max Extent: {extent:.2f})")

    start_time = time.time()

    for view_info in views:
        view_name = view_info["name"]
        print(f"  Generating view: {view_name}")

        vis_rgb = o3d.visualization.Visualizer()
        vis_rgb.create_window(window_name=f"RGB - {os.path.basename(obj_path)}",
                              width=width, height=height, visible=False)
        vis_rgb.add_geometry(obj)

        opt_rgb = vis_rgb.get_render_option()
        opt_rgb.background_color = np.asarray([0.0, 0.0, 0.0])
        opt_rgb.point_size = rgb_point_size
        opt_rgb.mesh_show_back_face = False
        opt_rgb.light_on = False

        ctrl_rgb = vis_rgb.get_view_control()
        ctrl_rgb.set_lookat(obj_center)
        ctrl_rgb.set_front(view_info["front"])
        ctrl_rgb.set_up(view_info["up"])
        ctrl_rgb.set_zoom(zoom_level)

        vis_rgb.poll_events()
        vis_rgb.update_renderer()

        img_float = vis_rgb.capture_screen_float_buffer(do_render=True)
        if img_float is not None:
            img_uint8 = (np.asarray(img_float) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            output_filename_rgb = f"{base_filename}_{view_name}_rgb.png"
            save_path_rgb = os.path.join(img_root_path, output_filename_rgb)
            cv2.imwrite(save_path_rgb, img_bgr)

        cam_params = ctrl_rgb.convert_to_pinhole_camera_parameters()
        intrinsic = cam_params.intrinsic.intrinsic_matrix
        extrinsic = cam_params.extrinsic
        
        camera_data = {
            'intrinsic': intrinsic.tolist(),
            'extrinsic': extrinsic.tolist(),
            'width': width,
            'height': height,
            'view_info': view_info
        }
        
        output_filename_camera = f"{base_filename}_{view_name}_camera.npy"
        save_path_camera = os.path.join(img_root_path, output_filename_camera)
        np.save(save_path_camera, camera_data)

        vis_rgb.destroy_window()
        del vis_rgb, opt_rgb, ctrl_rgb

        vis_depth = o3d.visualization.Visualizer()
        vis_depth.create_window(window_name=f"Depth - {os.path.basename(obj_path)}",
                                width=width, height=height, visible=False)
        vis_depth.add_geometry(obj)

        opt_depth = vis_depth.get_render_option()
        opt_depth.background_color = np.asarray([0.0, 0.0, 0.0])
        opt_depth.point_size = depth_point_size
        opt_depth.mesh_show_back_face = False
        opt_depth.light_on = False

        ctrl_depth = vis_depth.get_view_control()
        ctrl_depth.set_lookat(obj_center)
        ctrl_depth.set_front(view_info["front"])
        ctrl_depth.set_up(view_info["up"])
        ctrl_depth.set_zoom(zoom_level)

        vis_depth.poll_events()
        vis_depth.update_renderer()

        depth_float = vis_depth.capture_depth_float_buffer(do_render=True)
        if depth_float is not None:
            depth_np = np.asarray(depth_float)
            depth_np[np.isinf(depth_np)] = 0.0
            depth_scaled = depth_np * depth_scale
            depth_uint16 = depth_scaled.astype(np.uint16)
            output_filename_depth = f"{base_filename}_{view_name}_depth.png"
            save_path_depth = os.path.join(img_root_path, output_filename_depth)
            cv2.imwrite(save_path_depth, depth_uint16)

        vis_depth.destroy_window()
        del vis_depth, opt_depth, ctrl_depth

        if 'img_float' in locals(): del img_float
        if 'depth_float' in locals(): del depth_float

    end_time = time.time()
    print(f"Finished processing {os.path.basename(obj_path)}. Time taken: {end_time - start_time:.2f} seconds")

    del obj

def process_directory(root_path, img_root_path, width, height, depth_scale, rgb_point_size, depth_point_size, zoom_level):
    print(f"\nStarting icosahedron projection process:")
    print(f"  Source Path: {root_path}")
    print(f"  Output Path: {img_root_path} (20 Icosahedron Views - RGB & Depth)")
    print(f"  Image Size (Output): {width}x{height}")
    print(f"  Depth Scale: {depth_scale}")
    print(f"  RGB Point Size: {rgb_point_size}, Depth Point Size: {depth_point_size}")
    print(f"  Zoom Level: {zoom_level}")

    processed_files = 0

    for path, dir_list, file_list in os.walk(root_path):
        for file_name in file_list:
            if file_name.lower().endswith('.ply'):
                full_object_path = os.path.join(path, file_name)
                object_name_no_ext = os.path.splitext(file_name)[0]
                generate_icosahedron_views(
                    obj_path=full_object_path,
                    img_root_path=img_root_path,
                    base_filename=object_name_no_ext,
                    width=width,
                    height=height,
                    depth_scale=depth_scale,
                    rgb_point_size=rgb_point_size,
                    depth_point_size=depth_point_size,
                    zoom_level=zoom_level,
                )
                processed_files += 1

    print(f"\nIcosahedron projection process finished.")
    print(f"  Processed {processed_files} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 20 views from icosahedron face centers')
    parser.add_argument('--input', type=str, required=True, 
                        help='Input directory containing PLY files')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output directory for generated images')
    parser.add_argument('--width', type=int, default=1080, 
                        help='Output image width (default: 1080)')
    parser.add_argument('--height', type=int, default=1080, 
                        help='Output image height (default: 1080)')
    parser.add_argument('--depth_scale', type=float, default=1000.0, 
                        help='Depth scaling factor (default: 1000.0)')
    parser.add_argument('--rgb_point_size', type=float, default=4.0,
                        help='Point size for RGB visualization (default: 4.0)')
    parser.add_argument('--depth_point_size', type=float, default=7.0,
                        help='Point size for depth visualization (default: 7.0)')
    parser.add_argument('--zoom_level', type=float, default=0.5,
                        help='Camera zoom level (default: 0.5)')
    
    args = parser.parse_args()
    
    generate_dir(args.output)
    
    process_directory(
        args.input,
        args.output,
        args.width,
        args.height,
        args.depth_scale,
        args.rgb_point_size,
        args.depth_point_size,
        args.zoom_level,
    )
