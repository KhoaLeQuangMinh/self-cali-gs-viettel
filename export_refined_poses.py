#!/usr/bin/env python3
"""
export_refined_poses.py
========================
Bridge script to export Phase 1 Self-Cali-GS refined camera poses (opt_cams.pt / cameras.json)
into standard COLMAP binary and text formats (cameras.bin/txt, images.bin/txt, points3D.bin/txt).

This enables hbb1/2d-gaussian-splatting (and any standard 3DGS pipeline) to load calibrated poses natively.
"""

import os
import sys
import glob
import shutil
import math
import torch
import struct
import numpy as np
from argparse import ArgumentParser

def rotation_matrix_to_quaternion(R):
    """Converts a 3x3 rotation matrix to COLMAP quaternion [qw, qx, qy, qz]."""
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    
    R = np.asarray(R, dtype=np.float64)
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
        
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if norm > 0:
        qvec = qvec / norm
    return qvec


def fov2focal(fov, pixels):
    """Converts field of view angle (radians) to focal length in pixels."""
    return pixels / (2.0 * math.tan(fov / 2.0))


def find_opt_cams_path(saved_poses_dir, scene):
    """Searches multiple candidate paths for opt_cams.pt for a specific scene."""
    candidate_paths = [
        os.path.join(saved_poses_dir, "content", "working", "temp_model", scene, "opt_cams.pt"),
        os.path.join(saved_poses_dir, scene, "opt_cams.pt"),
        os.path.join(saved_poses_dir, "temp_model", scene, "opt_cams.pt"),
        os.path.join(saved_poses_dir, f"{scene}_poses.json"),
        os.path.join(saved_poses_dir, scene, "cams_train30000.pt"),
    ]
    
    for path in candidate_paths:
        if os.path.exists(path):
            return path
            
    # Glob search fallback strictly matching scene directory
    pattern = os.path.join(saved_poses_dir, "**", scene, "opt_cams.pt")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0]
        
    return None


def export_colmap_format(cameras_info, output_dir):
    """
    Writes both text (cameras.txt, images.txt, points3D.txt) and binary 
    (cameras.bin, images.bin, points3D.bin) COLMAP files into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cameras_txt_path = os.path.join(output_dir, "cameras.txt")
    images_txt_path = os.path.join(output_dir, "images.txt")
    points_txt_path = os.path.join(output_dir, "points3D.txt")
    
    cameras_bin_path = os.path.join(output_dir, "cameras.bin")
    images_bin_path = os.path.join(output_dir, "images.bin")
    points_bin_path = os.path.join(output_dir, "points3D.bin")
    
    # 1. Write Text Format
    with open(cameras_txt_path, "w") as f_cam:
        f_cam.write("# Camera list with one line of data per camera:\n")
        f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam_id, cam in enumerate(cameras_info, 1):
            w = int(cam["width"])
            h = int(cam["height"])
            fx = float(cam["fx"])
            fy = float(cam["fy"])
            cx = float(cam.get("cx", w / 2.0))
            cy = float(cam.get("cy", h / 2.0))
            f_cam.write(f"{cam_id} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
            
    with open(images_txt_path, "w") as f_img:
        f_img.write("# Image list with two lines of data per image:\n")
        f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img_id, cam in enumerate(cameras_info, 1):
            qw, qx, qy, qz = cam["qvec"]
            tx, ty, tz = cam["tvec"]
            name = cam["name"]
            f_img.write(f"{img_id} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {img_id} {name}\n\n")
            
    with open(points_txt_path, "w") as f_pts:
        f_pts.write("# 3D point list with one line of data per point:\n")

    # 2. Write Binary Format
    with open(cameras_bin_path, "wb") as f_cam:
        f_cam.write(struct.pack("<Q", len(cameras_info)))
        for cam_id, cam in enumerate(cameras_info, 1):
            w = int(cam["width"])
            h = int(cam["height"])
            fx = float(cam["fx"])
            fy = float(cam["fy"])
            cx = float(cam.get("cx", w / 2.0))
            cy = float(cam.get("cy", h / 2.0))
            # PINHOLE model ID = 1
            f_cam.write(struct.pack("<i", cam_id))
            f_cam.write(struct.pack("<i", 1)) # PINHOLE
            f_cam.write(struct.pack("<Q", w))
            f_cam.write(struct.pack("<Q", h))
            f_cam.write(struct.pack("<dddd", fx, fy, cx, cy))

    with open(images_bin_path, "wb") as f_img:
        f_img.write(struct.pack("<Q", len(cameras_info)))
        for img_id, cam in enumerate(cameras_info, 1):
            qw, qx, qy, qz = cam["qvec"]
            tx, ty, tz = cam["tvec"]
            name = cam["name"].encode('utf-8') + b'\x00'
            f_img.write(struct.pack("<i", img_id))
            f_img.write(struct.pack("<dddd", qw, qx, qy, qz))
            f_img.write(struct.pack("<ddd", tx, ty, tz))
            f_img.write(struct.pack("<i", img_id))
            f_img.write(name)
            f_img.write(struct.pack("<Q", 0)) # 0 points2D

    with open(points_bin_path, "wb") as f_pts:
        f_pts.write(struct.pack("<Q", 0))

    print(f"  [OK] Exported {len(cameras_info)} cameras to COLMAP format in: {output_dir}")


def process_scene_poses(scene_train_dir, opt_cams_path, target_train_dir=None):
    """Loads opt_cams.pt / JSON and converts to COLMAP format in train/sparse/0/ and train/sparse_refined/."""
    print(f"Loading refined poses from: {opt_cams_path}")
    
    if target_train_dir is None:
        target_train_dir = scene_train_dir

    # Check if target_train_dir is writable; if not, create a fallback in current working dir
    try:
        os.makedirs(target_train_dir, exist_ok=True)
        test_file = os.path.join(target_train_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (OSError, PermissionError):
        # Read-only filesystem fallback (e.g. /kaggle/input/)
        fallback_dir = os.path.join(os.getcwd(), "processed_dataset", os.path.basename(os.path.dirname(scene_train_dir)), "train")
        print(f"[Read-Only Detected] Redirecting pose export to writable directory: {fallback_dir}")
        target_train_dir = fallback_dir
        os.makedirs(target_train_dir, exist_ok=True)

    # Symlink / Copy images directory if target_train_dir is separate from scene_train_dir
    if os.path.abspath(target_train_dir) != os.path.abspath(scene_train_dir):
        orig_images = os.path.join(scene_train_dir, "images")
        target_images = os.path.join(target_train_dir, "images")
        if os.path.exists(orig_images) and not os.path.exists(target_images):
            try:
                os.symlink(orig_images, target_images)
            except Exception:
                shutil.copytree(orig_images, target_images)

    # Patch torch.tensor and torch.device for CPU execution
    if not torch.cuda.is_available():
        _orig_tensor = torch.tensor
        def mock_tensor(*args, **kwargs):
            if 'device' in kwargs and kwargs['device'] == 'cuda':
                kwargs['device'] = 'cpu'
            return _orig_tensor(*args, **kwargs)
        torch.tensor = mock_tensor

        _orig_device = torch.device
        def mock_device(dev, *args, **kwargs):
            if isinstance(dev, str) and 'cuda' in dev and not torch.cuda.is_available():
                return _orig_device('cpu')
            return _orig_device(dev, *args, **kwargs)
        torch.device = mock_device

        import types
        mock_knn = types.ModuleType('simple_knn')
        mock_knn_C = types.ModuleType('simple_knn._C')
        mock_knn_C.distCUDA2 = lambda *args: None
        sys.modules['simple_knn'] = mock_knn
        sys.modules['simple_knn._C'] = mock_knn_C

        torch.Tensor.cuda = lambda self, *args, **kwargs: self

    self_cali_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Self-Cali-GS")
    if os.path.exists(self_cali_path) and self_cali_path not in sys.path:
        sys.path.insert(0, self_cali_path)
        
    cameras_info = []
    
    if opt_cams_path.endswith(".pt"):
        try:
            if os.path.getsize(opt_cams_path) == 0:
                raise ValueError("File size is 0 bytes")

            cams_data = torch.load(opt_cams_path, map_location="cpu", weights_only=False)
            if isinstance(cams_data, dict) and 1.0 in cams_data:
                cams = cams_data[1.0]
            elif isinstance(cams_data, dict):
                cams = list(cams_data.values())[0]
            elif isinstance(cams_data, list):
                cams = cams_data
            else:
                cams = [cams_data]

            for cam in cams:
                R = getattr(cam, "R", np.eye(3))
                if hasattr(R, "cpu"):
                    R = R.cpu().numpy()
                if R.shape == (3, 3):
                    qvec = rotation_matrix_to_quaternion(R)
                else:
                    qvec = np.array([1.0, 0.0, 0.0, 0.0])

                T = getattr(cam, "T", np.zeros(3))
                if hasattr(T, "cpu"):
                    T = T.cpu().numpy()
                tvec = T.flatten()

                w = getattr(cam, "image_width", getattr(cam, "width", 1920))
                h = getattr(cam, "image_height", getattr(cam, "height", 1080))

                if hasattr(cam, "fx") and hasattr(cam, "fy"):
                    fx, fy = float(cam.fx), float(cam.fy)
                elif hasattr(cam, "FoVx") and hasattr(cam, "FoVy"):
                    fx = fov2focal(float(cam.FoVx), w)
                    fy = fov2focal(float(cam.FoVy), h)
                else:
                    fx = fy = 1000.0

                cx = getattr(cam, "cx", w / 2.0)
                cy = getattr(cam, "cy", h / 2.0)

                name = getattr(cam, "image_name", f"{len(cameras_info):04d}.png")
                name = os.path.basename(str(name))

                # Match extension from real images directory if name missing extension
                orig_images_dir = os.path.join(scene_train_dir, "images")
                if os.path.exists(orig_images_dir) and not os.path.exists(os.path.join(orig_images_dir, name)):
                    for ext in [".JPG", ".jpg", ".png", ".jpeg", ".PNG", ".JPEG"]:
                        if os.path.exists(os.path.join(orig_images_dir, name + ext)):
                            name = name + ext
                            break

                cameras_info.append({
                    "name": name,
                    "qvec": qvec,
                    "tvec": tvec,
                    "width": w,
                    "height": h,
                    "fx": fx,
                    "fy": fy,
                    "cx": cx,
                    "cy": cy
                })
        except Exception as e:
            print(f"[Warning] Could not load corrupted/incomplete {opt_cams_path}: {e}")
            print(f"  --> Falling back to original COLMAP sparse/0 directory for {target_train_dir}")
            orig_sparse = os.path.join(scene_train_dir, "sparse", "0")
            target_sparse = os.path.join(target_train_dir, "sparse", "0")
            if os.path.exists(orig_sparse) and not os.path.exists(target_sparse):
                shutil.copytree(orig_sparse, target_sparse)
            return target_train_dir
            
    sparse_0_dir = os.path.join(target_train_dir, "sparse", "0")
    sparse_ref_dir = os.path.join(target_train_dir, "sparse_refined")
    
    export_colmap_format(cameras_info, sparse_0_dir)
    export_colmap_format(cameras_info, sparse_ref_dir)
    return target_train_dir


def main():
    parser = ArgumentParser(description="Export Phase 1 refined poses to COLMAP format for Phase 2 2DGS")
    parser.add_argument("--dataset_dir", required=True, help="Path to input dataset root (e.g., /content/VAI_NVS_DATA_ROUND2)")
    parser.add_argument("--saved_poses_dir", required=True, help="Path to saved poses directory (e.g., /content/drive/MyDrive/saved_poses)")
    parser.add_argument("--output_dir", default=None, help="Optional writable output directory for processed dataset")
    parser.add_argument("--scenes", nargs="+", default=[], help="Specific scenes to process (default: all)")
    args = parser.parse_args()

    if not args.scenes:
        search_path = os.path.join(args.dataset_dir, "*")
        args.scenes = [os.path.basename(p) for p in glob.glob(search_path) if os.path.isdir(p)]
        args.scenes = [s for s in args.scenes if os.path.exists(os.path.join(args.dataset_dir, s, "train"))]

    args.scenes = sorted(args.scenes)
    print(f"=== Converting Refined Poses for {len(args.scenes)} Scenes: {args.scenes} ===")

    for scene in args.scenes:
        scene_train_dir = os.path.join(args.dataset_dir, scene, "train")
        opt_cams_path = find_opt_cams_path(args.saved_poses_dir, scene)
        
        if not opt_cams_path:
            print(f"[Warning] Could not find opt_cams.pt for scene {scene} in {args.saved_poses_dir}. Skipping pose export for {scene}.")
            continue
            
        target_train_dir = os.path.join(args.output_dir, scene, "train") if args.output_dir else None
        process_scene_poses(scene_train_dir, opt_cams_path, target_train_dir)

    print("\n🎉 All available refined poses successfully exported to COLMAP format!")

if __name__ == "__main__":
    main()
