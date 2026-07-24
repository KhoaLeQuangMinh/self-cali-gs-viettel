#!/usr/bin/env python3
"""
render_phase2_test_poses.py
===========================
Dedicated Test Pose Renderer for hbb1/2d-gaussian-splatting on VAI_NVS_DATA_ROUND2.

Parses test_poses.csv (qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height)
and renders exact test RGB PNG images using trained 2DGS surfel models.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import torchvision
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

# Ensure 2d-gaussian-splatting root is in python path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
D2GS_DIR = os.path.join(REPO_ROOT, "2d-gaussian-splatting")
if os.path.exists(D2GS_DIR) and D2GS_DIR not in sys.path:
    sys.path.insert(0, D2GS_DIR)

try:
    from gaussian_renderer import render, GaussianModel
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov
except ImportError as e:
    print(f"[Import Error] Could not import 2DGS modules: {e}")


def qvec2rotmat(qvec):
    """Converts COLMAP quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec[0], qvec[1], qvec[2], qvec[3]
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2]
    ])


def load_test_cameras_from_csv(csv_path):
    """
    Parses test_poses.csv and returns a list of Camera objects for 2DGS rendering.
    CSV Columns: image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
    """
    df = pd.read_csv(csv_path)
    cameras = []

    for idx, row in df.iterrows():
        image_name = str(row['image_name'])
        qw, qx, qy, qz = float(row['qw']), float(row['qx']), float(row['qy']), float(row['qz'])
        tx, ty, tz = float(row['tx']), float(row['ty']), float(row['tz'])
        fx, fy = float(row['fx']), float(row['fy'])
        cx, cy = float(row['cx']), float(row['cy'])
        width, height = int(row['width']), int(row['height'])

        # Quaternion to rotation matrix (transposed for camera-to-world / world-to-camera standard)
        qvec = np.array([qw, qx, qy, qz])
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array([tx, ty, tz])

        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)

        # Create zero dummy image tensor (3, H, W) for Camera object init
        dummy_tensor = torch.zeros((3, height, width), dtype=torch.float32)

        cam = Camera(
            colmap_id=idx,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=dummy_tensor,
            gt_alpha_mask=None,
            image_name=image_name,
            uid=idx,
            data_device="cuda"
        )
        cameras.append(cam)

    return cameras


def render_phase2_test_poses(model_path, iteration, csv_path, output_dir, sh_degree=3):
    """
    Loads 2DGS trained model checkpoint and renders PNGs for test poses.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Gaussian Model
    gaussians = GaussianModel(sh_degree)
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")

    if not os.path.exists(ply_path):
        fallback_ply = os.path.join(model_path, "point_cloud.ply")
        if os.path.exists(fallback_ply):
            ply_path = fallback_ply
        else:
            # Look for highest iteration point cloud
            pc_dir = os.path.join(model_path, "point_cloud")
            if os.path.exists(pc_dir):
                iters = [int(p.split("_")[-1]) for p in os.listdir(pc_dir) if p.startswith("iteration_")]
                if iters:
                    max_iter = max(iters)
                    ply_path = os.path.join(pc_dir, f"iteration_{max_iter}", "point_cloud.ply")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"[Error] Point cloud file not found in model path: {model_path}")

    print(f"Loading 2DGS Point Cloud from: {ply_path}")
    gaussians.load_ply(ply_path)

    # Pipeline params & background color
    class PipelineParamsDummy:
        def __init__(self):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.depth_ratio = 0.0
            self.debug = False

    pipe = PipelineParamsDummy()
    bg_color = [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load cameras from csv
    cameras = load_test_cameras_from_csv(csv_path)
    print(f"Rendering {len(cameras)} test poses to: {output_dir}")

    with torch.no_grad():
        for cam in tqdm(cameras, desc="2DGS Rendering"):
            render_pkg = render(cam, gaussians, pipe, background)
            image = render_pkg["render"]
            
            # Save rendered image
            out_img_path = os.path.join(output_dir, cam.image_name)
            torchvision.utils.save_image(image, out_img_path)

    print(f"[OK] Successfully rendered {len(cameras)} test images in {output_dir}")


def main():
    parser = ArgumentParser(description="Render test poses for 2DGS models")
    parser.add_argument("--model_path", "-m", required=True, help="Path to 2DGS trained model directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Checkpoint iteration to render")
    parser.add_argument("--csv_path", required=True, help="Path to test_poses.csv")
    parser.add_argument("--output_dir", "-o", required=True, help="Output folder for rendered images")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")
    args = parser.parse_args()

    render_phase2_test_poses(
        model_path=args.model_path,
        iteration=args.iteration,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        sh_degree=args.sh_degree
    )


if __name__ == "__main__":
    main()
