#
# Dedicated Test Pose Renderer for Self-Cali-GS on VAI_NVS_DATA_ROUND2
#

import torch
import os
import sys
import pandas as pd
import numpy as np
import torchvision
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser

# Ensure Self-Cali-GS root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat


def load_test_cameras_from_csv(csv_path):
    """
    Parses test_poses.csv and returns a list of Camera objects configured for rendering.
    CSV format expected: image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
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

        # COLMAP quaternion to rotation matrix
        qvec = np.array([qw, qx, qy, qz])
        R = np.transpose(qvec2rotmat(qvec))
        T = np.array([tx, ty, tz])

        # Compute FoV
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)

        # Build 3x3 Intrinsic Matrix
        intrinsic_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Create dummy image tensor for camera init (not used in rendering)
        dummy_img = Image.new('RGB', (width, height), (0, 0, 0))

        cam = Camera(
            colmap_id=idx,
            R=R,
            T=T,
            intrinsic_matrix=intrinsic_matrix,
            FoVx=FoVx,
            FoVy=FoVy,
            focal_length_x=fx,
            focal_length_y=fy,
            image=dummy_img,
            gt_alpha_mask=None,
            fish_gt_image=dummy_img,
            image_name=image_name,
            uid=idx,
            orig_fov_w=width,
            orig_fov_h=height,
            original_image_resolution=(3, height, width),
            fish_gt_image_resolution=(3, height, width),
            data_device="cuda"
        )
        cameras.append(cam)

    return cameras


def render_test_poses(model_path, iteration, csv_path, output_dir, sh_degree=3):
    """
    Loads trained Gaussian model checkpoint and renders images for all test poses in test_poses.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Gaussian Model & Load Checkpoint
    gaussians = GaussianModel(sh_degree)
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")

    if not os.path.exists(ply_path):
        # Fallback check for root ply if iteration directory not present
        fallback_ply = os.path.join(model_path, "point_cloud.ply")
        if os.path.exists(fallback_ply):
            ply_path = fallback_ply
        else:
            raise FileNotFoundError(f"Point cloud file not found at: {ply_path}")

    print(f"Loading trained Gaussian checkpoint from: {ply_path}")
    gaussians.load_ply(ply_path)

    # Load Pipeline Parameters
    parser = ArgumentParser(description="Render Test Poses")
    pipeline = PipelineParams(parser).extract(parser.parse_args([]))

    # Background color (black)
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # Identity alignment matrix for baseline rendering
    global_alignment = [
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device='cuda'),
        torch.tensor([1.0], device='cuda')
    ]

    # Parse test cameras
    test_cameras = load_test_cameras_from_csv(csv_path)
    print(f"Rendering {len(test_cameras)} test views into: {output_dir}")

    rendered_count = 0
    with torch.no_grad():
        for cam in tqdm(test_cameras, desc="Rendering Test Images"):
            render_pkg = render(
                viewpoint_camera=cam,
                pc=gaussians,
                pipe=pipeline,
                bg_color=bg_color,
                mlp_color=0,
                shift_factors=None,
                hybrid=False,
                global_alignment=global_alignment
            )
            rendering = render_pkg["render"]

            # Save rendered image with exact filename from test_poses.csv
            save_path = os.path.join(output_dir, cam.image_name)
            torchvision.utils.save_image(rendering, save_path)
            rendered_count += 1

    print(f"Successfully rendered {rendered_count} images to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render test poses from CSV using Self-Cali-GS")
    parser.add_argument("--model_path", "-m", required=True, help="Path to trained model output directory")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration checkpoint to render")
    parser.add_argument("--csv_path", required=True, help="Path to test_poses.csv file")
    parser.add_argument("--output_dir", "-o", required=True, help="Directory to save rendered images")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree used in training")

    args = parser.parse_args()
    render_test_poses(
        model_path=args.model_path,
        iteration=args.iteration,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        sh_degree=args.sh_degree
    )
