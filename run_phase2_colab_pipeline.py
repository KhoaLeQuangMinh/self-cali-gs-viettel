#!/usr/bin/env python3
"""
Master Google Colab Orchestration Script for Phase 2 (2DGS + Refined Poses)
===========================================================================
Single-command end-to-end runner for Viettel AI Race NVS Round 2.

Features:
- Auto-compiles 2DGS diff-surfel-rasterization and simple-knn CUDA extensions
- Converts Phase 1 opt_cams.pt poses to COLMAP format
- Trains 2DGS (hbb1/2d-gaussian-splatting) for 30,000 steps per scene
- Renders test_poses.csv and zips output directly to Google Drive
- Smart Resume: skips scenes already completed
"""

import os
import sys
import glob
import shutil
import gc
import zipfile
import subprocess
import pandas as pd
from argparse import ArgumentParser

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
D2GS_DIR = os.path.join(REPO_ROOT, "2d-gaussian-splatting")
if D2GS_DIR not in sys.path:
    sys.path.insert(0, D2GS_DIR)


def install_python_dependencies():
    """Installs required Python packages for 2DGS."""
    print("=== Ensuring Python Dependencies (plyfile, trimesh, imageio, etc.) ===")
    required_packages = [
        "plyfile",
        "trimesh",
        "imageio",
        "scipy",
        "opencv-python",
        "matplotlib",
        "pandas",
        "tqdm",
        "easydict",
        "FrEIA"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + required_packages, check=True)


def compile_cuda_extensions():
    """Compiles 2d-gaussian-splatting diff-surfel-rasterization and simple-knn CUDA extensions."""
    install_python_dependencies()

    print("\n=== Compiling CUDA Extensions for 2DGS (diff-surfel-rasterization & simple-knn) ===")
    
    rasterizer_dir = os.path.join(D2GS_DIR, "submodules", "diff-surfel-rasterization")
    knn_dir = os.path.join(D2GS_DIR, "submodules", "simple-knn")

    if not os.path.exists(os.path.join(rasterizer_dir, "setup.py")):
        print(f"Submodule diff-surfel-rasterization missing in {rasterizer_dir}. Cloning...")
        if os.path.exists(rasterizer_dir):
            shutil.rmtree(rasterizer_dir)
        subprocess.run(["git", "clone", "https://github.com/hbb1/diff-surfel-rasterization", rasterizer_dir], check=True)

    if not os.path.exists(os.path.join(knn_dir, "setup.py")):
        print(f"Submodule simple-knn missing in {knn_dir}. Cloning...")
        if os.path.exists(knn_dir):
            shutil.rmtree(knn_dir)
        subprocess.run(["git", "clone", "https://gitlab.inria.fr/bkerbl/simple-knn.git", knn_dir], check=True)

    # Build diff-surfel-rasterization
    print(f"Building diff-surfel-rasterization from {rasterizer_dir}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "."], cwd=rasterizer_dir, check=True)
    
    # Build simple-knn
    print(f"Building simple-knn CUDA extension from {knn_dir}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "."], cwd=knn_dir, check=True)
    
    print("=== 2DGS CUDA Extensions Successfully Compiled ===")


def export_refined_poses(dataset_dir, saved_poses_dir, scenes):
    """Invokes export_refined_poses.py to convert opt_cams.pt to COLMAP format."""
    print("\n=== Synchronizing Refined Poses from Phase 1 ===")
    export_script = os.path.join(REPO_ROOT, "export_refined_poses.py")
    
    cmd = [
        sys.executable, export_script,
        "--dataset_dir", dataset_dir,
        "--saved_poses_dir", saved_poses_dir,
        "--scenes"
    ] + scenes
    
    print(f"Running Pose Exporter: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def verify_submission(submission_dir, dataset_dir, scenes):
    """Verifies that all images required by test_poses.csv exist for all scenes."""
    print("\n=== Verifying Phase 2 Submission Completeness ===")
    total_expected = 0
    total_found = 0
    all_ok = True

    for scene in scenes:
        csv_path = os.path.join(dataset_dir, scene, "test", "test_poses.csv")
        scene_sub_dir = os.path.join(submission_dir, scene)

        if not os.path.exists(csv_path):
            print(f"[Warning] test_poses.csv missing for scene {scene}")
            continue

        df = pd.read_csv(csv_path)
        expected_images = [str(x) for x in df['image_name'].tolist()]
        total_expected += len(expected_images)

        missing = []
        rendered_imgs = os.listdir(scene_sub_dir) if os.path.exists(scene_sub_dir) else []
        for img_name in expected_images:
            if img_name in rendered_imgs:
                total_found += 1
            else:
                missing.append(img_name)

        if missing:
            all_ok = False
            print(f"[FAILED] Scene {scene}: Missing {len(missing)} images: {missing[:5]}...")
        else:
            print(f"[OK] Scene {scene}: All {len(expected_images)} images verified.")

    print(f"Summary: Found {total_found}/{total_expected} total expected test images.")
    return all_ok


def create_submission_zip(submission_dir, zip_output_path):
    """Zips submission_phase2 directory into output zip file preserving directory structure."""
    print(f"\n=== Updating Phase 2 Submission Zip: {zip_output_path} ===")
    zip_dir = os.path.dirname(zip_output_path)
    if zip_dir:
        os.makedirs(zip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, os.path.dirname(submission_dir))
                zipf.write(abs_file, rel_file)

    zip_size_mb = os.path.getsize(zip_output_path) / (1024 * 1024)
    print(f"Successfully updated {zip_output_path} ({zip_size_mb:.2f} MB)")


def main():
    parser = ArgumentParser(description="Master Phase 2 (2DGS) Colab Orchestrator for Viettel AI Race")
    parser.add_argument("--dataset_dir", default="/content/VAI_NVS_DATA_ROUND2", help="Path to dataset directory")
    parser.add_argument("--saved_poses_dir", default="/content/drive/MyDrive/saved_poses", help="Path to saved poses directory")
    parser.add_argument("--working_dir", default="/content/working_phase2", help="Path to working output folder")
    parser.add_argument("--output_zip", default="/content/drive/MyDrive/submission_phase2_2dgs.zip", help="Path for output zip file")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations per scene (Full 30k)")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree")
    parser.add_argument("--compile_cuda", action="store_true", help="Compile CUDA submodules before running")
    parser.add_argument("--scenes", nargs="+", default=[], help="Specific scenes to process (default: all)")

    args = parser.parse_args()

    # Step 1: Compile CUDA Extensions if requested
    if args.compile_cuda:
        compile_cuda_extensions()

    # Auto-resolve nested dataset directories
    if os.path.exists(args.dataset_dir):
        nested_same = os.path.join(args.dataset_dir, os.path.basename(args.dataset_dir))
        if os.path.exists(nested_same) and os.path.isdir(nested_same):
            print(f"[Dataset Resolver] Auto-detected nested dataset folder: {nested_same}")
            args.dataset_dir = nested_same

    # Auto-detect scenes if not specified
    if not args.scenes:
        search_path = os.path.join(args.dataset_dir, "*")
        scene_candidates = [os.path.basename(p) for p in glob.glob(search_path) if os.path.isdir(p)]
        args.scenes = [s for s in scene_candidates if os.path.exists(os.path.join(args.dataset_dir, s, "train"))]

    args.scenes = sorted(args.scenes)
    print(f"Target Phase 2 scenes to process ({len(args.scenes)}): {args.scenes}")

    # Step 2: Export Phase 1 Refined Poses to COLMAP format
    export_refined_poses(args.dataset_dir, args.saved_poses_dir, args.scenes)

    submission_dir = os.path.join(args.working_dir, "submission_phase2")
    os.makedirs(submission_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Step 3: Sequential Scene Processing Loop
    for idx, scene in enumerate(args.scenes, 1):
        print(f"\n=======================================================")
        print(f" Phase 2 (2DGS) Scene [{idx}/{len(args.scenes)}]: {scene}")
        print(f"=======================================================")

        scene_train_path = os.path.join(args.dataset_dir, scene, "train")
        scene_csv_path = os.path.join(args.dataset_dir, scene, "test", "test_poses.csv")
        temp_model_dir = os.path.join(args.working_dir, "temp_model", scene)
        scene_render_output = os.path.join(submission_dir, scene)

        if not os.path.exists(scene_train_path) or not os.path.exists(scene_csv_path):
            print(f"[Error] Required paths missing for scene {scene}. Skipping.")
            continue

        # Smart Resume Check
        if os.path.exists(scene_render_output):
            df = pd.read_csv(scene_csv_path)
            expected_imgs = [str(x) for x in df['image_name'].tolist()]
            rendered_imgs = os.listdir(scene_render_output) if os.path.exists(scene_render_output) else []
            if all(img in rendered_imgs for img in expected_imgs):
                print(f"[SKIP] Scene {scene} is ALREADY fully rendered ({len(expected_imgs)} images). Skipping!")
                create_submission_zip(submission_dir, args.output_zip)
                continue

        # A. Run 2DGS Training (Official hbb1 train.py)
        train_py = os.path.join(D2GS_DIR, "train.py")
        train_cmd = [
            sys.executable, train_py,
            "-s", scene_train_path,
            "-m", temp_model_dir,
            "--iterations", str(args.iterations),
            "--sh_degree", str(args.sh_degree),
            "--eval"
        ]

        print(f"Running Phase 2 (2DGS) Training: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, env=env, check=True)

        # B. Run Test Poses Renderer
        render_py = os.path.join(REPO_ROOT, "render_phase2_test_poses.py")
        render_cmd = [
            sys.executable, render_py,
            "-m", temp_model_dir,
            "--iteration", str(args.iterations),
            "--csv_path", scene_csv_path,
            "-o", scene_render_output,
            "--sh_degree", str(args.sh_degree)
        ]
        print(f"Running 2DGS Test Pose Renderer: {' '.join(render_cmd)}")
        subprocess.run(render_cmd, env=env, check=True)

        # C. Clean Up Memory & Checkpoints
        print(f"Cleaning up temporary model files for {scene}...")
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Update zip output immediately
        create_submission_zip(submission_dir, args.output_zip)

    # Step 4: Verification & Final Zip
    is_valid = verify_submission(submission_dir, args.dataset_dir, args.scenes)
    create_submission_zip(submission_dir, args.output_zip)

    if is_valid:
        print("\n🎉 Phase 2 (2DGS) Colab Pipeline Completed Successfully!")
    else:
        print("\n⚠️ Phase 2 Colab pipeline completed with warnings. Check logs.")


if __name__ == "__main__":
    main()
