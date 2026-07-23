#
# Master Google Colab Orchestration Script for Self-Cali-GS (Maximum A100 Settings)
#

import os
import sys
import glob
import shutil
import gc
import zipfile
import subprocess
import pandas as pd
from argparse import ArgumentParser


def install_python_dependencies():
    """Installs all required Python packages for Self-Cali-GS if missing."""
    print("=== Ensuring Python Dependencies (plyfile, FrEIA, easydict, trimesh, etc.) ===")
    required_packages = [
        "plyfile",
        "FrEIA",
        "easydict",
        "trimesh",
        "imageio",
        "scipy",
        "opencv-python",
        "matplotlib",
        "termcolor",
        "ipdb",
        "visdom",
        "wandb",
        "pandas"
    ]
    subprocess.run([sys.executable, "-m", "pip", "install"] + required_packages, check=True)


def compile_cuda_extensions(repo_root):
    """Compiles 3dgs-pose and simple-knn CUDA extensions for NVIDIA A100."""
    install_python_dependencies()

    print("\n=== Compiling CUDA Extensions for A100 (3dgs-pose & simple-knn) ===")
    
    pose_dir = os.path.join(repo_root, "3dgs-pose")
    knn_dir = os.path.join(repo_root, "simple-knn")

    # Check and clone 3dgs-pose if setup.py is missing
    if not os.path.exists(os.path.join(pose_dir, "setup.py")):
        print(f"Submodule 3dgs-pose setup.py missing in {pose_dir}. Cloning repository...")
        if os.path.exists(pose_dir):
            shutil.rmtree(pose_dir)
        subprocess.run(["git", "clone", "https://github.com/denghilbert/3dgs-pose", pose_dir], check=True)

    # Check and clone simple-knn if setup.py is missing
    if not os.path.exists(os.path.join(knn_dir, "setup.py")):
        print(f"Submodule simple-knn setup.py missing in {knn_dir}. Cloning repository...")
        if os.path.exists(knn_dir):
            shutil.rmtree(knn_dir)
        subprocess.run(["git", "clone", "https://github.com/camenduru/simple-knn", knn_dir], check=True)

    # Build 3dgs-pose (diff_gaussian_rasterization)
    print(f"Building CUDA rasterizer from {pose_dir}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "."], cwd=pose_dir, check=True)
    
    # Build simple-knn
    print(f"Building simple-knn CUDA extension from {knn_dir}...")
    subprocess.run([sys.executable, "-m", "pip", "install", "."], cwd=knn_dir, check=True)
    
    print("=== CUDA Extensions Successfully Compiled ===")


def verify_submission(submission_dir, dataset_dir, scenes):
    """Verifies that all images required by test_poses.csv exist for all scenes."""
    print("\n=== Verifying Submission Completeness ===")
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
        expected_images = df['image_name'].tolist()
        total_expected += len(expected_images)

        missing = []
        for img_name in expected_images:
            img_path = os.path.join(scene_sub_dir, str(img_name))
            if os.path.exists(img_path):
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
    """Zips submission_round1 directory into submission_round1.zip preserving directory structure."""
    print(f"\n=== Creating / Updating Submission Zip: {zip_output_path} ===")
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
    parser = ArgumentParser(description="Master Colab Orchestration for Self-Cali-GS (A100 Max Settings)")
    parser.add_argument("--dataset_dir", default="/content/VAI_NVS_DATA_ROUND2", help="Path to input dataset folder")
    parser.add_argument("--working_dir", default="/content/working", help="Path to working output folder")
    parser.add_argument("--output_zip", default="/content/submission_round1.zip", help="Path for output zip file")
    parser.add_argument("--iterations", type=int, default=30000, help="Training iterations per scene (Full 30k)")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical Harmonics degree (Full SH3)")
    parser.add_argument("--resolution", "-r", type=int, default=1, help="Image resolution scaling (1 = 100% Full Resolution)")
    parser.add_argument("--densi_num", type=float, default=0.0002, help="Original baseline densification threshold for A100 max detail")
    parser.add_argument("--opacity_threshold", type=float, default=0.005, help="Original baseline opacity threshold")
    parser.add_argument("--compile_cuda", action="store_true", help="Compile CUDA submodules before running")
    parser.add_argument("--scenes", nargs="+", default=[], help="Specific scenes to process (default: all 7)")

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Compile CUDA Extensions & Install dependencies if requested
    if args.compile_cuda:
        compile_cuda_extensions(repo_root)

    # Automatic resolution for nested unzipped dataset folders
    if os.path.exists(args.dataset_dir):
        # Check if dataset_dir contains a single nested folder with the same name
        nested_same = os.path.join(args.dataset_dir, os.path.basename(args.dataset_dir))
        if os.path.exists(nested_same) and os.path.isdir(nested_same):
            print(f"[Dataset Resolver] Auto-detected nested dataset folder: {nested_same}")
            args.dataset_dir = nested_same

    # Auto-detect scenes if not provided
    if not args.scenes:
        search_path = os.path.join(args.dataset_dir, "*")
        scene_candidates = [os.path.basename(p) for p in glob.glob(search_path) if os.path.isdir(p)]
        # Filter for folders that contain a 'train' subdirectory
        args.scenes = [s for s in scene_candidates if os.path.exists(os.path.join(args.dataset_dir, s, "train"))]

    args.scenes = sorted(args.scenes)
    print(f"Target scenes to process ({len(args.scenes)}): {args.scenes}")

    # Submission output directory
    submission_dir = os.path.join(args.working_dir, "submission_round1")
    os.makedirs(submission_dir, exist_ok=True)

    # Set CUDA memory allocation flag to prevent memory fragmentation
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Step 2: Sequential Processing Loop
    for idx, scene in enumerate(args.scenes, 1):
        print(f"\n=======================================================")
        print(f" Processing Scene [{idx}/{len(args.scenes)}]: {scene}")
        print(f"=======================================================")

        scene_train_path = os.path.join(args.dataset_dir, scene, "train")
        scene_csv_path = os.path.join(args.dataset_dir, scene, "test", "test_poses.csv")
        temp_model_dir = os.path.join(args.working_dir, "temp_model", scene)
        scene_render_output = os.path.join(submission_dir, scene)

        if not os.path.exists(scene_train_path):
            print(f"[Error] Training path missing for scene {scene}: {scene_train_path}")
            continue
        if not os.path.exists(scene_csv_path):
            print(f"[Error] test_poses.csv missing for scene {scene}: {scene_csv_path}")
            continue

        # Check if scene is already rendered (Smart Resume)
        if os.path.exists(scene_csv_path) and os.path.exists(scene_render_output):
            df = pd.read_csv(scene_csv_path)
            expected_imgs = [str(x) for x in df['image_name'].tolist()]
            rendered_imgs = os.listdir(scene_render_output) if os.path.exists(scene_render_output) else []
            if all(img in rendered_imgs for img in expected_imgs):
                print(f"[SKIP] Scene {scene} is ALREADY fully rendered ({len(expected_imgs)}/{len(expected_imgs)} images exist). Skipping!")
                create_submission_zip(submission_dir, args.output_zip)
                continue

        # A. Run Training (Full Maximum A100 Settings)
        train_cmd = [
            sys.executable, os.path.join(repo_root, "train.py"),
            "-s", scene_train_path,
            "-m", temp_model_dir,
            "--iterations", str(args.iterations),
            "--sh_degree", str(args.sh_degree),
            "-r", str(args.resolution),
            "--densi_num", str(args.densi_num),
            "--opacity_threshold", str(args.opacity_threshold),
            "--save_iterations", str(args.iterations),
            "--test_iterations", "-1",
            "--r_t_noise", "0.0", "0.0", "1.0",
            "--eval"
        ]
        print(f"Running Training (MAX A100 Settings): {' '.join(train_cmd)}")
        subprocess.run(train_cmd, env=env, check=True)

        # B. Run Test Poses Rendering
        render_cmd = [
            sys.executable, os.path.join(repo_root, "render_test_poses.py"),
            "-m", temp_model_dir,
            "--iteration", str(args.iterations),
            "--csv_path", scene_csv_path,
            "-o", scene_render_output,
            "--sh_degree", str(args.sh_degree)
        ]
        print(f"Running Renderer: {' '.join(render_cmd)}")
        subprocess.run(render_cmd, env=env, check=True)

        # C. Clean Up Scene Checkpoints (Disk & RAM Protection)
        print(f"Cleaning up temporary model files for {scene} from disk...")
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Progressive zipping: Update submission zip immediately after each scene completes!
        create_submission_zip(submission_dir, args.output_zip)

    # Step 3: Verification
    is_valid = verify_submission(submission_dir, args.dataset_dir, args.scenes)

    # Step 4: Final Zip Sync
    create_submission_zip(submission_dir, args.output_zip)

    if is_valid:
        print("\n🎉 Colab Pipeline Completed Successfully!")
    else:
        print("\n⚠️ Colab pipeline completed with warnings. Check output logs.")


if __name__ == "__main__":
    main()
