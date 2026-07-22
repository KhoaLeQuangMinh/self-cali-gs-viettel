#
# Offline Merge Utility for Combining 7 Parallel Scene Submissions into submission_round1.zip
#

import os
import zipfile
import shutil
import pandas as pd
from argparse import ArgumentParser


def merge_scene_zips(input_dir, output_zip_path, dataset_dir):
    """
    Merges individual scene zips or scene directories from input_dir into submission_round1.zip.
    """
    temp_extract_dir = os.path.join(input_dir, "merged_submission_temp")
    submission_root = os.path.join(temp_extract_dir, "submission_round1")
    os.makedirs(submission_root, exist_ok=True)

    print(f"=== Merging Submissions from: {input_dir} ===")

    # 1. Process Zip files or Folders in input_dir
    items = os.listdir(input_dir)
    for item in items:
        item_path = os.path.join(input_dir, item)
        if item == "merged_submission_temp" or item == os.path.basename(output_zip_path):
            continue

        if item.endswith(".zip"):
            print(f"Extracting zip archive: {item}...")
            with zipfile.ZipFile(item_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    # Extract scene folders into submission_root
                    if "submission_round1/" in member:
                        rel_path = member.split("submission_round1/")[-1]
                        if rel_path:
                            target_path = os.path.join(submission_root, rel_path)
                            if member.endswith('/'):
                                os.makedirs(target_path, exist_ok=True)
                            else:
                                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                    shutil.copyfileobj(source, target)
        elif os.path.isdir(item_path):
            # Direct scene folder
            scene_name = item
            target_scene_dir = os.path.join(submission_root, scene_name)
            print(f"Copying scene folder: {scene_name}...")
            if os.path.exists(target_scene_dir):
                shutil.rmtree(target_scene_dir)
            shutil.copytree(item_path, target_scene_dir)

    # 2. Verify Completeness if dataset_dir is provided
    if dataset_dir and os.path.exists(dataset_dir):
        print("\n=== Verifying Merged Submission Completeness ===")
        all_scenes = [s for s in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, s))]
        total_expected = 0
        total_found = 0

        for scene in sorted(all_scenes):
            csv_path = os.path.join(dataset_dir, scene, "test", "test_poses.csv")
            scene_sub_dir = os.path.join(submission_root, scene)
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            expected_imgs = [str(x) for x in df['image_name'].tolist()]
            total_expected += len(expected_imgs)

            missing = []
            if os.path.exists(scene_sub_dir):
                rendered_imgs = os.listdir(scene_sub_dir)
                for img in expected_imgs:
                    if img in rendered_imgs:
                        total_found += 1
                    else:
                        missing.append(img)
            else:
                missing = expected_imgs

            if missing:
                print(f"[FAILED] Scene {scene}: Missing {len(missing)} images!")
            else:
                print(f"[OK] Scene {scene}: All {len(expected_imgs)} images present.")

        print(f"\nVerification Summary: Found {total_found}/{total_expected} total test images.")

    # 3. Create Final submission_round1.zip
    print(f"\n=== Creating Final Archive: {output_zip_path} ===")
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_root):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_file = os.path.relpath(abs_file, temp_extract_dir)
                zipf.write(abs_file, rel_file)

    # Clean up temp
    shutil.rmtree(temp_extract_dir)

    zip_size_mb = os.path.getsize(output_zip_path) / (1024 * 1024)
    print(f"🎉 Final Submission Archive Created Successfully: {output_zip_path} ({zip_size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge individual scene submissions into final submission_round1.zip")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing individual scene zips or folders")
    parser.add_argument("--output_zip", "-o", default="submission_round1.zip", help="Path to output final zip file")
    parser.add_argument("--dataset_dir", "-d", default=None, help="Path to VAI_NVS_DATA_ROUND2 root for verification")

    args = parser.parse_args()
    merge_scene_zips(args.input_dir, args.output_zip, args.dataset_dir)
