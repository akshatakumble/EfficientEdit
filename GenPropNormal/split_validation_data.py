import os
import argparse
import random
import shutil


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # Training folders
    orig_train = os.path.join(args.train_dir, "original_videos")
    edit_train = os.path.join(args.train_dir, "edited_videos")
    mask_train = os.path.join(args.train_dir, "masks")

    # Validation folders
    orig_val = os.path.join(args.val_dir, "original_videos")
    edit_val = os.path.join(args.val_dir, "edited_videos")
    mask_val = os.path.join(args.val_dir, "masks")

    # Create validation dirs
    ensure_dir(orig_val)
    ensure_dir(edit_val)
    ensure_dir(mask_val)

    # Check existence
    for p in [orig_train, edit_train, mask_train]:
        if not os.path.isdir(p):
            raise RuntimeError(f"Training folder missing: {p}")

    # List original videos (ground-truth list for dataset.py)
    all_originals = sorted([
        f for f in os.listdir(orig_train)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ])

    if len(all_originals) == 0:
        raise RuntimeError("No videos found in original_videos folder.")

    if args.num_samples > len(all_originals):
        raise ValueError("num_samples is larger than the available dataset.")

    # Randomly choose which samples become validation
    val_files = random.sample(all_originals, args.num_samples)

    print(f"Selected {len(val_files)} samples for validation.")

    # Move files from train → val
    for fname in val_files:
        base = os.path.splitext(fname)[0]

        orig_src = os.path.join(orig_train, fname)
        edit_src = os.path.join(edit_train, fname)
        mask_src = os.path.join(mask_train, fname)

        orig_dst = os.path.join(orig_val, fname)
        edit_dst = os.path.join(edit_val, fname)
        mask_dst = os.path.join(mask_val, fname)

        # Check corresponding edited + mask video exists
        if not os.path.exists(edit_src):
            raise RuntimeError(f"Missing edited video: {edit_src}")
        if not os.path.exists(mask_src):
            raise RuntimeError(f"Missing mask video: {mask_src}")

        print(f"Moving: {fname}")
        shutil.move(orig_src, orig_dst)
        shutil.move(edit_src, edit_dst)
        shutil.move(mask_src, mask_dst)

    print("\nValidation set created successfully.")
    print(f"Validation dir: {args.val_dir}")
    print("Remaining training videos:", len(os.listdir(orig_train)))


if __name__ == "__main__":
    main()
