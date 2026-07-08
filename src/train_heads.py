"""Train every head (linear + conv) for classification, segmentation, and
depth by invoking the existing per-task training scripts in sequence.

Example:
    python -m src.train_heads \
        --imagenet-dir /home/wakr/datasets/imagenetfeatures \
        --ade-dir /media/wakr/steam/datasets/ade20kfeatures \
        --nyu-dir /media/wakr/steam/datasets/nyufeatures \
        --embed-dim 1280 \
        --out-dir ./weights/ \
        --only segmentation depth        # optional filter
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", required=True, help="pooled feature cache root (train/ val/)")
    parser.add_argument("--ade-dir", required=True, help="dense ADE20K cache root (train/ val/)")
    parser.add_argument("--nyu-dir", required=True, help="dense NYUv2 cache root (train/ val/)")
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--out-dir", default="./weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--only", nargs="*", default=None, help="subset of: classification segmentation depth")
    args = parser.parse_args()

    def io(root, out_name):
        return [
            "--data-dir", f"{root}",
            "--out-dir", str(Path(args.out_dir) / out_name),
            "--epochs", str(args.epochs)]

    jobs = []
    jobs.append(
        (
            "classification", 
            [
                "-m", "src.train.classification", 
                *io(args.imagenet_dir, "classification"), 
                "--embed-dim", str(args.embed_dim), 
                "--head-type", "bn_linear"
            ]
        )
    )
    for head in ["linear", "conv"]:
        # Semantic segmentation
        jobs.append(
            (
                "segmentation", 
                [
                    "-m", "src.train.segmentation",
                     *io(args.ade_dir, "segmentation"), 
                     "--head-type", head
                ]
            )
        )
        # Depth estimation
        jobs.append(
            (
                "depth",
                [
                    "-m", 
                    "src.train.depth",
                    *io(args.nyu_dir, "depth"),
                    "--head-type",
                    head
                ]
            )
        )

    if args.only:
        jobs = [(t, cmd) for t, cmd in jobs if t in args.only]

    failed = []
    for i, (task, cmd) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {' '.join(cmd)}", flush=True)
        result = subprocess.run([sys.executable, *cmd])
        if result.returncode != 0:
            failed.append(" ".join(cmd))
            print(f"FAILED (exit {result.returncode}), continuing", flush=True)

    print(f"\ndone: {len(jobs) - len(failed)}/{len(jobs)} succeeded")
    for cmd in failed:
        print(f"failed: {cmd}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()