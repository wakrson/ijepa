import argparse
import json
import os
import sys
from pathlib import Path
import time

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.helper import init_target_encoder, load_target_checkpoint

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

"""
python -m src.extract_features \
    --checkpoint=/home/wakr/dev/ijepa/weights/IN1K-vit.h.14-300e.pth.tar \
    --model-name="vit_huge" \
    --data-dir=/home/wakr/datasets/imagenet1k \
    --out-dir=/home/wakr/dev/ijepa/output
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained I-JEPA checkpoint")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--model-name", type=str, default="vit_huge")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--num-last-layers", type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available")
    
    model = init_target_encoder(device, args.patch_size, args.model_name)
    model, _, _, _ = load_target_checkpoint(device, args.checkpoint, model)

    model.eval().to(device)
    if device.type == "cuda":
        model.half()
    
    for param in model.parameters():
        param.requires_grad_(False)
    
    embed_dim = model.embed_dim
    k = args.num_last_layers
    feat_dim = k * embed_dim

    # Outputs from last k blocks via forward hooks
    captured = {}

    def make_hook(name: str):
        def hook(_module, _inp, out):
            captured[name] = out
        return hook
    
    handles = []
    hooked_blocks = list(model.blocks)[-k:]
    for i, blk in enumerate(hooked_blocks):
        handles.append(blk.register_forward_hook(make_hook(f"block_{i}")))

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )

    n = len(dataset)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        out_dir /= str(time.time_ns())
    
    out_dir.mkdir(exist_ok=True, parents=True)
    feats_path = out_dir / "features.npy"
    labels_path = out_dir / "labels.npy"

    feats_mm = np.lib.format.open_memmap(
        feats_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, feat_dim)
    )
    labels_mm = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.int64,
        shape=(n,)
    )

    print(f"Extracting {n} images => ({n}, {feat_dim}) fp16")

    idx = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(total=n, unit="img", smoothing=0.1)
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.half()

            _ = model(images) # (B x N_patches x D)

            per_layer = []
            for i in range(k):
                x = captured[f"block_{i}"] # (B x N x D)
                x = model.norm(x) 
                per_layer.append(x.mean(dim=1)) # (B x D)
            feats = torch.cat(per_layer, dim=-1) # (B x k*D)

            b = feats.shape[0]
            feats_mm[idx:idx + b] = feats.float().cpu().numpy().astype(np.float16)
            labels_mm[idx:idx + b] = labels.numpy()
            idx += b
            pbar.update(b)
        pbar.close()

    assert idx == n
    feats_mm.flush()
    labels_mm.flush()
    for h in handles:
        h.remove()
    
    with open(out_dir / "meta.json", "w") as f:
        json.dump({
            "patch_size": args.patch_size,
            "resolution": args.resolution,
            "checkpoint": args.checkpoint,
            "encoder": "target_encoder",
            "num_last_layers": k,
            "embed_dim": embed_dim,
            "num_images": n
        }, f)

if __name__ == "__main__":
    main()