"""Extract dense per-patch I-JEPA features for NYUv2 depth, once per split.

Writes to <out-dir>/<split>/:
    features.npy  (n, n_patches, 4*embed_dim) float16, memmapped
    labels.npy    (n, img_size, img_size)     float16 depth in meters (0 = invalid)
    meta.json     shapes + provenance

Example (run for train and val):
    python -m src.extract_nyu_features \
        --data-dir /home/wakr/datasets/nyuv2 \
        --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
        --out-dir /media/wakr/steam/datasets/nyufeatures \
        --split train
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import src.models.vision_transformer as vit
from src.datasets.nyu import NYU

IMG_SIZE = 224
DEPTH_SCALE = 1000.0  # NYU depth PNGs store millimeters; /1000 -> meters

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def target_transform(depth_png):
    """16-bit depth PNG -> float32 meters, NEAREST-resized, 0 = invalid."""
    depth_png = depth_png.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    return torch.from_numpy(np.array(depth_png).astype(np.float32) / DEPTH_SCALE)


def load_backbone(path, device):
    ckpt = torch.load(path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt["target_encoder"].items()}
    del ckpt

    proj = state["patch_embed.proj.weight"]
    embed_dim, patch_size = proj.shape[0], proj.shape[-1]
    model_name = {v: k for k, v in vit.VIT_EMBED_DIMS.items()}[embed_dim]
    encoder = vit.__dict__[model_name](patch_size=patch_size, img_size=[IMG_SIZE]).to(device)
    encoder.load_state_dict(state)
    encoder.half().eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, embed_dim, patch_size


@torch.no_grad()
def get_tokens(encoder, x):
    """Last-4-block tokens, concatenated: [B, N, 4*D]."""
    outs = []
    handles = [
        blk.register_forward_hook(lambda m, i, o: outs.append(o))
        for blk in encoder.blocks[-4:]
    ]
    encoder(x)
    for h in handles:
        h.remove()
    return torch.cat([encoder.norm(o) for o in outs], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="NYUv2 root (with nyu_train.txt etc.)")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading backbone...", flush=True)
    encoder, embed_dim, patch_size = load_backbone(args.backbone, device)

    dataset = NYU(split=NYU.Split(args.split), root=args.data_dir,
                  transform=IMAGE_TRANSFORM, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    n = len(dataset)
    n_patches = (IMG_SIZE // patch_size) ** 2
    dim = 4 * embed_dim

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"allocating cache ({n} x {n_patches} x {dim} fp16)...", flush=True)
    feats = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+",
                                      dtype=np.float16, shape=(n, n_patches, dim))
    labels = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+",
                                       dtype=np.float16, shape=(n, IMG_SIZE, IMG_SIZE))

    done = 0
    for x, y in tqdm.tqdm(loader, desc=f"extracting {args.split}"):
        if done == 0:  # depth-scale sanity check
            v = y[y > 0]
            print(f"first-batch depth range: {v.min():.2f} .. {v.max():.2f} m "
                  "(expect roughly 0.5 .. 10 for NYU; if ~1000x off, fix DEPTH_SCALE)")
        tokens = get_tokens(encoder, x.to(device).half())
        b = tokens.shape[0]
        feats[done:done + b] = tokens.cpu().numpy()
        labels[done:done + b] = y.numpy().astype(np.float16)
        done += b

    feats.flush()
    labels.flush()

    meta = {
        "split": args.split,
        "backbone": args.backbone,
        "n": n,
        "n_patches": n_patches,
        "embed_dim": embed_dim,
        "patch_size": patch_size,
        "img_size": IMG_SIZE,
        "depth_scale": DEPTH_SCALE,
        "layers": "last4-concat, final block = last embed_dim columns",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {done} images to {out_dir}/")


if __name__ == "__main__":
    main()