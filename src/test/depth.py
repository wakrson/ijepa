"""Predict NYUv2 depth end-to-end (frozen I-JEPA target encoder + trained
depth linear probe) and write [image | prediction | ground truth] composites
into <out-dir>/run#/. Depth is rendered with a fixed colormap (near = dark).

Example:
    python -m src.test_depth \
        --data-dir /home/wakr/datasets/nyuv2 \
        --split val \
        --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
        --head ./weights/depth_lp/run1 \
        --out-dir ./results/
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import src.models.vision_transformer as vit
from src.datasets.nyu import NYU
from src.models.heads import build_head

DEPTH_SCALE = 1000.0
CMAP = matplotlib.colormaps["viridis"]


def load_backbone(path, img_size, device):
    ckpt = torch.load(path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt["target_encoder"].items()}
    del ckpt

    proj = state["patch_embed.proj.weight"]
    embed_dim, patch_size = proj.shape[0], proj.shape[-1]
    model_name = {v: k for k, v in vit.VIT_EMBED_DIMS.items()}[embed_dim]
    encoder = vit.__dict__[model_name](patch_size=patch_size, img_size=[img_size]).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder


@torch.no_grad()
def get_tokens(encoder, x, repr_mode):
    """Patch tokens [B, N, D] -- must match src/extract_nyu_features.py."""
    if repr_mode == "last":
        return encoder(x)
    outs = []
    handles = [
        blk.register_forward_hook(lambda m, i, o: outs.append(o))
        for blk in encoder.blocks[-4:]
    ]
    encoder(x)
    for h in handles:
        h.remove()
    return torch.cat([encoder.norm(o) for o in outs], dim=-1)


def colorize(depth, max_depth):
    """(H, W) meters -> (H, W, 3) uint8; invalid (<= 0) pixels stay black."""
    norm = np.clip(depth / max_depth, 0.0, 1.0)
    rgb = (CMAP(norm)[..., :3] * 255).astype(np.uint8)
    rgb[depth <= 0] = 0
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="NYUv2 root")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--backbone", required=True, help="I-JEPA pretraining checkpoint")
    parser.add_argument("--head", required=True, help="dir containing model.pth + config.json")
    parser.add_argument("--out-dir", default="./results/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=50, help="images to predict (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dir = Path(args.head)
    cfg = json.loads((head_dir / "config.json").read_text())
    img_size, max_depth = cfg["img_size"], cfg["max_depth"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def target_transform(depth_png):
        depth_png = depth_png.resize((img_size, img_size), Image.NEAREST)
        return torch.from_numpy(np.array(depth_png).astype(np.float32) / DEPTH_SCALE)

    dataset = NYU(split=NYU.Split(args.split), root=args.data_dir,
                  transform=transform, target_transform=target_transform)

    indices = range(len(dataset))
    if args.limit:
        indices = random.sample(indices, min(args.limit, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=4)

    encoder = load_backbone(args.backbone, img_size, device)
    in_dim = cfg["embed_dim"] * (4 if cfg["repr"] == "last4" else 1)
    head = build_head("depth", head_type=cfg["head_type"], dim=in_dim).to(device)
    head.load_state_dict(torch.load(head_dir / "model.pth", map_location=device))
    head.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run{sum(1 for _ in out_root.glob('run*')) + 1}"
    run_dir.mkdir()

    sq_err, abs_rel, n_delta1, n_valid = 0.0, 0.0, 0, 0
    done = 0
    with torch.no_grad():
        for x, y in tqdm.tqdm(loader, desc=f"predicting {args.split}"):
            x, y = x.to(device), y.to(device)
            tokens = get_tokens(encoder, x, cfg["repr"])
            g = int(tokens.shape[1] ** 0.5)
            pred = head(tokens, (g, g), (img_size, img_size)).squeeze(1)
            pred = pred.clamp(1e-3, max_depth)

            valid = (y > 0) & (y <= max_depth)
            p, t = pred[valid], y[valid]
            sq_err += ((p - t) ** 2).sum().item()
            abs_rel += ((p - t).abs() / t).sum().item()
            n_delta1 += (torch.maximum(p / t, t / p) < 1.25).sum().item()
            n_valid += valid.sum().item()

            for pd, gt in zip(pred.cpu().numpy(), y.cpu().numpy()):
                relpath = dataset.image_paths[indices[done]]
                img = Image.open(Path(args.data_dir) / relpath).convert("RGB")
                img = np.array(img.resize((img_size, img_size)))

                composite = np.concatenate(
                    [img, colorize(pd, max_depth), colorize(gt, max_depth)], axis=1
                )
                Image.fromarray(composite).save(run_dir / f"{Path(relpath).stem}.png")
                done += 1

    metrics = {
        "rmse": (sq_err / n_valid) ** 0.5,
        "abs_rel": abs_rel / n_valid,
        "delta1": 100.0 * n_delta1 / n_valid,
    }
    print(f"n={done}  rmse={metrics['rmse']:.3f}  "
          f"abs_rel={metrics['abs_rel']:.3f}  delta1={metrics['delta1']:.1f}%")

    summary = {
        "split": args.split,
        "backbone": args.backbone,
        "head": str(head_dir / "model.pth"),
        "config": cfg,
        "n_images": done,
        **metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {done} composites to {run_dir}/")


if __name__ == "__main__":
    main()