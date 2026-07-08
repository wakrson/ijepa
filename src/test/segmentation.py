"""Segment ADE20K images end-to-end (frozen I-JEPA target encoder + trained
linear probe head) and write [image | prediction | ground truth] composites
into <out-dir>/run#/.

Example:
python -m src.test_segmentation \
    --data-dir /home/wakr/datasets/ADEChallengeData2016 \
    --split val \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --head ./weights/segmentation_lp/run1 \
    --out-dir ./results/
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import src.models.vision_transformer as vit
from src.datasets.ade20k import ADE20K
from src.models.heads import build_head

NUM_CLASSES = 150
IGNORE = 255

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
    """Patch tokens [B, N, D] -- must match src/extract_ade20k_features.py."""
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


def colorize(mask, palette):
    """(H, W) class indices -> (H, W, 3) uint8; ignore pixels stay black."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != IGNORE
    out[valid] = palette[mask[valid]]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="ADEChallengeData2016 root")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--backbone", required=True, help="I-JEPA pretraining checkpoint")
    parser.add_argument("--head", required=True, help="dir containing model.pth + config.json")
    parser.add_argument("--out-dir", default="./results/")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=50, help="images to segment (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dir = Path(args.head)
    cfg = json.loads((head_dir / "config.json").read_text())
    img_size = cfg["img_size"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def target_transform(mask):
        mask = mask.resize((img_size, img_size), Image.NEAREST)
        t = torch.from_numpy(np.array(mask)).long()
        t -= 1
        t[t == -1] = IGNORE
        return t

    dataset = ADE20K(split=ADE20K.Split(args.split), root=args.data_dir,
                     transform=transform, target_transform=target_transform)

    indices = range(len(dataset))
    if args.limit:
        indices = random.sample(indices, min(args.limit, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=4)

    encoder = load_backbone(args.backbone, img_size, device)
    in_dim = cfg["embed_dim"] * (4 if cfg["repr"] == "last4" else 1)
    head = build_head("segmentation", head_type=args.head_type, dim=in_dim, num_classes=NUM_CLASSES)
    head = build_head(in_dim, NUM_CLASSES).to(device)
    head.load_state_dict(torch.load(head_dir / "model.pth", map_location=device))
    head.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run{sum(1 for _ in out_root.glob('run*')) + 1}"
    run_dir.mkdir()

    # fixed 150-color palette (seeded -> same colors every run)
    palette = np.random.default_rng(0).integers(0, 256, (NUM_CLASSES, 3), dtype=np.uint8)

    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    done = 0
    with torch.no_grad():
        for x, y in tqdm.tqdm(loader, desc=f"segmenting {args.split}"):
            x, y = x.to(device), y.to(device)
            tokens = get_tokens(encoder, x, cfg["repr"])
            g = int(tokens.shape[1] ** 0.5)
            pred = head(tokens, (g, g), (img_size, img_size)).argmax(dim=1)

            valid = y != IGNORE
            conf += torch.bincount(
                y[valid] * NUM_CLASSES + pred[valid],
                minlength=NUM_CLASSES ** 2,
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            for p, t in zip(pred.cpu().numpy(), y.cpu().numpy()):
                relpath = dataset.image_paths[indices[done]]
                img = Image.open(Path(args.data_dir) / relpath).convert("RGB")
                img = np.array(img.resize((img_size, img_size)))

                composite = np.concatenate(
                    [img, colorize(p, palette), colorize(t, palette)], axis=1
                )
                Image.fromarray(composite).save(run_dir / f"{Path(relpath).stem}.png")
                done += 1

    inter = conf.diag().float()
    union = conf.sum(0) + conf.sum(1) - conf.diag()
    iou = inter / union.clamp(min=1).float()
    miou = 100.0 * iou[union > 0].mean().item()
    print(f"mIoU on {done} images: {miou:.2f}")

    summary = {
        "split": args.split,
        "backbone": args.backbone,
        "head": str(head_dir / "model.pth"),
        "config": cfg,
        "n_images": done,
        "miou": miou,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {done} composites to {run_dir}/")


if __name__ == "__main__":
    main()