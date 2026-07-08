"""Train a segmentation linear probe on cached dense I-JEPA features
(built by src.extract_ade20k_features), mirroring the classification trainer.

Example:
python -m src.train_segmentation \
    --train-dir /media/wakr/steam/datasets/ade20kfeatures/train \
    --val-dir /media/wakr/steam/datasets/ade20kfeatures/val \
    --out-dir ./weights/segmentation/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb

from src.models.linear_probing import SegLinearHead

NUM_CLASSES = 150
IGNORE = 255


class DenseFeatureCache:
    """Memory-mapped (n, n_patches, 4*D) tokens + (n, H, W) uint8 masks."""

    def __init__(self, root, repr_mode):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.feats = np.load(root / "features.npy", mmap_mode="r")
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.n = self.feats.shape[0]

        g = int(self.meta["n_patches"] ** 0.5)
        self.grid_hw = (g, g)
        self.out_hw = (self.meta["img_size"], self.meta["img_size"])

        embed_dim = self.meta["embed_dim"]
        if repr_mode == "last":
            self.col_slice = slice(self.feats.shape[2] - embed_dim, self.feats.shape[2])
            self.dim = embed_dim
        elif repr_mode == "last4":
            self.col_slice = slice(0, self.feats.shape[2])
            self.dim = self.feats.shape[2]
        else:
            raise ValueError(repr_mode)

    def batch(self, indices, device):
        x = self.feats[indices][:, :, self.col_slice]
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device).float()
        y = torch.from_numpy(np.ascontiguousarray(self.labels[indices])).to(device).long()
        return x, y


@torch.no_grad()
def evaluate(head, cache, device, batch_size):
    head.eval()
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=device)
    for start in range(0, cache.n, batch_size):
        idx = np.arange(start, min(start + batch_size, cache.n))
        x, y = cache.batch(idx, device)
        pred = head(x, cache.grid_hw, cache.out_hw).argmax(dim=1)
        valid = y != IGNORE
        conf += torch.bincount(
            y[valid] * NUM_CLASSES + pred[valid],
            minlength=NUM_CLASSES ** 2,
        ).reshape(NUM_CLASSES, NUM_CLASSES)
    head.train()

    inter = conf.diag().float()
    union = conf.sum(0) + conf.sum(1) - conf.diag()
    iou = inter / union.clamp(min=1).float()
    present = union > 0
    return 100.0 * iou[present].mean().item()


def train(train_cache, val_cache, device, lr, wd, epochs, batch_size, out_dir, run, seed=0):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    head = SegLinearHead(train_cache.dim, NUM_CLASSES).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    steps_per_epoch = train_cache.n // batch_size
    best_miou = 0.0

    pbar = tqdm.tqdm(range(epochs), unit=" epoch", desc=f"lr={lr} wd={wd}")
    for epoch in pbar:
        perm = np_rng.permutation(train_cache.n)
        for step in tqdm.tqdm(range(steps_per_epoch), unit=" batch", leave=False,
                              desc=f"epoch {epoch + 1}/{epochs}"):
            idx = np.sort(perm[step * batch_size:(step + 1) * batch_size])
            x, y = train_cache.batch(idx, device)
            loss = criterion(head(x, train_cache.grid_hw, train_cache.out_hw), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scheduler.step()

        miou = evaluate(head, val_cache, device, batch_size)
        if miou > best_miou:
            torch.save(head.state_dict(), f"{out_dir}/model.pth")
        best_miou = max(best_miou, miou)

        run.log({"miou": miou, "loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
        pbar.set_postfix(loss=f"{loss.item():.3f}", miou=f"{miou:.2f}", best=f"{best_miou:.2f}")

    run.finish()
    pbar.close()
    return best_miou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, help="dense feature cache (train)")
    parser.add_argument("--val-dir", required=True, help="dense feature cache (val)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repr", choices=["last", "last4"], default="last4")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cache = DenseFeatureCache(args.train_dir, args.repr)
    val_cache = DenseFeatureCache(args.val_dir, args.repr)
    assert train_cache.meta["backbone"] == val_cache.meta["backbone"], \
        "train/val caches were built from different backbones"

    out_dir = Path(args.out_dir)
    run_dir = out_dir / f"run{sum(1 for _ in out_dir.glob('run*')) + 1}"
    run_dir.mkdir(parents=True)

    cfg = {
        "architecture": "SEGLP",
        "embed_dim": train_cache.meta["embed_dim"],
        "repr": args.repr,
        "num_classes": NUM_CLASSES,
        "img_size": train_cache.meta["img_size"],
        "lr": args.lr,
        "wd": args.wd,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    run = wandb.init(entity="", project="segmentation", name=run_dir.name, config=cfg)

    miou = train(train_cache, val_cache, device,
                 lr=args.lr, wd=args.wd, epochs=args.epochs,
                 batch_size=args.batch_size, out_dir=run_dir, run=run)
    cfg["val_miou"] = miou

    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()