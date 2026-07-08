import json
import time
import argparse
import itertools
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import tqdm
import wandb

from src.utils.lars import LARS
from src.models.heads import build_head, FeatureCache
    
def evaluate(head, cache, device, batch_size):
    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, cache.n, batch_size):
            idx = np.arange(start, min(start + batch_size, cache.n))
            x, y = cache.batch(idx, device)
            pred = head(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    head.train()
    return 100.0 * correct / total

def train_one_config(
    train_cache,
    val_cache,
    device,
    ref_lr,
    wd,
    out_dir,
    run=None,
    head_type=None,
    epochs=50,
    batch_size=16384,
    lr_decay_epochs=15,
    lr_decay_factor=0.1,
    seed=0
):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    head = build_head("classification", head_type=head_type, dim=train_cache.dim, num_classes=1000).to(device)
    base_lr = ref_lr * batch_size / 256.0
    opt = LARS(head.parameters(), lr=base_lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = train_cache.n // batch_size
    best_val = 0.0

    pbar = tqdm.tqdm(
        range(epochs),
        unit=" epoch",
        desc=f"lr={ref_lr} wd={wd}"
    )

    for epoch in pbar:
        lr = base_lr * (lr_decay_factor ** (epoch // lr_decay_epochs))
        for g in opt.param_groups:
            g['lr'] = lr

        perm = np_rng.permutation(train_cache.n)
        for step in range(steps_per_epoch):
            idx = np.sort(perm[step * batch_size:(step + 1) * batch_size])
            x, y = train_cache.batch(idx, device)
            loss = criterion(head(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
        val_acc = evaluate(head, val_cache, device, batch_size)

        if val_acc > best_val:
            torch.save(head.state_dict(), f"{out_dir}/model.pth")

        best_val = max(best_val, val_acc)

        run.log({"acc": val_acc, "loss": loss.item()})

        pbar.set_postfix(
            lr=f"{lr}",
            loss=f"{loss.item()}",
            val=f"{val_acc}",
            best=f"{best_val}"
        )
    run.finish()
    pbar.close()

    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--repr", choices=["last", "last4"], default="last4")
    parser.add_argument("--head", choices=["linear", "bn_linear"], default="bn_linear")
    parser.add_argument("--ref-lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--head-type", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.sweep:
        grid = list(itertools.product(
            ["last", "last4"],
            ["linear", "bn_linear"],
            [0.01, 0.05, 0.001],
            [0.0005, 0.0],
        ))
    else:
        grid = [(args.repr, args.head, args.ref_lr, args.wd)]
    
    # Make directory
    out_dir = Path(args.out_dir)

    for idx, (repr_mode, head_type, ref_lr, wd) in enumerate(grid):
        run_dir = out_dir / f"run{sum(1 for _ in out_dir.glob('run*')) + 1}"
        run_dir.mkdir(parents=True)
        
        train_cache = FeatureCache(Path(args.data_dir) / "train", args.embed_dim, repr_mode)
        val_cache = FeatureCache(Path(args.data_dir) / "val", args.embed_dim, repr_mode)

        cfg = {
            "embed_dim": args.embed_dim,
            "architecture": "CLSLP",
            "repr": repr_mode,
            "head": head_type,
            "ref_lr": ref_lr,
            "wd": wd,
            "lr_decay_epochs": 15,
            "lr_decay_factor": 0.1,
            "seed": 0,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }

        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        run = wandb.init(
            entity="",
            project=f"classification",
            name=out_dir.stem,
            config=cfg
        )

        acc = train_one_config(
            train_cache,
            val_cache,
            device,
            run=run,
            ref_lr=ref_lr,
            wd=wd,
            out_dir=run_dir,
            head_type=head_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        cfg["val_top1"] = acc

        with open(run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

if __name__ == "__main__":
    main()