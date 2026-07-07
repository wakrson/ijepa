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
from src.models.linear_probing import SegLinearHead, FeatureCache
    
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
    base_lr,
    wd,
    out_dir,
    head_type,
    num_classes=150,
    epochs=50,
    batch_size=16384,
    seed=0
):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    use_bn = True if "bn" in head_type else False
    head = SegLinearHead(train_cache.dim, num_classes=num_classes, use_bn=use_bn).to(device)

    base_lr = lr * batch_size / 16.0
    opt = torch.optim.SGD(head.parameters(), lr=base_lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    steps_per_epoch = train_cache.n // batch_size
    best_miou = 0.0


    run = wandb.init(
        entity="",
        project="segmentation_lp",
        name=out_dir.stem,
        config={
            "learning_rate": base_lr,
            "ref_lr": lr,
            "architecture": "SLP",
            "head_type": head_type,
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": wd,
            "optimizer": "sgd_momentum_0.9",
            "schedule": "cosine",
            "seed": seed,
        },
    )

    pbar = tqdm.tqdm(range(epochs), unit=" epoch", desc=f"lr={lr} wd={wd}")

    for epoch in pbar:
        head.train()
        perm = 

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
            torch.save(head.state_dict(), f"{out_dir}/model_best.pth")

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
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--repr", choices=["last", "last4"], default="last4")
    parser.add_argument("--head", choices=["linear", "bn_linear"], default="bn_linear")
    parser.add_argument("--ref-lr", type=float, default=0.05)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--out", default="probe_results.json")
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
    
    out_dir = Path(args.out_dir)
    if out_dir.exists() is False:
        out_dir.mkdir(parents=True)
    out_dir /= str(int(time.time()))
    out_dir.mkdir(parents=True)

    results = []
    for idx, (repr_mode, head_type, ref_lr, wd) in enumerate(grid):
        Path(out_dir / str(idx)).mkdir()
        train_cache = FeatureCache(args.train_dir, args.embed_dim, repr_mode)
        val_cache = FeatureCache(args.val_dir, args.embed_dim, repr_mode)
        acc = train(
            train_cache,
            val_cache,
            device,
            lr=lr,
            wd=wd,
            out_dir=out_dir / str(idx),
            head_type=head_type,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        results.append({
            "repr": repr_mode,
            "head": head_type,
            "ref_lr": ref_lr,
            "wd": wd,
            "val_top1": acc
        })

        with open(out_dir / str(idx) / "results.json", "w") as f:
            json.dump(sorted(results, key=lambda r: -r["val_top1"]), f, indent=2)

    best = max(results, key=lambda r: r["val_top1"])
    print(f"Best : {best}")

if __name__ == "__main__":
    main()