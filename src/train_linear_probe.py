import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils.lars import LARS
 

class FeatureCache:
    """Memory-mapped (N, 4*D) features + int64 labels"""

    def __init__(self, root, embed_dim, repr_mode):
        self.feats = np.load(os.path.join(root, "features.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(root, "labels.npy"))
        self.n = self.feats.shape[0]

        if repr_mode == "last":
            self.col_slice = slice(self.feats.shape[1] - embed_dim, self.feats.shape[1])
            self.dim = embed_dim
        elif repr_mode == "last4":
            self.col_slice = slice(0, self.feats.shape[1])
            self.dim = self.feats.shape[1]
        else:
            raise ValueError(repr_mode)
        
    def batch(self, indices, device):
        x = self.feats[indices, self.col_slice]
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device).float()
        y = torch.from_numpy(self.labels[indices]).to(device)
        return x, y
    
def build_head(head_type, dim, num_classes=1000):
    linear = nn.Linear(dim, num_classes)
    nn.init.trunc_normal_(linear.weight, std=0.01)
    nn.init.zeros_(linear.bias)
    if head_type == "linear":
        return linear
    elif head_type == "bn_linear":
        return nn.Sequential(nn.BatchNorm1d(dim, affine=False, eps=1e-6), linear)
    raise ValueError(head_type)
    
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
    head_type,
    epochs=50,
    batch_size=16384,
    lr_decay_epochs=15,
    lr_decay_factor=0.1,
    log_every=20,
    seed=0
):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    head = build_head(head_type, train_cache.dim).to(device)
    base_lr = ref_lr * batch_size / 256.0
    opt = LARS(head.parameters(), lr=base_lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = train_cache.n // batch_size
    best_val = 0.0
    for epoch in range(epochs):
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
            if step % log_every == 0:
                print(f"epoch {epoch} step {step / steps_per_epoch} lr : {lr} loss {loss.item()}")
            
            val_acc = evaluate(head, val_cache, device)
            best_val = max(best_val, val_acc)
            print(f"epoch {epoch} val top-1 {val_acc} (best {best_val})")
    
    return best_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--embed-dim", type=int, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--repr")