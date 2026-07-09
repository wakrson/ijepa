import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from src.utils.lars import LARS

class DenseFeatureCache:
    """Memory-mapped (n, n_patches, 4*D) tokens + (n, H, W) float16 depth."""

    def __init__(self, root, repr_mode, label_dtype="float"):
        root = Path(root)
        self.meta = json.loads((root / "meta.json").read_text())
        self.feats = np.load(root / "features.npy", mmap_mode="r") # (n, N, 4*D) = (n, 256, 5120)
        self.labels = np.load(root / "labels.npy", mmap_mode="r")
        self.n = self.feats.shape[0]
        self.label_dtype = label_dtype

        g = int(self.meta["n_patches"] ** 0.5) # 16
        self.grid_hw = (g, g) # (16, 16)
        self.out_hw = (self.meta["img_size"], self.meta["img_size"]) # (224, 224)

        embed_dim = self.meta["embed_dim"]
        if repr_mode == "last":
            self.col_slice = slice(self.feats.shape[2] - embed_dim, self.feats.shape[2]) # (5120-1280, 5120)
            self.dim = embed_dim
        elif repr_mode == "last4":
            self.col_slice = slice(0, self.feats.shape[2]) # (0, 51280)
            self.dim = self.feats.shape[2]
        else:
            raise ValueError(repr_mode)

    def batch(self, indices, device):
        x = self.feats[indices][:, :, self.col_slice]
        x = torch.from_numpy(np.ascontiguousarray(x)).to(device).float()
        y = torch.from_numpy(np.ascontiguousarray(self.labels[indices])).to(device)
        y = y.long() if self.label_dtype == "long" else y.float()
        return x, y

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

class LinearHead(nn.Module):
    """(B x D) -> (B x out_dim). BN-whitened linear probe."""
 
    def __init__(self, dim, out_dim, use_bn=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6) if use_bn else nn.Identity()
        self.linear = nn.Linear(dim, out_dim)
        nn.init.trunc_normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
 
    def forward(self, x):
        return self.linear(self.bn(x))
 
 
class LinearDenseHead(LinearHead):
    """(B x N x D) tokens -> (B x out_dim x H x W).
 
    Same parameters as LinearHead, applied per token, then reshaped to the
    patch grid and bilinearly upsampled to the target resolution.
    """
 
    def forward(self, tokens, grid_hw, out_hw):
        B, N, D = tokens.shape
        x = super().forward(tokens.reshape(B * N, D))
        x = x.reshape(B, *grid_hw, -1).permute(0, 3, 1, 2)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    
class ConvDenseHead(nn.Module):
    """(B x N x D) tokens -> (B x out_dim x H x W). 3x3 conv + BN + ReLU + 1x1."""
 
    def __init__(self, dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_dim, kernel_size=1),
        )
 
    def forward(self, tokens, grid_hw, out_hw):
        B, N, D = tokens.shape
        x = tokens.permute(0, 2, 1).reshape(B, D, *grid_hw)
        x = self.net(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

    
def build_head(task, head_type, dim, num_classes=None):
    """Single construction path for train and test -- config drives both."""
    if task == "classification":
        if head_type in ("linear", "bn_linear"):
            return LinearHead(dim, num_classes, use_bn="bn" in head_type)
    elif task == "segmentation":
        if head_type == "linear":
            return LinearDenseHead(dim, num_classes)
        if head_type == "conv":
            return ConvDenseHead(dim, num_classes)
    elif task == "depth":
        if head_type == "linear":
            return LinearDenseHead(dim, 1)
        if head_type == "conv":
            return ConvDenseHead(dim, 1)
    raise ValueError(f"no head '{head_type}' for task '{task}'")
