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

class DepthLinearHead(nn.Module):
    """ (B x N x D) -> (B x C x H x W)"""
    def __init__(self, dim, use_bn=True):
        super().__init__()
        if use_bn:
            self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        else:
            self.bn = nn.Identity()
        self.linear = nn.Linear(dim, 1)
        nn.init.trunc_normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, tokens, grid_hw, out_hw):
        B, N, D = tokens.shape
        x = self.bn(tokens.reshape(B * N, D))
        x = self.linear(x)
        x = x.reshape(B, *grid_hw, -1).permute(0, 3, 1, 2)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

class SegLinearHead(nn.Module):
    """ (B x N x D) -> (B x C x H x W)"""
    def __init__(self, dim, num_classes, use_bn=True):
        super().__init__()
        if use_bn:
            self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        else:
            self.bn = nn.Identity()
        self.linear = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, tokens, grid_hw, out_hw):
        B, N, D = tokens.shape
        x = self.bn(tokens.reshape(B * N, D))
        x = self.linear(x)
        x = x.reshape(B, *grid_hw, -1).permute(0, 3, 1, 2)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    
class ClassificationLinearHead(nn.Module):
    def __init__(self, dim, num_classes, use_bn=True):
        super().__init__()
        self.bn = nn.Identity()
        if use_bn:
            self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.bn(x))