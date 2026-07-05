import collections
import argparse

import torch
import torchvision
import numpy as np
from PIL import Image
import os
from torchvision.datasets import VisionDataset
import matplotlib.pyplot as plt

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class ADE20K(VisionDataset):
    cmap = colormap()

    def __init__(
        self,
        root,
        split="training",
        transform=None,
        target_transform=None,
        transforms=None,
    ):
        super(ADE20K, self).__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform
        )
        assert split in ['training', 'validation'], "split should be \'training\' or \'validation\'"
        self.root = os.path.expanduser(root)
        self.split = split
        self.num_classes = 150

        img_list = []
        lbl_list = []
        img_dir = os.path.join( self.root, 'images', self.split )
        lbl_dir = os.path.join( self.root, 'annotations', self.split )

        for img_name in os.listdir( img_dir ):
            img_list.append( os.path.join( img_dir, img_name ) )
            lbl_list.append( os.path.join( lbl_dir, img_name[:-3]+'png') )

        self.img_list = img_list
        self.lbl_list = lbl_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open( self.img_list[index] )
        lbl = Image.open( self.lbl_list[index] )
        if self.transforms:
            img, lbl = self.transforms(img, lbl)
            lbl = np.array(lbl, dtype='uint8') - 1 # 1-150 => 0-149 + 255
        return img, lbl
    
    def visualize(self, index):
        img = Image.open(self.img_list[index]).convert("RGB")
        lbl = np.array(Image.open(self.lbl_list[index]), dtype='uint8') - 1
        color = self.decode_seg_to_color(lbl)

        fig, axes = plt.subplot(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title(os.path.basename(self.img_list[index]))
        axes[1].imshow(color)
        axes[1].set_title("labels")

        for ax in axes:
            ax.axis("off")
        
        fig.tight_layout()
        plt.close()

    @classmethod
    def decode_seg_to_color(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask+1]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    ds = ADE20K(args.dataset, split="validation")
    
    for i in len(ds):
        ds.visualize(i)