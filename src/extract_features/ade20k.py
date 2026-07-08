"""Extract dense per-patch I-JEPA features for ADE20K, once per split.

Writes to <out-dir>/<split>/:
    features.npy  (n, n_patches, 4*embed_dim) float16, memmapped
    labels.npy    (n, img_size, img_size)     uint8 masks (255 = ignore)
    meta.json     shapes + provenance

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
from src.datasets.ade20k import ADE20K

IGNORE = 255
IMG_SIZE = 224

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def target_transform(mask):
    """ADE20K PNGs store 0=unlabeled, 1..150=classes -> 0..149, 255=ignore."""
    mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    t = torch.from_numpy(np.array(mask)).long()
    t -= 1
    t[t == -1] = IGNORE
    return t


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
    parser.add_argument("--data-dir", required=True, help="ADEChallengeData2016 root")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, embed_dim, patch_size = load_backbone(args.backbone, device)

    dataset = ADE20K(split=ADE20K.Split(args.split), root=args.data_dir,
                     transform=IMAGE_TRANSFORM, target_transform=target_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    n = len(dataset)
    n_patches = (IMG_SIZE // patch_size) ** 2
    dim = 4 * embed_dim

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+",
                                      dtype=np.float16, shape=(n, n_patches, dim))
    labels = np.lib.format.open_memmap(out_dir / "labels.npy", mode="w+",
                                       dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE))

    done = 0
    for x, y in tqdm.tqdm(loader, desc=f"extracting {args.split}"):
        tokens = get_tokens(encoder, x.to(device).half())
        b = tokens.shape[0]
        feats[done:done + b] = tokens.cpu().numpy()
        labels[done:done + b] = y.numpy().astype(np.uint8)
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
        "layers": "last4-concat, final block = last embed_dim columns",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {done} images to {out_dir}/")


if __name__ == "__main__":
    main()