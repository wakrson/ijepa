"""Classify ImageNet images end-to-end (frozen I-JEPA target encoder + trained
linear probe head) and sort them into <out-dir>/run#/<predicted_class>/.
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import src.models.vision_transformer as vit
from src.datasets.imagenet import ImageNet
from src.models.heads import build_head

def load_backbone(path, device):
    ckpt = torch.load(path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt["target_encoder"].items()}
    del ckpt

    proj = state["patch_embed.proj.weight"]
    embed_dim, patch_size = proj.shape[0], proj.shape[-1]
    num_patches = state["pos_embed"].shape[1]
    img_size = patch_size * int(num_patches ** 0.5)

    model_name = {v: k for k, v in vit.VIT_EMBED_DIMS.items()}[embed_dim]
    encoder = vit.__dict__[model_name](patch_size=patch_size, img_size=[img_size]).to(device)
    encoder.load_state_dict(state)
    encoder.eval()
    return encoder

@torch.no_grad()
def get_features(encoder, x, repr_mode):
    """Pooled features -- must match src/extract_features.py exactly."""
    if repr_mode == "last":
        return encoder(x).mean(dim=1)
    outs = []
    handles = [
        blk.register_forward_hook(lambda m, i, o: outs.append(o))
        for blk in encoder.blocks[-4:]
    ]
    encoder(x)
    for h in handles:
        h.remove()
    return torch.cat([encoder.norm(o).mean(dim=1) for o in outs], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="CLS-LOC root")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--backbone", required=True, help="I-JEPA pretraining checkpoint")
    parser.add_argument("--head", required=True, help="dir containing model.pth + config.json")
    parser.add_argument("--out-dir", default="./results/")
    parser.add_argument("--extra-dir", default=None, help="entries-*.npy dir (default: data-dir)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=200, help="images to classify (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dir = Path(args.head)
    cfg = json.loads((head_dir / "config.json").read_text())

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    split = ImageNet.Split(args.split)
    dataset = ImageNet(
        split=split,
        root=args.data_dir,
        extra=args.extra_dir or args.data_dir,
        transform=transform,
        target_transform=lambda t: -1 if t is None else t,  # TEST targets are None
    )
    # sorted rglob == ImageFolder/entries ordering for all three splits
    files = sorted((Path(args.data_dir) / split.value).rglob("*.JPEG"))
    assert len(files) == len(dataset), "files on disk don't match dataset entries"

    train_ds = ImageNet(split=ImageNet.Split.TRAIN, root=args.data_dir,
                        extra=args.extra_dir or args.data_dir)
    classes = [train_ds.find_class_id(i) for i in range(1000)]

    indices = range(len(dataset))
    if args.limit:
        indices = random.sample(indices, min(args.limit, len(dataset)))
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=4)

    encoder = load_backbone(args.backbone, device)
    in_dim = cfg["embed_dim"] * (4 if cfg["repr"] == "last4" else 1)
    head = build_head("classification", head_type=cfg["head_type"], dim=in_dim, num_classes=len(classes)).to(device)
    head.load_state_dict(torch.load(head_dir / "model.pth", map_location=device))
    head.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run{sum(1 for _ in out_root.glob('run*')) + 1}"
    run_dir.mkdir()

    classes = [train_ds.find_class_id(i) for i in range(1000)]
    names = [train_ds.find_class_name(i).split(",")[0].replace(" ", "_") for i in range(1000)]
    correct, done = 0, 0
    for x, y in tqdm.tqdm(loader, desc=f"classifying {split.value}"):
        feats = get_features(encoder, x.to(device), cfg["repr"])
        preds = head(feats).argmax(dim=-1).cpu()
        correct += (preds == y).sum().item()

        for pred in preds:
            src = files[indices[done]]
            dst_dir = run_dir / classes[pred]
            dst_dir.mkdir(exist_ok=True)
            shutil.copy2(src, dst_dir / f"{names[pred]}_{src.name}")
            done += 1

    summary = {
        "split": split.value,
        "backbone": args.backbone,
        "head": str(head_dir / "model.pth"),
        "config": cfg,
        "n_images": done,
    }
    if split != ImageNet.Split.TEST:
        summary["top1"] = 100.0 * correct / done
        print(f"top-1 accuracy on {done} images: {summary['top1']:.2f}%")

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {done} images to {run_dir}/")


if __name__ == "__main__":
    main()