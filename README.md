# I-JEPA

Official PyTorch codebase for I-JEPA (the **Image-based Joint-Embedding Predictive Architecture**) published @ CVPR-23.
[\[arXiv\]](https://arxiv.org/pdf/2301.08243.pdf) [\[JEPAs\]](https://ai.facebook.com/blog/yann-lecun-advances-in-ai-research/) [\[blogpost\]](https://ai.facebook.com/blog/yann-lecun-ai-model-i-jepa/)

## Method
I-JEPA is a method for self-supervised learning. At a high level, I-JEPA predicts the representations of part of an image from the representations of other parts of the same image. Notably, this approach learns semantic image features:
1. without relying on pre-specified invariances to hand-crafted data transformations, which tend to be biased for particular downstream tasks,
2. and without having the model fill in pixel-level details, which tend to result in learning less semantically meaningful representations.

![ijepa](https://github.com/facebookresearch/ijepa/assets/7530871/dbad94ab-ac35-433b-8b4c-ca227886d311)

## Visualizations

As opposed to generative methods that have a pixel decoder, I-JEPA has a predictor that makes predictions in latent space.
The predictor in I-JEPA can be seen as a primitive (and restricted) world-model that is able to model spatial uncertainty in a static image from a partially observable context.
This world model is semantic in the sense that it predicts high level information about unseen regions in the image, rather than pixel-level details.

We trained a stochastic decoder that maps the I-JEPA predicted representations back in pixel space as sketches.
The model correctly captures positional uncertainty and produces high-level object parts with the correct pose (e.g., dog’s head, wolf’s front legs).

![ijepa-predictor-sketch](https://github.com/facebookresearch/ijepa/assets/7530871/9b66e461-fc8b-4b12-9f06-63ec4dfc1452)
<sub>
Caption: Illustrating how the predictor learns to model the semantics of the world. For each image, the portion outside of the blue box is encoded and given to the predictor as context. The predictor outputs a representation for what it expects to be in the region within the blue box. To visualize the prediction, we train a generative model that produces a sketch of the contents represented by the predictor output, and we show a sample output within the blue box. The predictor recognizes the semantics of what parts should be filled in (the top of the dog’s head, the bird’s leg, the wolf’s legs, the other side of the building).
</sub>

## Evaluations

I-JEPA pretraining is also computationally efficient.
It does not involve any overhead associated with applying more computationally intensive data augmentations to produce multiple views.
Only one view of the image needs to be processed by the target encoder, and only the context blocks need to be processed by the context encoder.
Empirically, I-JEPA learns strong off-the-shelf semantic representations without the use of hand-crafted view augmentations.

![1percenteval](https://github.com/facebookresearch/ijepa/assets/7530871/e6e5291f-ca51-43a4-a6cf-069811094ece)
![lineareval](https://github.com/facebookresearch/ijepa/assets/7530871/d8cffa73-5350-444e-987a-7e131a86d767)

## Downstream Evaluation Tasks

A roadmap of downstream tasks for probing I-JEPA's frozen features. The paper focuses on classification and the synthetic Clevr low-level probes; the most interesting open direction is **real dense prediction** (segmentation, depth), where I-JEPA can be compared head-to-head against augmentation/curation-heavy backbones like DINOv2.

** Headline experiment — real dense prediction.** The paper argues I-JEPA captures low-level spatial structure, but only validates this on the synthetic Clevr benchmark. Freeze the encoder, train a linear head, and see whether the claim holds on real dense tasks:
- [ ] Semantic segmentation — **ADE20K** (linear probe on frozen features)
- [ ] Monocular depth estimation — **NYUv2** (linear probe on frozen features)

**Reproduce the paper's suite (baseline sanity check).** Match the reported numbers first, to validate the feature-extraction and probing harness before extending:
- [ ] ImageNet-1K linear probe
- [ ] Semi-supervised ImageNet-1K (1% labels)
- [ ] Transfer classification — CIFAR-100, Places205, iNaturalist-18 (linear probe)
- [ ] Low-level probes — Clevr/Count (object counting), Clevr/Dist (depth/distance)

**Extend dense prediction.** Broaden beyond the headline pair:
- [ ] Semantic segmentation — Cityscapes, Pascal VOC
- [ ] Monocular depth estimation — KITTI, plus NYUv2 -> SUN RGB-D as an out-of-distribution transfer check

**Notes on the I-JEPA backbone.**
- I-JEPA uses a ViT *without* a `[cls]` token — use **average-pooled patch representations** for global/image-level tasks.
- Dense-task granularity is set by the **patch size**; larger patches (e.g. ViT-g/16) helped semantic tasks but hurt low-level ones in the paper, so expect segmentation/depth to be patch-size sensitive.

## Pretrained models

<table>
  <tr>
    <th colspan="1">arch.</th>
    <th colspan="1">patch size</th>
    <th colspan="1">resolution</th>
    <th colspan="1">epochs</th>
    <th colspan="1">data</th>
    <th colspan="3">download</th>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>300</td>
    <td>ImageNet-1K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith14_ep300.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>16x16</td>
    <td>448x448</td>
    <td>300</td>
    <td>ImageNet-1K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16.448-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in1k_vith16-448_ep300.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-H</td>
    <td>14x14</td>
    <td>224x224</td>
    <td>66</td>
    <td>ImageNet-22K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vith14_ep66.yaml">configs</a></td>
  </tr>
  <tr>
    <td>ViT-g</td>
    <td>16x16</td>
    <td>224x224</td>
    <td>44</td>
    <td>ImageNet-22K</td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar">full checkpoint</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-logs-rank.0.csv">logs</a></td>
    <td><a href="https://github.com/facebookresearch/ijepa/blob/main/configs/in22k_vitg16_ep44.yaml">configs</a></td>
  </tr>
</table>

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── train.py              #   the I-JEPA training loop
│   ├── helper.py             #   helper functions for init of models & opt/loading checkpoint
│   ├── transforms.py         #   pre-train data transforms
│   ├── datasets              #   datasets, data loaders, ...
│   ├── models                #   model definitions
│   ├── masks                 #   mask collators, masking utilities, ...
│   └── utils                 #   shared utilities
├── main_distributed.py       # entrypoint for launch distributed I-JEPA pretraining on SLURM cluster
└── main.py                   # entrypoint for launch I-JEPA pretraining locally on your machine
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/](configs/) directory for example config files.

## Launching I-JEPA pretraining

### Extract features
```bash
python -m src.extract_features \
    --checkpoint=/home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --model-name="vit_huge" \
    --data-dir=/home/wakr/datasets/imagenet/ILSVRC/Data/CLS-LOC \
    --split="train" \
    --extra=/home/wakr/datasets/imagenet/extra \
    --out-dir=/media/wakr/steam/datasets/imagenetfeatures

python -m src.extract_features \
    --checkpoint=/home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --model-name="vit_huge" \
    --data-dir=/home/wakr/datasets/imagenet/ILSVRC/Data/CLS-LOC \
    --split="val" \
    --extra=/home/wakr/datasets/imagenet/extra \
    --out-dir=/media/wakr/steam/datasets/imagenetvalfeatures
```

### ImageNet-1K linear probe train
```bash
python -m src.train_classification \
    --train-dir=/media/wakr/steam/datasets/imagenetfeatures \
    --val-dir=/media/wakr/steam/datasets/imagenetvalfeatures \
    --out-dir=/home/wakr/dev/ijepa/weights/classification \
    --embed-dim=1280 \
    --repr=last4 \
    --head=bn_linear \
    --sweep \
```

### ImageNet-1K linear probe test
```bash
python -m src.test_classification \
    --data-dir=/home/wakr/datasets/imagenet/ILSVRC/Data/CLS-LOC \
    --extra=/home/wakr/datasets/imagenet/extra \
    --split="test" \
    --backbone=/home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --head=/home/wakr/dev/ijepa/weights/classification/run1 \
    --out-dir=./results/
```
### ADE20K extract features
```bash
python -m src.extract_ade20k_features \
    --data-dir /home/wakr/datasets/ADEChallengeData2016 \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --out-dir /media/wakr/steam/datasets/ade20kfeatures \
    --split train

python -m src.extract_ade20k_features \
    --data-dir /home/wakr/datasets/ADEChallengeData2016 \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --out-dir /media/wakr/steam/datasets/ade20kfeatures \
    --split val
```

### ADE20K linear probe train
```bash
python -m src.train_segmentation \
    --train-dir=/media/wakr/steam/datasets/ade20kfeatures/train \
    --val-dir=/media/wakr/steam/datasets/ade20kfeatures/val \
    --out-dir=./weights/segmentation/
    --repr=last
```

python -m src.test_segmentation \
    --data-dir /home/wakr/datasets/ADEChallengeData2016 \
    --split val \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --head ./weights/segmentation/run6 \
    --out-dir ./results/

### NYUv2 Extract Features
```bash
python -m src.extract_nyu_features \
    --data-dir /home/wakr/datasets/nyuv2 \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --out-dir /media/wakr/steam/datasets/nyufeatures \
    --split train
```

```bash
python -m src.extract_nyu_features \
    --data-dir /home/wakr/datasets/nyuv2 \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --out-dir /media/wakr/steam/datasets/nyufeatures \
    --split val
```

### Train Depth
```bash
python -m src.train_depth \
    --train-dir /media/wakr/steam/datasets/nyufeatures/train \
    --val-dir /media/wakr/steam/datasets/nyufeatures/val \
    --out-dir ./weights/depth/
```

### Test Depth
```bash
python -m src.test_depth \
    --data-dir /home/wakr/datasets/nyuv2 \
    --split val \
    --backbone /home/wakr/dev/ijepa/checkpoints/IN1K-vit.h.14-300e.pth.tar \
    --head ./weights/depth/run1 \
    --out-dir ./results/
```

### Single-GPU training
This implementation starts from the [main.py](main.py), which parses the experiment config file and runs the pre-training locally on a multi-GPU (or single-GPU) machine. For example, to run I-JEPA pretraining on GPUs "0","1", and "2" on a local machine using the config [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --devices cuda:0 cuda:1 cuda:2
```
*Note: This example is just used for illustrative purposes, as the ViT-H/14 config should be run on 16 A100 80G GPUs for an effective batch-size of 2048, in order to reproduce our results.*

### Multi-GPU training
In the multi-GPU setting, the implementation starts from [main_distributed.py](main_distributed.py), which, in addition to parsing the config file, also allows for specifying details about distributed training. For distributed training, we use the popular open-source [submitit](https://github.com/facebookincubator/submitit) tool and provide examples for a SLURM cluster.

For example, to pre-train on 16 A100 80G GPUs using the pre-training experiment configs specificed inside [configs/in1k_vith14_ep300.yaml](configs/in1k_vith14_ep300.yaml), type the command:
```
python main_distributed.py \
  --fname configs/in1k_vith14_ep300.yaml \
  --folder $path_to_save_submitit_logs \
  --partition $slurm_partition \
  --nodes 2 --tasks-per-node 8 \
  --time 1000
```

---

### Requirements
* Python 3.8 (or newer)
* PyTorch 2.0
* torchvision
* Other dependencies: pyyaml, numpy, opencv, submitit

### Installation
Install the package (and its dependencies) in editable mode. Pass the PyTorch CUDA 12.8 wheel index so that a matching GPU build of PyTorch is pulled in:
```
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu13
```

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation
```
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```