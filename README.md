# Panoptic Scene Graph Generation
<!-- <br /> -->
<!-- <p align="center">
  <img src="https://live.staticflickr.com/65535/52193879677_751a4e0b79_k.jpg" align="center" width="60%"> -->
<p align="center">
  <img src="./assets/psgtr_long.gif" align="center" width="80%">

  <p align="center">
  <a href="https://arxiv.org/abs/2207.11247" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-ECCV%202022-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://psgdataset.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Page-psgdataset.org-228c22?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.cvmart.net/race/10349/dataset" target='_blank'>
    <img src="https://img.shields.io/badge/Data-PSGDataset-334b7f?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://www.cvmart.net/race/10349/base" target='_blank'>
    <img src="https://img.shields.io/badge/Competition-PSG Challenge-f2d297?style=flat-square">
  </a>
  <br>
  <a href="https://huggingface.co/spaces/mmlab-ntu/OpenPSG" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-HuggingFace-ffca37?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://paperswithcode.com/task/panoptic-scene-graph-generation/" target='_blank'>
    <img src="https://img.shields.io/badge/Benchmark-PapersWithCode-00c4c6?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://join.slack.com/t/psgdataset/shared_invite/zt-1f8wkjfky-~uikum1YA1giLGZphFZdAQ" target='_blank'>
    <img src="https://img.shields.io/badge/Forum-Slack-4c1448?style=flat-square">
    &nbsp;&nbsp;&nbsp;
  <a href="https://replicate.com/cjwbw/openpsg" target='_blank'>
    <img src="https://img.shields.io/badge/Replicate-Demo & Cloud API-1b82c2?style=flat-square">
  </a>


</p>


  <p align="center">
  <font size=5><strong>Panoptic Scene Graph Generation</strong></font>
    <br>
      <a href="http://jingkang50.github.io/" target='_blank'>Jingkang Yang</a>,&nbsp;
      <a href="https://yizhe-ang.github.io/" target='_blank'>Yi Zhe Ang</a>,&nbsp;
      <a href="https://www.linkedin.com/in/zujin-guo-652b0417a/" target='_blank'>Zujin Guo</a>,&nbsp;
      <a href="https://kaiyangzhou.github.io/" target='_blank'>Kaiyang Zhou</a>,&nbsp;
      <a href="http://www.statfe.com/" target='_blank'>Wayne Zhang</a>,&nbsp;
      <a href="https://liuziwei7.github.io/" target='_blank'>Ziwei Liu</a>
    <br>
  S-Lab, Nanyang Technological University & SenseTime Research
  </p>
</p>


---

## Updates
- **Sep 4, 2022**: We introduce the PSG Classification Task for NTU CE7454 Coursework, as described [here](https://github.com/Jingkang50/OpenPSG/blob/main/ce7454).
- **Aug 21, 2022**: We provide guidance on PSG challenge registration [here](https://github.com/Jingkang50/OpenPSG/blob/main/psg_challenge.md).
- **Aug 12, 2022**: Replicate demo and Cloud API is added, try it [here](https://replicate.com/cjwbw/openpsg)!
- **Aug 10, 2022**: We launched [Hugging Face demo 🤗](https://huggingface.co/spaces/mmlab-ntu/OpenPSG). Try it with your scene!
- **Aug 5, 2022**: The PSG Challenge will be available on [International Algorithm Case Competition ](https://iacc.pazhoulab-huangpu.com/)! All the data will be available there then! Stay tuned!
- **July 25, 2022**: :boom: We are preparing a PSG competition with [ECCV'22 SenseHuman Workshop](https://sense-human.github.io) and [International Algorithm Case Competition](https://iacc.pazhoulab-huangpu.com/), starting from Aug 6, with a prize pool of :money_mouth_face: **US$150K** :money_mouth_face:. Join us on our [Slack](https://join.slack.com/t/psgdataset/shared_invite/zt-1f8wkjfky-~uikum1YA1giLGZphFZdAQ) to stay updated!
- **July 25, 2022**: PSG paper is available on [arXiv](https://arxiv.org/abs/2207.11247).
- **July 3, 2022**: PSG is accepted by ECCV'22.
## What is PSG Task?
<strong>The Panoptic Scene Graph Generation (PSG) Task</strong> aims to interpret a complex scene image with a scene graph representation, with each node in the scene graph grounded by its pixel-accurate segmentation mask in the image.

To promote comprehensive scene understanding, we take into account all the content in the image, including "things" and "stuff", to generate the scene graph.

| ![psg.jpg](https://live.staticflickr.com/65535/52231748332_4945d88929_b.jpg) |
|:--:|
| <b>PSG Task: To generate a scene graph that is grounded by its panoptic segmentation</b>|

<!-- ## Demo of the Current SOTA PSGTR -->


<!-- ## Demo of the Current SOTA PSGTR -->


## PSG addresses many SGG problems
We believe that the biggest problem of classic scene graph generation (SGG) comes from noisy datasets.
Classic scene graph generation datasets adopt a bounding box-based object grounding, which inevitably causes a number of issues:
- **Coarse localization**: bounding boxes cannot reach pixel-level accuracy,
- **Inability to ground comprehensively**: bounding boxes cannot ground backgrounds,
- **Tendency to provide trivial information**: current datasets usually capture frivolous objects like `head` to form trivial relations like `person-has-head`, due to too much freedom given during bounding box annotation.
- **Duplicate groundings**: the same object could be grounded by multiple separate bounding boxes.

All of the problems above can be easily addressed by the PSG dataset, which grounds the objects using panoptic segmentation with an appropriate granularity of object categories (adopted from COCO).

In fact, the PSG dataset contains 49k overlapping images from COCO and Visual Genome. In a nutshell, we asked annotators to annotate relations based on COCO panoptic segmentations, i.e., relations are mask-to-mask.

| ![psg.jpg](https://live.staticflickr.com/65535/52231743087_2bda038ee2_b.jpg) |
|:--:|
| <b>Comparison between the classic VG-150 and PSG.</b>|

## Clear Predicate Definition
We also find that a good definition of predicates is unfortunately ignored in the previous SGG datasets.
To better formulate PSG task, we carefully define 56 predicates for PSG dataset.
We try hard to avoid trivial or duplicated relations, and find that the designed 56 predicates are enough to cover the entire PSG dataset (or common everyday scenarios).

Type    | Predicates  |
---    | ---       |
Positional Relations (6)     | over, in front of, beside, on, in, attached to. |
Common Object-Object Relations (5) | hanging from, on the back of, falling off, going down, painted on.|
Common Actions (31) | walking on, running on, crossing, standing on, lying on, sitting on, leaning on, flying over, jumping over, jumping from, wearing, holding, carrying, looking at, guiding, kissing, eating, drinking, feeding, biting, catching, picking (grabbing), playing with, chasing, climbing, cleaning (washing, brushing), playing, touching, pushing, pulling, opening.|
Human Actions (4)	 | cooking, talking to, throwing (tossing), slicing.
Actions in Traffic Scene (4) |	driving, riding, parked on, driving on.
Actions in Sports Scene (3)	| about to hit, kicking, swinging.
Interaction between Background (3) |	entering, exiting, enclosing (surrounding, warping in)


## Get Started
To setup the environment, we use `conda` to manage our dependencies.

Our developers use `CUDA 10.1` to do experiments.

You can specify the appropriate `cudatoolkit` version to install on your machine in the `environment.yml` file, and then run the following to create the `conda` environment:
```bash
conda env create -f environment.yml
```
You shall manually install the following dependencies.
```bash
# Install mmcv
## CAUTION: The latest versions of mmcv 1.5.3, mmdet 2.25.0 are not well supported, due to bugs in mmdet.
pip install mmcv-full==1.4.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# Install mmdet
pip install openmim
mim install mmdet=2.20.0

# Install coco panopticapi
pip install git+https://github.com/cocodataset/panopticapi.git

# For visualization
conda install -c conda-forge pycocotools
pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html

# If you're using wandb for logging
pip install wandb
wandb login

# If you develop and run openpsg directly, install it from source:
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

[Datasets](https://www.cvmart.net/race/10349/dataset) and [pretrained models](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErQ4stbMxp1NqP8MF8YPFG8BG-mt5geOrrJfAkeitjzASw?e=9taAaU) are provided. Please unzip the files if necessary.

**Before October 2022, we only release part of the PSG data for competition, where part of the test set annotations are wiped out. Users should change the `json` filename in [`psg.py` (Line 4-5)](https://github.com/Jingkang50/OpenPSG/blob/d66dfa70429001ad80c2a8984be9d86a9da703bc/configs/_base_/datasets/psg.py#L4) to a correct filename for training or submission.**

**For the PSG competition, we provide `psg_train_val.json` (45697 training data + 1000 validation data with GT). Participant should use `psg_val_test.json` (1000 validation data with GT + 1177 test data without GT) to submit. Example submit script is [here](https://github.com/Jingkang50/OpenPSG/blob/main/scripts/imp/submit_panoptic_fpn_r50_sgdet.sh). You can use [`grade.sh`](https://github.com/Jingkang50/OpenPSG/blob/main/scripts/grade.sh) to simulate the competition's grading mechanism locally.**

Our codebase accesses the datasets from `./data/` and pretrained models from `./work_dirs/checkpoints/` by default.

```
├── ...
├── configs
├── data
│   ├── coco
│   │   ├── panoptic_train2017
│   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   └── val2017
│   └── psg
│       ├── psg_train_val.json
│       ├── psg_val_test.json
│       └── ...
├── openpsg
├── scripts
├── tools
├── work_dirs
│   ├── checkpoints
│   ├── psgtr_r50
│   └── ...
├── ...
```
We suggest our users to play with `./tools/Visualize_Dataset.ipynb` to quickly get familiar with PSG dataset.

To train or test PSG models, please see https://github.com/Jingkang50/OpenPSG/tree/main/scripts for scripts of each method. Some example scripts are below.

**Training**
```bash
# Single GPU for two-stage methods, debug mode
PYTHONPATH='.':$PYTHONPATH \
python -m pdb -c continue tools/train.py \
  configs/psg/motif_panoptic_fpn_r50_fpn_1x_sgdet_psg.py

# Multiple GPUs for one-stage methods, running mode
PYTHONPATH='.':$PYTHONPATH \
python -m torch.distributed.launch \
--nproc_per_node=8 --master_port=29500 \
  tools/train.py \
  configs/psgformer/psgformer_r50_psg.py \
  --gpus 8 \
  --launcher pytorch
```

**Testing**
```bash
# sh scripts/imp/test_panoptic_fpn_r50_sgdet.sh
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  configs/imp/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  path/to/checkpoint.pth \
  --eval sgdet
```

**Submitting for PSG Competition**
```bash
# sh scripts/imp/submit_panoptic_fpn_r50_sgdet.sh
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  configs/imp/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  path/to/checkpoint.pth \
  --submit
```

## OpenPSG: Benchmarking PSG Task
### Supported methods (Welcome to Contribute!)

<details open>
<summary><b>Two-Stage Methods (4)</b></summary>

> - [x] IMP (CVPR'17)
> - [x] MOTIFS (CVPR'18)
> - [x] VCTree (CVPR'19)
> - [x] GPSNet (CVPR'20)
</details>

<details open>
<summary><b>One-Stage Methods (2)</b></summary>

> - [x] PSGTR (ECCV'22)
> - [x] PSGFormer (ECCV'22)
</details>


### Supported datasets (Welcome to Contribute!)

- [ ] VG-150 (IJCV'17)
- [x] PSG (ECCV'22)


## Model Zoo
Method    | Backbone | #Epoch | R/mR@20 | R/mR@50 | R/mR@100 | ckpt | SHA256
---       | ---  | --- | --- | --- |--- |--- |--- |
IMP       | ResNet-50 | 12 | 16.5 / 6.52 | 18.2 / 7.05 | 18.6 / 7.23 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EiTgJ9q2h3hDpyXSdu6BtlQBHAZNwNaYmcO7SElxhkIFXw?e=8fytHc) |7be2842b6664e2b9ef6c7c05d27fde521e2401ffe67dbb936438c69e98f9783c |
MOTIFS    | ResNet-50 | 12 | 20.0 / 9.10 | 21.7 / 9.57 | 22.0 / 9.69 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eh4hvXIspUFKpNa_75qwDoEBJTCIozTLzm49Ste6HaoPow?e=ZdAs6z) | 956471959ca89acae45c9533fb9f9a6544e650b8ea18fe62cdead495b38751b8 |
VCTree    | ResNet-50 | 12 | 20.6 / 9.70 | 22.1 / 10.2 | 22.5 / 10.2 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EhKfi9kqAd9CnSoHztQIChABeBjBD3hF7DflrNCjlHfh9A?e=lWa1bd) |e5fdac7e6cc8d9af7ae7027f6d0948bf414a4a605ed5db4d82c5d72de55c9b58 |
GPSNet    | ResNet-50 | 12 | 17.8 / 7.03 | 19.6 / 7.49 | 20.1 / 7.67 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EipIhZgVgx1LuK2RUmjRg2sB8JqxMIS5GnPDHeaYy5GF6A?e=5j53VF) | 98cd7450925eb88fa311a20fce74c96f712e45b7f29857c5cdf9b9dd57f59c51 |
PSGTR     | ResNet-50 | 60 | 28.4 / 16.6 | 34.4 / 20.8 | 36.3 / 22.1 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eonc-KwOxg9EmdtGDX6ss-gB35QpKDnN_1KSWOj6U8sZwQ?e=zdqwqP) | 1c4ddcbda74686568b7e6b8145f7f33030407e27e390c37c23206f95c51829ed |
PSGFormer | ResNet-50 | 60 | 18.0 / 14.8 | 19.6 / 17.0 | 20.1 / 17.6 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EnaJchJzJPtGrkl4k09evPIB5JUkkDZ2tSS9F-Hd-1KYzA?e=9QA8Nc) | 2f0015ce67040fa00b65986f6ce457c4f8cc34720f7e47a656b462b696a013b7 |

---
## Contributing
We appreciate all contributions to improve OpenPSG.
We sincerely welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](https://github.com/Jingkang50/OpenOOD/blob/v0.5/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgements
OpenPSG is developed based on [MMDetection](https://github.com/open-mmlab/mmdetection). Most of the two-stage SGG implementations refer to [MMSceneGraph](https://github.com/Kenneth-Wong/MMSceneGraph) and [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).
We sincerely appreciate the efforts of the developers from the previous codebases.

## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{yang2022psg,
    author = {Yang, Jingkang and Ang, Yi Zhe and Guo, Zujin and Zhou, Kaiyang and Zhang, Wayne and Liu, Ziwei},
    title = {Panoptic Scene Graph Generation},
    booktitle = {ECCV}
    year = {2022}
}
```
