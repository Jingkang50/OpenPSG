<br />
<p align="center">
  <img src="https://live.staticflickr.com/65535/52193879677_751a4e0b79_k.jpg" align="center" width="60%">

  <p align="center">
  <a href="https://arxiv.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-ECCV%202022-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://psgdataset.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Data-psgdataset.org-228c22?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://paperswithcode.com" target='_blank'>
    <img src="https://img.shields.io/badge/Benchmark-PapersWithCode-797ef6?style=flat-square">
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
## What is PSG Task?
We introduce a new task named <strong>Panoptic Scene Graph Generation (PSG)</strong>, which aims to interpret a complex scene image with the scene graph representation, and each node in the scene graph should be grounded by its segmentation mask in the image.

To promote comprehensive scene understanding, we take account all the content in the image, including things and stuff, to generate the scene graph.

| ![psg.jpg](https://live.staticflickr.com/65535/52193735035_940fe9479c_b.jpg) |
|:--:|
| <b>PSG Task: For each image, to generate scene graph that grounded by panoptic segmentation</b>|

## Segmentation-based PSG solves many SGG problems
We believe that the biggest problem of classic scene graph generation (SGG) comes from the noisy dataset.
Classic scene graph generation datasets adopt bounding box-based object grounding, which inevitably causes a number of issues:
- **Coarse localization**: bounding boxes cannot reach pixel-level accuracy,
- **Inability to ground comprehensively**: bounding boxes cannot ground backgrounds,
- **tendency to provide trivial information**: current datasets usually capture objects like `head` to form the trivial relation of `person-has-head`, due to the large freedom of bounding box annotation.
- **Duplicate groundings**: the same object could be grounded by multiple separate bounding boxes.

All of the problems above can be easily addressed by PSG dataset, which we ground the objects using panoptic segmentation with appropriate granularity of object categories (adopted from COCO).

In fact, PSG dataset contains 49k overlapping images from COCO and Visual Genome. In the nutshell, we ask annotators to annotate relations based on COCO panoptic segmentation, i.e., relations are mask-to-mask.

| ![psg.jpg](https://live.staticflickr.com/65535/52087916793_23a27ca536_k.jpg) |
|:--:|
| <b>Comparison between classic VG-150 and PSG.</b>|

## Clear Predicate Definition
We also find that a good definition of predicates are unfortunately ignored in the previous SGG datasets.
To better formulate PSG task, we carefully define 56 predicates for PSG dataset.
We try hard to avoid trivial or duplicated relations, and find that 56 predicates are enough to cover the entire PSG dataset (or common everyday scenarios).

Type    | Predicates  |
---    | ---       |
Positional Relations (6)     | over, in front of, beside, on, in, attached to. |
Common Object-Object Relations (5) | hanging from, on the back of, falling off, going down, painted on.|
Common Actions (31) | walking on, running on, crossing, standing on, lying on, sitting on, leaning on, flying over, jumping over, jumping from, wearing, holding, carrying, looking at, guiding, kissing, eating, drinking, feeding, biting, catching, picking (grabbing), playing with, chasing, climbing, cleaning (washing, brushing), playing, touching, pushing, pulling, opening.|
Human Actions (4)	 | cooking, talking to, throwing (tossing), slicing.
Actions in Traffic Scene (4) |	driving, riding, parked on, driving on.
Actions in Sports Scene (3)	| about to hit, kicking, swinging.
Interaction between Background (3) |	entering, exiting, enclosing (surrounding, warping in)

## Updates
- **July 3, 2022**: PSG is accepted by ECCV'22.


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
```

[Datasets](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBY9ZRx3XJzfPjo8uhw5Rv6Q?e=KApssd) and [pretrained models](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErQ4stbMxp1NqP8MF8YPFG8BG-mt5geOrrJfAkeitjzASw?e=LWdJ9h) are provided. Please unzip the files if necessary.

Our codebase accesses the datasets from `./data/` and pretrained models from `./work_dirs/checkpoints/` by default.

```
├── ...
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   └── ...
│   └── psg
│       ├── psg.json
│       ├── tiny_psg.json
│       └── ...
├── openpsg
├── scripts
├── tools
├── work_dirs
│   ├── checkpoints
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
# sh scripts/basics/test_panoptic_fpn_coco.sh
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  configs/psg/panoptic_fpn_r50_fpn_1x_psg.py \
  path/to/checkpoint.pth \
  --out work_dirs/panoptic_fpn_r50_fpn/result.pkl \
  --eval PQ
```

## OpenPSG: Benchmarking PSG Task
### Supported methods (Welcome to Contribute!)

<details open>
<summary><b>Two-Stage Methods (6)</b></summary>

> - [x] IMP (CVPR'17)
> - [x] MOTIFS (CVPR'18)
> - [x] VCTree (CVPR'19)
> - [x] GPSNet (CVPR'20)
> - [ ] EBSGG (CVPR'21)
> - [ ] TopicSG (ICCV'21)
</details>

<details open>
<summary><b>One-Stage Methods (2)</b></summary>

> - [x] PSGTR (ECCV'22)
> - [x] PSGFormer (ECCV'22)
</details>


### Supported datasets

- [ ] VG-150 (IJCV'17)
- [x] PSG (ECCV'22)


## Model Zoo
Method    | Backbone | #Epoch | R/mR@20 | R/mR@50 | R/mR@100 | ckpt
---       | ---  | --- | --- | --- |--- |--- |
IMP       | ResNet-50 | 12 | 16.5 / 6.52 | 18.2 / 7.05 | 18.6 / 7.23 |  |
MOTIFS    | ResNet-50 | 12 | 20.0 / 9.10 | 21.7 / 9.57 | 22.0 / 9.69 |  |
VCTree    | ResNet-50 | 12 | 20.6 / 9.70 | 22.1 / 10.2 | 22.5 / 10.2 |  |
GPSNet    | ResNet-50 | 12 | 17.8 / 7.03 | 19.6 / 7.49 | 20.1 / 7.67 |  |
PSGTR     | ResNet-50 | 12 | 3.82 / 1.29 | 4.16 / 1.54 | 4.27 / 1.57 |  |
PSGFormer | ResNet-50 | 12 | 16.8 / 14.5 | 19.2 / 17.4 | 20.2 / 18.7 |  |
PSGTR     | ResNet-50 | 60 | 28.4 / 16.6 | 34.4 / 20.8 | 36.3 / 22.1 |  |
PSGFormer | ResNet-50 | 60 | 18.0 / 14.8 | 19.6 / 17.0 | 20.1 / 17.6 |  |



---
## Contributing
We appreciate all contributions to improve OpenPSG.
We sincerely welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](https://github.com/Jingkang50/OpenOOD/blob/v0.5/CONTRIBUTING.md) for the contributing guideline.

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
