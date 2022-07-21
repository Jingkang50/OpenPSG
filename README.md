<br />
<p align="center">
  <img src="https://live.staticflickr.com/65535/52193879677_751a4e0b79_k.jpg" align="center" width="60%">

  <p align="center">
  <a href="https://arxiv.org/abs/2202.03377" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-ECCV%202022-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="http://psgdataset.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Data-psgdataset.org-228c22?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://zhuanlan.zhihu.com/p/529498676" target='_blank'>
    <img src="https://img.shields.io/badge/Benchmark-5+%20Methods-797ef6?style=flat-square">
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


## OpenPSG: Benchmarking PSG Task
### Supported methods

<details open>
<summary><b>Two-Stage Methods (6)</b></summary>

> - [x] IMP* (CVPR'17)
> - [x] MOTIFS* (CVPR'18)
> - [x] VCTree* (CVPR'19)
> - [x] GPSNet* (CVPR'20)
> - [ ] EBSGG* (CVPR'21)
> - [ ] TopicSG* (ICCV'21)
</details>

<details open>
<summary><b>One-Stage Methods (2)</b></summary>

> - [x] PSGTR (ECCV'22)
> - [x] PSGFormer (ECCV'22)
</details>


### Supported datasets

- [ ] VG-150 (IJCV'17)
- [x] PSG (ECCV'22)

| ![psg.jpg](https://live.staticflickr.com/65535/52087916793_23a27ca536_k.jpg) |
|:--:|
| <b>Comparison between classic VG-150 and PSG.</b>|


## Installation
```bash
conda env create -f environment.yml

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

## Model Zoo
Method    | Backbone | #Epoch | R/mR@20 | R/mR@50 | R/mR@100 | PQ | ckpt
---       | ---  | --- | --- | --- |--- |--- |--- |
IMP       | ResNet-50 | 12 | 283 | 290 | 286 | 289 | 289 |
MOTIFS    | ResNet-50 | 12 | 283 | 290 | 286 | 289 | 289 |
VCTree    | ResNet-50 | 12 | 283 | 290 | 286 | 289 | 289 |
GPSNet    | ResNet-50 | 12 | 283 | 290 | 286 | 289 | 289 |
PSGTR     | ResNet-50 | 12 | 283 | 290 | 286 | 289 | 289 |
PSGTR     | ResNet-50 | 60 | 283 | 290 | 286 | 289 | 289 |
PSGTR     | ResNet-101 | 60 | 283 | 290 | 286 | 289 | 289 |
PSGFormer | ResNet-50 | 12 | 283 | 290 | 286 | 289 | 289 |
PSGFormer | ResNet-50 | 60 | 283 | 290 | 286 | 289 | 289 |
PSGFormer | ResNet-101 | 60 | 283 | 290 | 286 | 289 | 289 |


## Scripts
See https://github.com/Jingkang50/OpenPSG/tree/main/scripts for scripts of each method.

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

**Visualization**

See https://mmdetection.readthedocs.io/en/v2.19.0/useful_tools.html for more tools.
```bash
# Visualize detection training dataset
PYTHONPATH='.':$PYTHONPATH \
python tools/test.py \
  configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  work_dirs/motifs_panoptic_fpn_r50_fpn_1x_sgdet_psg/latest.pth \
  --out work_dirs/motifs_panoptic_fpn_r50_fpn_1x_sgdet_psg/result.pkl \
  --eval sgdet

# Visualize evaluation results
PYTHONPATH='.':$PYTHONPATH \
python tools/vis_results.py \
  configs/motifs/panoptic_fpn_r50_fpn_1x_sgdet_psg.py \
  work_dirs/motifs_panoptic_fpn_r50_fpn_1x_sgdet_psg/result.pkl \
  work_dirs/motifs_panoptic_fpn_r50_fpn_1x_sgdet_psg/analyze_viz \
  --img_idx 3 \
  --topk 20 \
  --show-score-thr 0.3
```

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
