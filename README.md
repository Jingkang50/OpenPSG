# Panoptic Scene Graph Generation
<br />
<p align="center">
  <img src="https://live.staticflickr.com/65535/52193879677_751a4e0b79_k.jpg" align="center" width="60%">

  <p align="center">
  <a href="https://arxiv.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-ECCV%202022-b31b1b?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://psgdataset.org/" target='_blank'>
    <img src="https://img.shields.io/badge/Page-psgdataset.org-228c22?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://entuedu-my.sharepoint.com/personal/jingkang001_e_ntu_edu_sg/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjingkang001%5Fe%5Fntu%5Fedu%5Fsg%2FDocuments%2Fopenpsg%2Fdata&ga=1" target='_blank'>
    <img src="https://img.shields.io/badge/Data-PSGDataset-334b7f?style=flat-square">
  </a>
  <br>
  <a href="https://huggingface.co/" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-HuggingFace-ffca37?style=flat-square">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://paperswithcode.com" target='_blank'>
    <img src="https://img.shields.io/badge/Benchmark-PapersWithCode-00c4c6?style=flat-square">
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
<strong>The Panoptic Scene Graph Generation (PSG) Task</strong> aims to interpret a complex scene image with a scene graph representation, and each node in the scene graph should be grounded by its pixel-accurate segmentation mask in the image.

To promote comprehensive scene understanding, we take account all the content in the image, including things and stuff, to generate the scene graph.

| ![psg.jpg](https://live.staticflickr.com/65535/52231748332_4945d88929_b.jpg) |
|:--:|
| <b>PSG Task: To generate a scene graph that is grounded by the panoptic segmentation</b>|

## PSG addresses many SGG problems
We believe that the biggest problem of classic scene graph generation (SGG) comes from the noisy dataset.
Classic scene graph generation datasets adopt bounding box-based object grounding, which inevitably causes a number of issues:
- **Coarse localization**: bounding boxes cannot reach pixel-level accuracy,
- **Inability to ground comprehensively**: bounding boxes cannot ground backgrounds,
- **Tendency to provide trivial information**: current datasets usually capture objects like `head` to form the trivial relation of `person-has-head`, due to the large freedom of bounding box annotation.
- **Duplicate groundings**: the same object could be grounded by multiple separate bounding boxes.

All of the problems above can be easily addressed by PSG dataset, which we ground the objects using panoptic segmentation with appropriate granularity of object categories (adopted from COCO).

In fact, PSG dataset contains 49k overlapping images from COCO and Visual Genome. In the nutshell, we ask annotators to annotate relations based on COCO panoptic segmentation, i.e., relations are mask-to-mask.

| ![psg.jpg](https://live.staticflickr.com/65535/52231743087_2bda038ee2_b.jpg) |
|:--:|
| <b>Comparison between classic VG-150 and PSG.</b>|

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

## Updates
- **July 22, 2022**: We submit the paper to arXiv and will appear on July 25.
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

# If you develop and run openpsg directly, install it from source:
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

[Datasets](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EgQzvsYo3t9BpxgMZ6VHaEMBY9ZRx3XJzfPjo8uhw5Rv6Q?e=KApssd) and [pretrained models](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/ErQ4stbMxp1NqP8MF8YPFG8BG-mt5geOrrJfAkeitjzASw?e=9taAaU) are provided. Please unzip the files if necessary.

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
IMP       | ResNet-50 | 12 | 16.5 / 6.52 | 18.2 / 7.05 | 18.6 / 7.23 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EiTgJ9q2h3hDpyXSdu6BtlQBHAZNwNaYmcO7SElxhkIFXw?e=8fytHc) |
MOTIFS    | ResNet-50 | 12 | 20.0 / 9.10 | 21.7 / 9.57 | 22.0 / 9.69 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eh4hvXIspUFKpNa_75qwDoEBJTCIozTLzm49Ste6HaoPow?e=ZdAs6z) |
VCTree    | ResNet-50 | 12 | 20.6 / 9.70 | 22.1 / 10.2 | 22.5 / 10.2 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EhKfi9kqAd9CnSoHztQIChABeBjBD3hF7DflrNCjlHfh9A?e=lWa1bd) |
GPSNet    | ResNet-50 | 12 | 17.8 / 7.03 | 19.6 / 7.49 | 20.1 / 7.67 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EipIhZgVgx1LuK2RUmjRg2sB8JqxMIS5GnPDHeaYy5GF6A?e=5j53VF) |
PSGTR     | ResNet-50 | 60 | 28.4 / 16.6 | 34.4 / 20.8 | 36.3 / 22.1 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/Eonc-KwOxg9EmdtGDX6ss-gB35QpKDnN_1KSWOj6U8sZwQ?e=zdqwqP) |
PSGFormer | ResNet-50 | 60 | 18.0 / 14.8 | 19.6 / 17.0 | 20.1 / 17.6 |  [link](https://entuedu-my.sharepoint.com/:f:/g/personal/jingkang001_e_ntu_edu_sg/EnaJchJzJPtGrkl4k09evPIB5JUkkDZ2tSS9F-Hd-1KYzA?e=9QA8Nc) |



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
