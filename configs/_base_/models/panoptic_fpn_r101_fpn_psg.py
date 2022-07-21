_base_ = './panoptic_fpn_r50_fpn_psg.py'

model = dict(backbone=dict(
    depth=101,
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

expt_name = 'panoptic_fpn_r101_fpn_psg'
load_from = 'work_dirs/checkpoints/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth'
