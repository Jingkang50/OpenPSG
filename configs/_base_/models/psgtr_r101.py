_base_ = './psgtr_r50.py'

model = dict(backbone=dict(
    depth=101,
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))
