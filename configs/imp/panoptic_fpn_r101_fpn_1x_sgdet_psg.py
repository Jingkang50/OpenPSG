_base_ = './panoptic_fpn_r50_fpn_1x_sgdet_psg.py'

model = dict(backbone=dict(
    depth=101,
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

# Log config
project_name = 'openpsg'
expt_name = 'imp_panoptic_fpn_r101_fpn_1x_sgdet_psg'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=project_name,
                name=expt_name,
            ),
        ),
    ],
)

load_from = 'work_dirs/checkpoints/panoptic_fpn_r101_fpn_1x_coco_20210820_193950-ab9157a2.pth'
