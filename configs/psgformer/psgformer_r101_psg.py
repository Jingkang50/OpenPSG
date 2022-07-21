_base_ = './psgformer_r50_psg.py'

model = dict(backbone=dict(
    depth=101,
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')))

# learning policy
lr_config = dict(policy='step', step=48)
runner = dict(type='EpochBasedRunner', max_epochs=60)

project_name = 'psgformer'
expt_name = 'psgformer_r101_psg'
work_dir = f'./work_dirs/{expt_name}'
checkpoint_config = dict(interval=12, max_keep_ckpts=10)

load_from = './work_dirs/checkpoints/detr4psgformer_r101.pth'
