# dataset settings
dataset_type = 'SceneGraphDataset'
ann_file = '/mnt/ssd/gzj/data/VisualGenome/data_openpsg.json'
img_dir = '/mnt/ssd/gzj/data/VisualGenome/VG_100K'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SceneGraphFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels', 'gt_relmaps']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Since the forward process may need gt info, annos must be loaded.
    dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # NOTE: Do not change the img to DC.
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            dict(type='ToDataContainer',
                 fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
data = dict(samples_per_gpu=2,
            workers_per_gpu=2,
            train=dict(type=dataset_type,
                       ann_file=ann_file,
                       img_prefix=img_dir,
                       pipeline=train_pipeline,
                       split='train'),
            val=dict(type=dataset_type,
                     ann_file=ann_file,
                     img_prefix=img_dir,
                     pipeline=test_pipeline,
                     split='test'),
            test=dict(type=dataset_type,
                      ann_file=ann_file,
                      img_prefix=img_dir,
                      pipeline=test_pipeline,
                      split='test'))
