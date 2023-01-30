_base_ = [
    './psgformer_r50_psg.py'
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(type='MultiScaleFlipAug',
         img_scale=(1333, 800),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=1),
             dict(type='ImageToTensor', keys=['img']),
            #  dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            #  dict(type='ToDataContainer',
                #   fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
             dict(type='Collect', keys=['img']),
         ])
]

data = dict(
    test=dict(
        pipeline=pipeline,
    ),
)
