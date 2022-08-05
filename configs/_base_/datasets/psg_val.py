# dataset settings
dataset_type = 'PanopticSceneGraphDataset'
ann_file = 'data/psg/psg.json'
coco_root = 'data/coco'

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # Since the forward process may need gt info, annos must be loaded.
    dict(type='LoadPanopticSceneGraphAnnotations',
         with_bbox=True,
         with_rel=True),
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
            dict(
                type='ToDataContainer',
                fields=(dict(key='gt_bboxes'), dict(key='gt_labels')),
            ),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ],
    ),
]
data = dict(
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        img_prefix=coco_root,
        seg_prefix=coco_root,
        pipeline=test_pipeline,
        split='test',
        all_bboxes=True,
    ),
)

evaluation1 = dict(metric=['sgdet'],
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')

evaluation2 = dict(metric=['PQ'],
                  relation_mode=True,
                  classwise=True,
                  iou_thrs=0.5,
                  detection_method='pan_seg')
