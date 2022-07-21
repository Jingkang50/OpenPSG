_base_ = [
    '../_base_/models/psgtr_r101.py', '../_base_/datasets/psg.py',
    '../_base_/custom_runtime.py'
]

custom_imports = dict(imports=[
    'openpsg.models.frameworks.psgtr', 'openpsg.models.losses.seg_losses',
    'openpsg.models.relation_heads.psgtr_head', 'openpsg.datasets',
    'openpsg.datasets.pipelines.loading',
    'openpsg.datasets.pipelines.rel_randomcrop',
    'openpsg.models.relation_heads.approaches.matcher', 'openpsg.utils'
],
                      allow_failed_imports=False)

dataset_type = 'PanopticSceneGraphDataset'

# HACK:
object_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit',
    'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
    'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea',
    'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone',
    'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other',
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged',
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged',
    'food-other-merged', 'building-other-merged', 'rock-merged',
    'wall-other-merged', 'rug-merged'
]

predicate_classes = [
    'over',
    'in front of',
    'beside',
    'on',
    'in',
    'attached to',
    'hanging from',
    'on back of',
    'falling off',
    'going down',
    'painted on',
    'walking on',
    'running on',
    'crossing',
    'standing on',
    'lying on',
    'sitting on',
    'flying over',
    'jumping over',
    'jumping from',
    'wearing',
    'holding',
    'carrying',
    'looking at',
    'guiding',
    'kissing',
    'eating',
    'drinking',
    'feeding',
    'biting',
    'catching',
    'picking',
    'playing with',
    'chasing',
    'climbing',
    'cleaning',
    'playing',
    'touching',
    'pushing',
    'pulling',
    'opening',
    'cooking',
    'talking to',
    'throwing',
    'slicing',
    'driving',
    'riding',
    'parked on',
    'driving on',
    'about to hit',
    'kicking',
    'swinging',
    'entering',
    'exiting',
    'enclosing',
    'leaning on',
]

model = dict(bbox_head=dict(
    num_classes=len(object_classes),
    num_relations=len(predicate_classes),
    object_classes=object_classes,
    predicate_classes=predicate_classes,
    use_mask=True,
    num_query=100,
), )

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadPanopticSceneGraphAnnotations',
         with_bbox=True,
         with_rel=True,
         with_mask=True,
         with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(type='Resize',
                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                (576, 1333), (608, 1333), (640, 1333),
                                (672, 1333), (704, 1333), (736, 1333),
                                (768, 1333), (800, 1333)],
                     multiscale_mode='value',
                     keep_ratio=True)
            ],
            [
                dict(type='Resize',
                     img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                     multiscale_mode='value',
                     keep_ratio=True),
                dict(type='RelRandomCrop',
                     crop_type='absolute_range',
                     crop_size=(384, 600),
                     allow_negative_crop=False),  # no empty relations
                dict(type='Resize',
                     img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                (576, 1333), (608, 1333), (640, 1333),
                                (672, 1333), (704, 1333), (736, 1333),
                                (768, 1333), (800, 1333)],
                     multiscale_mode='value',
                     override=True,
                     keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='RelsFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels', 'gt_masks'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            # dict(type='ToDataContainer', fields=(dict(key='gt_bboxes'), dict(key='gt_labels'))),
            dict(type='Collect', keys=['img']),
        ])
]

evaluation = dict(
    interval=1,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg',
)

data = dict(samples_per_gpu=1,
            workers_per_gpu=2,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    }))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(policy='step', step=40)
runner = dict(type='EpochBasedRunner', max_epochs=60)

project_name = 'psgtr'
expt_name = 'psgtr_r101_psg'
work_dir = f'./work_dirs/{expt_name}'
checkpoint_config = dict(interval=2, max_keep_ckpts=10)

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
        )
    ],
)

load_from = 'work_dirs/checkpoints/detr_pan_r101.pth'
