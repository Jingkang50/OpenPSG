_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/psg.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/custom_runtime.py',
]

find_unused_parameters = True
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

model = dict(
    type='SceneGraphPanopticFPN',
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=None,
        loss_seg=dict(type='CrossEntropyLoss',
                      ignore_index=255,
                      loss_weight=0.5),
    ),
    panoptic_fusion_head=dict(type='HeuristicFusionHead',
                              num_things_classes=80,
                              num_stuff_classes=53),
    test_cfg=dict(panoptic=dict(
        score_thr=0.6,
        max_per_img=100,
        mask_thr_binary=0.5,
        mask_overlap=0.5,
        nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
        stuff_area_limit=4096,
    )),
    relation_head=dict(
        type='MotifHead',
        object_classes=object_classes,
        predicate_classes=predicate_classes,
        num_classes=len(object_classes) + 1,  # with background class
        num_predicates=len(predicate_classes) + 1,
        use_bias=False,  # NOTE: whether to use frequency bias
        head_config=dict(
            # NOTE: Evaluation type
            use_gt_box=True,
            use_gt_label=True,
            use_vision=True,
            embed_dim=200,
            hidden_dim=512,
            roi_dim=1024,
            context_pooling_dim=4096,
            dropout_rate=0.2,
            context_object_layer=1,
            context_edge_layer=1,
            glove_dir='data/glove/',
            causal_effect_analysis=False,
        ),
        bbox_roi_extractor=dict(
            type='VisualSpatialExtractor',
            bbox_roi_layer=dict(type='RoIAlign',
                                output_size=7,
                                sampling_ratio=2),
            with_visual_bbox=True,
            with_visual_mask=False,
            with_visual_point=False,
            with_spatial=False,
            in_channels=256,
            fc_out_channels=1024,
            featmap_strides=[4, 8, 16, 32],
        ),
        relation_roi_extractor=dict(
            type='VisualSpatialExtractor',
            bbox_roi_layer=dict(type='RoIAlign',
                                output_size=7,
                                sampling_ratio=2),
            with_visual_bbox=True,
            with_visual_mask=False,
            with_visual_point=False,
            with_spatial=True,
            separate_spatial=False,
            in_channels=256,
            fc_out_channels=1024,
            featmap_strides=[4, 8, 16, 32],
        ),
        relation_sampler=dict(
            type='Motif',
            pos_iou_thr=0.5,
            require_overlap=False,  # for sgdet training, not require
            num_sample_per_gt_rel=4,
            num_rel_per_image=1024,
            pos_fraction=0.25,
            # NOTE: To only include overlapping bboxes?
            test_overlap=False,  # for testing
        ),
        loss_object=dict(type='CrossEntropyLoss',
                         use_sigmoid=False,
                         loss_weight=1.0),
        loss_relation=dict(type='CrossEntropyLoss',
                           use_sigmoid=False,
                           loss_weight=1.0),
    ),
)

custom_hooks = []

# To freeze modules
freeze_modules = [
    'backbone',
    'neck',
    'rpn_head',
    'roi_head',
    'semantic_head',
    'panoptic_fusion_head',
]

evaluation = dict(interval=1,
                  metric='predcls',
                  relation_mode=True,
                  classwise=True)

# Change batch size and learning rate
data = dict(samples_per_gpu=16, )
# optimizer = dict(lr=0.003)
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[7, 10])

# Log config
project_name = 'openpsg'
expt_name = 'motifs_panoptic_fpn_r50_fpn_1x_predcls_psg'
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

load_from = 'work_dirs/checkpoints/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth'
