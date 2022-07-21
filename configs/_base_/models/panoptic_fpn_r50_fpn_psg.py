_base_ = [
    '../models/mask_rcnn_r50_fpn.py',
    '../datasets/psg_panoptic.py',
    '../schedules/schedule_1x.py',
    '../custom_runtime.py',
]

model = dict(
    type='PanopticFPN',
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
)

custom_hooks = []

# Change batch size and learning rate
data = dict(samples_per_gpu=8,
            # workers_per_gpu=2
            )
# optimizer = dict(lr=0.02)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(policy='step',
                 warmup='linear',
                 warmup_iters=500,
                 warmup_ratio=1.0 / 3,
                 step=[8, 11])

project_name = 'openpsg'
expt_name = 'panoptic_fpn_r50_fpn_psg'
work_dir = f'./work_dirs/{expt_name}'

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=project_name,
                name=expt_name,
                # config=work_dir + "/cfg.yaml"
            ),
        ),
    ],
)

load_from = 'work_dirs/checkpoints/panoptic_fpn_r50_fpn_1x_coco_20210821_101153-9668fd13.pth'
