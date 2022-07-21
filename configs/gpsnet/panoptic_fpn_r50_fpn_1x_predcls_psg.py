_base_ = [
    '../motifs/panoptic_fpn_r50_fpn_1x_predcls_psg.py',
]

model = dict(relation_head=dict(
    type='GPSHead',
    head_config=dict(
        # NOTE: Evaluation type
        use_gt_box=True,
        use_gt_label=True,
    ),
))

evaluation = dict(interval=1,
                  metric='predcls',
                  relation_mode=True,
                  classwise=True,
                  detection_method='pan_seg')

# Change batch size and learning rate
data = dict(samples_per_gpu=16, workers_per_gpu=0)
optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)

# Log config
project_name = 'openpsg'
expt_name = 'gpsnet_panoptic_fpn_r50_fpn_1x_predcls_psg'
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
