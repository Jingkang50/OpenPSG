_base_ = [
    '../motifs/panoptic_fpn_r50_fpn_1x_predcls_psg.py',
]

model = dict(relation_head=dict(
    type='VCTreeHead',
    head_config=dict(
        # NOTE: Evaluation type
        use_gt_box=True,
        use_gt_label=True,
    ),
))

evaluation = dict(interval=1,
                  metric='predcls',
                  relation_mode=True,
                  classwise=True)

# Change batch size and learning rate
data = dict(samples_per_gpu=16,
            workers_per_gpu=0)  # FIXME: Is this the problem?
# optimizer = dict(lr=0.001)

# Log config
project_name = 'openpsg'
expt_name = 'vctree_panoptic_fpn_r50_fpn_1x_predcls_psg'
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
