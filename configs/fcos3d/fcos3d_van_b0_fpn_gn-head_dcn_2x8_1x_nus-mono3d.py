_base_ = [
    '../_base_/datasets/nus-mono3d.py', '../_base_/models/fcos3d.py',
    '../_base_/schedules/mmdet_schedule_1x.py', '../_base_/default_runtime.py'
]

dims = [32, 64, 160, 256]
pretrained = dict(type='Pretrained', checkpoint='/home/nils/VAN-Detection/models/van_tiny_754.pth.tar')
norm_cfg = dict(type='SyncBN', requires_grad=not bool(pretrained))  # train BN only when not using pretrained model

# dims = [64, 128, 320, 512]
# pretrained = dict(type='Pretrained', checkpoint='/home/nils/VAN-Detection/models/van-base_8xb128_in1k_20220501-conv.pth')
# norm_cfg = dict(type='SyncBN', requires_grad=not bool(pretrained))  # train BN only when not using pretrained model

# model settings
model = dict(
    backbone=dict(
        type='VAN',
        _delete_=True,
        arch='b0', 
        drop_path_rate=0.1, 
        init_cfg=pretrained, 
        out_indices=[0,1,2,3],
    ),
    # backbone=dict(
    #     type='VAN',
    #     embed_dims=dims,
    #     drop_rate=0.0,
    #     drop_path_rate=0.1,
    #     depths=[3, 3, 5, 2],
    #     norm_cfg=norm_cfg,
    #     init_cfg=pretrained,
    #   )
    neck=dict(
        in_channels=dims,
    ),
)

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
optimizer = dict(
    lr=0.002, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
evaluation = dict(interval=2)
checkpoint_config = dict(interval=2, max_keep_ckpts=3)
workflow=[('train', 4), ('val', 1)]


log_config = dict(
    interval=10,
    hooks=[
        dict(type='MMDetWandbHook',
          by_epoch=True,
          init_kwargs=dict(
            entity="nkoch-aitastic",
            project='van-detection3d',
            tags=[
              'backbone:VAN-B0',
              'neck:FPN',
              'head:FCOS3D',
              'pretrained',
              ]
          ),
          log_checkpoint=True,
          log_checkpoint_metadata=True,
          num_eval_images=100,
        ), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
        dict(type='TextLoggerHook', by_epoch=True),
    ])
