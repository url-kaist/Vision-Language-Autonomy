_base_ = ('../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# Basic hyperparameters
num_classes = 1203
num_training_classes = 80
text_channels = 512
img_scale = (1280, 1280)
text_model_name = 'openai/clip-vit-base-patch32'

neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

# Model
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    test_cfg=dict(score_thr=0.005, nms=dict(type='nms', iou_threshold=0.65))
)

# Disable training/eval components for demo
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
test_dataloader = dict(
    dataset=dict(
        type='YOLOv5LVISV1Dataset',
        data_root='unused/',
        ann_file='unused.json',
        data_prefix=dict(img='unused/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromNDArray'),
            dict(type='YOLOv5KeepRatioResize', scale=(1280, 1280)),
            dict(type='LetterResize',
                 scale=(1280, 1280),
                 allow_scale_up=False,
                 pad_val=dict(img=114)),
            dict(type='LoadText'),
            dict(type='mmdet.PackDetInputs',
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'pad_param', 'texts'))
        ])
)
train_cfg = None
optim_wrapper = None
default_hooks = None
custom_hooks = None

