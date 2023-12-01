from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv import Config
import mmcv
import os.path as osp
import numpy as np
import cv2



# data_root =''
# img_dir = ''
# ann_dir = ''
# split_dir = ''
checkpoint_file = r'D:\111_Su_Kevin\DIP_final\latest.pth'
# define class and plaette for better visualization
classes = ('background', 'powder_uncover', 'powder_uneven', 'scratch')
palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
# ---------------------------------------------------------------------------------
@DATASETS.register_module()
class StanfordBackgroundDataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None
# ---------------------------------------------------------------------------------
cfg = Config.fromfile('mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 4
cfg.model.auxiliary_head.num_classes = 4

# Modify dataset type and path
cfg.dataset_type = 'StanfordBackgroundDataset'
# cfg.data_root = data_root

cfg.data.samples_per_gpu = 16
cfg.data.workers_per_gpu = 0

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (256, 256)
# cfg.train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations'),
#     dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **cfg.img_norm_cfg),
#     dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 240),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# cfg.data.train.type = cfg.dataset_type
# cfg.data.train.data_root = cfg.data_root
# cfg.data.train.img_dir = img_dir
# cfg.data.train.ann_dir = ann_dir
# cfg.data.train.pipeline = cfg.train_pipeline
# cfg.data.train.split = 'splits/train.txt'

# cfg.data.val.type = cfg.dataset_type
# cfg.data.val.data_root = cfg.data_root
# cfg.data.val.img_dir = img_dir
# cfg.data.val.ann_dir = ann_dir
# cfg.data.val.pipeline = cfg.test_pipeline
# cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
# cfg.data.test.data_root = cfg.data_root
# cfg.data.test.img_dir = img_dir
# cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
# cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
# cfg.load_from = checkpoint_file

# # Set up working dir to save files and logs.
# cfg.work_dir = './work_dirs/0107_e500'

# cfg.runner.max_iters = 2000
# cfg.log_config.interval = 10
# cfg.evaluation.interval = 500
# cfg.checkpoint_config.interval = 250

# # Set seed to facitate reproducing the result
# cfg.seed = 0
# set_random_seed(0, deterministic=False)
# cfg.gpu_ids = range(1)
# cfg.device = 'cpu'

# ---------------------------------------------------------------------

# Build the detector
# model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
# model.CLASSES = classes
# LABELREAD = 
IMGREAD = r'D:\111_Su_Kevin\class_data\Val\powder_uneven\image\converted_ 0212.png'
IMGSAVE = ''
img = mmcv.imread(IMGREAD)
# model.cfg = cfg

model = init_segmentor(cfg, checkpoint_file, device='cpu')
model.PALETTE = palette

model.eval()
result = inference_segmentor(model, img)[0]

h, w = result.shape
simg = np.zeros((h, w, 3), dtype=np.uint8)
for a in range(h):
    for b in range(w):
        if result[a, b] == 1:
            simg[a, b, 0] = 255
        elif result[a, b] == 2:
            simg[a, b, 1] = 255
        elif result[a, b] == 3:
            simg[a, b, 2] = 255

cv2.imwrite('new.png', simg[:, :, ::-1].astype(np.uint8))

# show_result_pyplot(model, img, result, palette)
