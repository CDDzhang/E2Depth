import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.misc import get_config
from .base_dataset import construct_dataset

# 'configs/ddad_surround_fusion_vfdepth.yaml'
def get_train_dataset(cfg):
    height = cfg["training"]["height"]
    width = cfg["training"]["width"]

    _augmentation = {
        'image_shape': (int(height), int(width)),  #裁剪
        'jittering': (0.2, 0.2, 0.2, 0.05), # 颜色扰动（闪烁）
        'crop_train_borders': (), # 从边界裁剪掉固定的像素
        'crop_eval_borders': ()  # 
    }

    # construct validation dataset
    train_dataset = construct_dataset(cfg, 'train', **_augmentation)  # len 12350

    dataloader_opts = {
        'batch_size': cfg["training"]["batch_size"],
        'shuffle': True,
        'num_workers': cfg["training"]["num_workers"],
        'pin_memory': True,
        'drop_last': True
    }

    train_dataloader = DataLoader(train_dataset, **dataloader_opts)
    return train_dataset, train_dataloader

def get_val_dataset(cfg):
    height = cfg["training"]["height"]
    width = cfg["training"]["width"]

    _augmentation = {
            'image_shape': (int(height), int(width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
    }

    # construct validation dataset
    val_dataset = construct_dataset(cfg, 'val', **_augmentation)  # len 3036

    dataloader_opts = {
        'batch_size': cfg["training"]["batch_size"],
        'shuffle': False,
        'num_workers': 0,
        'pin_memory': True,
        'drop_last': True
    }

    val_dataloader = DataLoader(val_dataset, **dataloader_opts)

    return val_dataset, val_dataloader

def get_eval_dataset(cfg, dataset='ddad'):
    # Image resizing for the validation data
    height = cfg["training"]["height"]
    width = cfg["training"]["width"]

    _augmentation = {
        'image_shape': (int(height), int(width)),
        'jittering': (0.0, 0.0, 0.0, 0.0),
        'crop_train_borders': (),
        'crop_eval_borders': ()
    }
    # construct validation dataset
    eval_dataset = construct_dataset(cfg, 'val', **_augmentation)
    print('loaded ddad eval dataset')
    dataloader_opts = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'drop_last': True
    }

    dataloader = DataLoader(eval_dataset, **dataloader_opts)
    return eval_dataset, dataloader