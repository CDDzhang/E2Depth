"""
Training VFDepth model(without posenet part) on DDAD dataset
"""

import argparse 
import os

import torch
torch.backends.cudnn.deterministic = False  
torch.backends.cudnn.benchmark = True 
torch.backends.cuda.matmul.allow_tf32 = True

from utils.misc import get_config
from trainers.base_trainer import BaseTrainer
from models.e2depth import E2DepthModel
 
if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_path, 'configs/train/ddad_e2depth_train.yaml')
    cfg = get_config(config_file, mode='train')
    torch.cuda.empty_cache()
    device = "cuda: 0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = E2DepthModel(cfg, rank=0)
    trainer = BaseTrainer(cfg, rank=0, use_tb=True)
    trainer.train(model)
    