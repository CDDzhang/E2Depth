"""
E2Depth Evaluation & Visualization Script
==========================================
 
Usage:
    python eval.py --mode metric --config configs/eval/ddad_e2depth_eval.yaml
"""

import argparse 
import os
import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

from utils.misc import get_config
from trainers.base_evaluator import Evaluator
from models.e2depth import E2Depth, E2DepthModel
from datasets import get_eval_dataset, get_train_dataset

import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

VIEW_TITLES = ["FRONT", "FRONT-LEFT", "FRONT-RIGHT", "BACK-LEFT", "BACK-RIGHT", "BACK"]
_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

def to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).unsqueeze(0)
    if isinstance(x, torch.Tensor):
        return x.unsqueeze(0)
    return x

def load_sample(cfg, index, split='val'):
    if split == 'train':
        dataset, _ = get_train_dataset(cfg=cfg)
    else:
        dataset, _ = get_eval_dataset(cfg=cfg)
    sample = dataset[index]
    sample = {k: to_tensor(v) for k, v in sample.items()}
    return sample

def run_metrics(cfg):
    model = E2DepthModel(cfg, rank=0)
    evaluator = Evaluator(cfg, use_tb=False)
    evaluator.eval(model)

def run_inference(cfg, index, savedir):
    os.makedirs(savedir, exist_ok=True)
    model = E2Depth(cfg)
    model.load_weights()
    model.set_val()

    sample = load_sample(cfg, index, split='val')
    num_cams = 0

    with torch.no_grad():
        outputs = model.inference(sample)

    depths = [outputs[('cam', i)][('depth', 0)] for i in range(6)]
    print(f'Saved prediction figure (save_predict) to {savedir}/{index}.png')

def main(args):
    # load config
    cur_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(cur_path, f'configs/eval/{args.config}')

    cfg_mode = 'train' if args.mode == 'metrics' else 'eval'
    cfg = get_config(config_file, mode=cfg_mode)
    torch.cuda.empty_cache()

    if args.mode == 'metrics':
        run_metrics(cfg)
    elif args.mode == 'inference':
        run_inference(cfg, args.index, args.savedir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="metrics",
                        choices=['metrics', 'inference'],
                        help='metrics | inference')
    parser.add_argument('--config', type=str, default="ddad_e2depth_eval.yaml",
                        help='config file path (yaml)')
    parser.add_argument('--index', type=int, default=0,
                        help='dataset index')
    parser.add_argument('--cam', type=str, default='all',
                        help='camera ID(s). Single: "0", Multi: "0,1,2", All: "all"')
    parser.add_argument('--savedir', type=str, default='./outputs',
                        help='output dir')
    args = parser.parse_args()

    main(args)