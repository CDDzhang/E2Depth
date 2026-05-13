import time 
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist

from utils.logger import Logger


class Evaluator:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank=0, use_tb=True):
        self.read_config(cfg)
        self.rank = rank
        self.logger = Logger(cfg, use_tb=use_tb)
        # ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']
        self.depth_metric_names = self.logger.get_metric_names()  

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    @torch.no_grad()
    def eval(self, model):
        eval_dataloader = model._dataloaders['eval']
        
        # load model
        model.load_weights()
        model.set_val()

        avg_depth_eval_metric = defaultdict(float)
        avg_depth_eval_median = defaultdict(float)

        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):   
            outputs = model.eval_inference(inputs)
            depth_eval_metric, depth_eval_median, cam_abs_rel = self.logger.compute_depth_losses(inputs, outputs)

            for key in self.depth_metric_names:
                avg_depth_eval_metric[key] += depth_eval_metric[key]
                avg_depth_eval_median[key] += depth_eval_median[key]
            
        for key in self.depth_metric_names:
            avg_depth_eval_metric[key] /= len(eval_dataloader)
            avg_depth_eval_median[key] /= len(eval_dataloader)
            
        print('Evaluation result...\n')
        self.logger.print_perf(avg_depth_eval_metric, 'metric')
        self.logger.print_perf(avg_depth_eval_median, 'median')
        for cam_id, abs_rel in cam_abs_rel.items():
            print(f"          | cam {cam_id} abs_rel = {abs_rel:.3f}")
