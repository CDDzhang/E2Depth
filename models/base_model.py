# Copyright (c) 2023 42dot. All rights reserved.
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import construct_dataset


_OPTIMIZER_NAME ='adam'


class BaseModel:
    def __init__(self, cfg):
        self._dataloaders = {}
        self.mode = None
        self.models = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ddp_enable = False

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0:
                self.set_val_dataloader(cfg)
                
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)

    def set_train_dataloader(self, cfg, rank):                 
        # jittering augmentation and image resizing for the training data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),  #裁剪
            'jittering': (0.2, 0.2, 0.2, 0.05), # 颜色扰动（闪烁）
            'crop_train_borders': (), # 从边界裁剪掉固定的像素
            'crop_eval_borders': ()  # 
        }

        # construct train dataset
        train_dataset = construct_dataset(cfg, 'train', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        if self.ddp_enable:
            dataloader_opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, 
                num_replicas = self.world_size,
                rank=rank, 
                shuffle=True
            ) 
            dataloader_opts['sampler'] = self.train_sampler

        self._dataloaders['train'] = DataLoader(train_dataset, **dataloader_opts)
        num_train_samples = len(train_dataset)    
        self.num_total_steps = num_train_samples // (self.batch_size * self.world_size) * self.num_epochs

    def set_val_dataloader(self, cfg):         
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        val_dataset = construct_dataset(cfg, 'val', **_augmentation)

        dataloader_opts = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['val']  = DataLoader(val_dataset, **dataloader_opts)
    
    def set_eval_dataloader(self, cfg):  
        # Image resizing for the validation data
        _augmentation = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.0, 0.0, 0.0, 0.0),
            'crop_train_borders': (),
            'crop_eval_borders': ()
        }

        # construct validation dataset
        eval_dataset = construct_dataset(cfg, 'val', **_augmentation)
        
        dataloader_opts = {
            'batch_size': self.eval_batch_size,
            'shuffle': False,
            'num_workers': self.eval_num_workers,
            'pin_memory': True,
            'drop_last': True
        }

        self._dataloaders['eval'] = DataLoader(eval_dataset, **dataloader_opts)

    def set_optimizer(self):
        parameters_to_train = []
        for v in self.models.values():
            parameters_to_train += list(v.parameters())

        self.optimizer = optim.Adam(
        parameters_to_train, 
            self.learning_rate
        )

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )       
  
    def train_dataloader(self):
        return self._dataloaders['train']

    def val_dataloader(self):
        return self._dataloaders['val']

    def eval_dataloader(self):
        return self._dataloaders['eval']
    
    def set_train(self):
        self.mode = 'train'
        for m in self.models.values():
            m.train()

    def set_val(self):
        self.mode = 'val'
        for m in self.models.values():
            m.eval()

    def save_model(self, epoch):
        curr_model_weights_dir = os.path.join(self.save_weights_root, f'weights_{epoch}')
        os.makedirs(curr_model_weights_dir, exist_ok=True)

        for model_name, model in self.models.items():
            model_file_path = os.path.join(curr_model_weights_dir, f'{model_name}.pth')
            to_save = model.state_dict()
            torch.save(to_save, model_file_path)
        
        # save optimizer
        optim_file_path = os.path.join(curr_model_weights_dir, f'{_OPTIMIZER_NAME}.pth')
        torch.save(self.optimizer.state_dict(), optim_file_path)

    def load_weights(self):
        if not self.pretrain:
            return
        assert os.path.isdir(self.load_weights_dir), f'\tCannot find {self.load_weights_dir}'
        print(f'Loading a model from {self.load_weights_dir}')
        
        # to retrain
        if self.pretrain and self.ddp_enable:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % (self.world_size-1)}
            
        for n in self.models_to_load:
            print(f'Loading {n} weights...')
            path = os.path.join(self.load_weights_dir, f'{n}.pth')
            model_dict = self.models[n].state_dict()
            
            # distribute gpus for ddp retraining
            if self.pretrain and self.ddp_enable:
                pre_trained_dict = torch.load(path, map_location=map_location)
            else: 
                pre_trained_dict = torch.load(path)
                
            # load parameters
            pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
            model_dict.update(pre_trained_dict)
            self.models[n].load_state_dict(model_dict)

        if self.mode == 'train':
            # loading adam state
            optim_file_path = os.path.join(self.load_weights_dir, f'{_OPTIMIZER_NAME}.pth')
            if os.path.isfile(optim_file_path):
                try:
                    print(f'Loading {_OPTIMIZER_NAME} weights')
                    optimizer_dict = torch.load(optim_file_path)
                    self.optimizer.load_state_dict(optimizer_dict)
                except ValueError:
                    print(f'\tCannnot load {_OPTIMIZER_NAME} - the optimizer will be randomly initialized')
            else:
                print(f'\tCannot find {_OPTIMIZER_NAME} weights, so the optimizer will be randomly initialized')