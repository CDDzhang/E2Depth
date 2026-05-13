"""
base_pose_model.py
"""
 
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
 
from datasets import construct_dataset


class PoseLoss(nn.Module):
    def __init__(self, w_rot=1.0, w_trans=1.0):
        super().__init__()
        self.w_rot = w_rot
        self.w_trans = w_trans
    
    def rotation_loss(self, T_pred, T_gt):
        R_diff = T_pred[:, :3, :3] @ T_gt[:, :3, :3].transpose(-1, -2)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_angle = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
        return torch.acos(cos_angle).mean()
 
    def translation_loss(self, T_pred, T_gt):
        return (T_pred[:, :3, 3] - T_gt[:, :3, 3]).abs().mean()
    
    def forward(self, pred_cam0, gt_cam0):
        rot_loss = 0.0
        trans_loss = 0.0
        count = 0
 
        for fid_key, T_pred in pred_cam0.items():
            T_gt = gt_cam0[fid_key].to(T_pred.device)
            rot_loss += self.rotation_loss(T_pred, T_gt)
            trans_loss += self.translation_loss(T_pred, T_gt)
            count += 1
 
        rot_loss /= max(count, 1)
        trans_loss /= max(count, 1)
 
        total = self.w_rot * rot_loss + self.w_trans * trans_loss
        return {
            'total_loss': total,
            'rot_loss': rot_loss.detach() if isinstance(rot_loss, torch.Tensor) else torch.tensor(rot_loss),
            'trans_loss': trans_loss.detach() if isinstance(trans_loss, torch.Tensor) else torch.tensor(trans_loss),
        }
    

class PoseModelBase:
    def __init__(self, cfg, rank):
        self.rank = rank
        self.mode = None
        self.models = {}
        self.optimizer = None
        self.lr_scheduler = None
        self.ddp_enable = False
        self._dataloaders = {}

        self.read_config(cfg)
        self.pose_loss = PoseLoss(
            w_rot=cfg.get('pose', {}).get('w_rot', 1.0),
            w_trans=cfg.get('pose', {}).get('w_trans', 1.0),
        )
        self.frame_ids = cfg['training']['frame_ids']

    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Pose Datasets')
        aug_train = {
            'image_shape': (int(self.height), int(self.width)),
            'jittering': (0.2, 0.2, 0.2, 0.05),
            'crop_train_borders': (),
            'crop_eval_borders': (),
        }
 
        ds = construct_dataset(cfg, 'pose', **aug_train)
        opts = dict(batch_size=self.batch_size, shuffle=True,
                    num_workers=self.num_workers, pin_memory=True, drop_last=True)
        if self.ddp_enable:
            opts['shuffle'] = False
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                ds, num_replicas=self.world_size, rank=rank, shuffle=True)
            opts['sampler'] = self.train_sampler
        self._dataloaders['train'] = DataLoader(ds, **opts)

        if rank == 0:
            # val_ds = construct_dataset(cfg, 'val', **aug_eval)
            indices = list(range(min(1600, len(ds))))
            val_ds = torch.utils.data.Subset(ds, indices)
            self._dataloaders['val'] = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=0, pin_memory=True, drop_last=True)
 
    def train_dataloader(self):
        return self._dataloaders['train']
 
    def val_dataloader(self):
        return self._dataloaders['val']

    def set_optimizer(self):
        params = []
        for v in self.models.values():
            params += list(v.parameters())
        self.optimizer = optim.Adam(params, self.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, self.scheduler_step_size, 0.1)
 
    def set_train(self):
        for m in self.models.values():
            m.train()
 
    def set_val(self):
        for m in self.models.values():
            m.eval()
 
    def save_model(self, epoch):
        d = os.path.join(self.save_weights_root, f'pose_weights_{epoch}')
        os.makedirs(d, exist_ok=True)
        for name, m in self.models.items():
            torch.save(m.state_dict(), os.path.join(d, f'{name}.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(d, 'adam.pth'))
        print(f'  Saved → {d}')

    def load_weights(self):
        if not getattr(self, 'pretrain', False):
            return
        d = self.load_weights_dir
        assert os.path.isdir(d), f'Cannot find {d}'
        print(f'Loading from {d}')
        for n in getattr(self, 'models_to_load', self.models.keys()):
            path = os.path.join(d, f'{n}.pth')
            if os.path.isfile(path):
                state = torch.load(path, map_location=f'cuda:{self.rank}')
                self.models[n].load_state_dict(state, strict=False)
                print(f'  Loaded {n}')

    def process_batch(self, inputs, rank):
        raise NotImplementedError
    
    @staticmethod
    def _to_device(inputs, rank):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(rank, non_blocking=True)
        return inputs
    

    





    
    