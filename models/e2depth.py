# Copyright (c) 2023 42dot. All rights reserved.
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim  

from torch.utils.data import DataLoader

from datasets.base_dataset import construct_dataset
from models.base_model import BaseModel
from geometry import Pose, ViewRendering
from losses import DepthSynLoss, MultiCamLoss, SingleCamLoss, DepthCamLoss, GeoMultiCamLoss
from networks.e2block.dino_depthnet import DinoDepthNet

from networks import PriorPoseNet, MonoPoseNet, ConcatPoseNet
from networks.blocks import pack_cam_feat, unpack_cam_feat

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']


class E2Depth(BaseModel):
    """Model only for inference"""
    def __init__(self, cfg):
        super(E2Depth, self).__init__(cfg)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.read_config(cfg)

        self.models = self.prepare_model(cfg)          
        if self.pretrain: 
            self.load_weights()

    def prepare_model(self, cfg):
        models = {}      
        models['depth_net'] = DinoDepthNet(cfg=cfg).to(self.device)
        return models
    
    def inference(self, inputs):
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(0)  
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])

        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}           
        depth_feats = self.predict_depth(inputs)
        for cam in range(self.num_cams):                  
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
        self.compute_depth_maps(inputs, outputs)
        return outputs
    
    def inference_with_layer_rn_features(self, inputs):
        """推理并返回 decoder 中 layer_*_rn 特征（refinenet 输入前）"""
        outputs = self.inference(inputs)
        decoder = self.models['depth_net'].decoder
        layer_rn_cache = decoder._layer_rn_cache  # dict of 4 tensors [B*K, 128, H, W]
        return outputs, layer_rn_cache
    
    def predict_depth(self, inputs):          
        net = self.models['depth_net']
        return net(inputs)
    
    def compute_depth_maps(self, inputs, outputs):                   
        for cam in range(self.num_cams):
            for scale in self.scales: # [0]
                disp = outputs[('cam', cam)][('disp', scale)]
                outputs[('cam', cam)][('depth', scale)] = self.disp_to_depth(disp)
                if self.aug_depth:
                    disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                    outputs[('cam', cam)][('depth', scale, 'aug')] = self.disp_to_depth(disp) 
    
    def disp_to_depth(self, disp):
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp = F.interpolate(disp, [self.height, self.width], mode='bilinear', align_corners=False)
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        return 1 / scaled_disp

    def inference_with_features(self, inputs):
        outputs = self.inference(inputs)
        decoder = self.models['depth_net'].decoder
        wavelet_vis = decoder.collect_wavelet_features()
        return outputs, wavelet_vis


class E2DepthModel(BaseModel):
    def __init__(self, cfg, rank):
        super(E2DepthModel, self).__init__(cfg)
        self.rank = rank
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.read_config(cfg)
        self.prepare_dataset(cfg, rank)
        self.models = self.prepare_model(cfg, rank)   
        self.losses = self.init_losses(cfg, rank)        
        self.view_rendering, self.pose = self.init_geometry(cfg, rank) 
        self.set_optimizer()
        
        if self.pretrain and rank == 0:
            self.load_weights()

    def set_optimizer(self):
        """
        分组优化器：depth_net 和 pose_net 使用不同的学习率。
        
        pose_model='gt':    只有 depth_net
        pose_model='mono'/'prior': depth_net(小lr) + pose_net(大lr)
        """
        if self.pose_model != 'gt' and 'pose_net' in self.models:
            depth_lr = getattr(self, 'depth_lr', self.learning_rate * 0.1)
            pose_lr = getattr(self, 'pose_lr', self.learning_rate)
            
            self.optimizer = optim.Adam([
                {'params': self.models['depth_net'].parameters(), 
                 'lr': depth_lr},
                {'params': self.models['pose_net'].parameters(), 
                 'lr': pose_lr},
            ])
            print(f'[Optimizer] depth_net lr={depth_lr}, pose_net lr={pose_lr}')
        else:
            parameters_to_train = []
            for v in self.models.values():
                parameters_to_train += list(v.parameters())
            self.optimizer = optim.Adam(parameters_to_train, self.learning_rate)

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            self.scheduler_step_size,
            0.1
        )
     
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_geometry(self, cfg, rank):
        view_rendering = ViewRendering(cfg, rank)
        pose = Pose(cfg)
        return view_rendering, pose
    
    def init_losses(self, cfg, rank):
        spatio = self.spatio_temporal or self.spatio
        if self.aug_depth and spatio:
            loss_model = DepthSynLoss(cfg, rank)
        elif self.geo_consist and spatio:
            loss_model = GeoMultiCamLoss(cfg, rank)
        elif self.super_depth and spatio:
            loss_model = DepthCamLoss(cfg, rank)
        elif spatio:
            loss_model = MultiCamLoss(cfg, rank)
        else:
            loss_model = SingleCamLoss(cfg, rank)
        return loss_model
        
    def prepare_model(self, cfg, rank):
        models = {}
        # for posenet
        if self.pose_model != 'gt':
            models['pose_net'] = self.set_pose_net(cfg).to(self.device)
        
        # for depthnet
        models['depth_net'] = DinoDepthNet(cfg).to(self.device)

        # DDP training
        if self.ddp_enable == True:
            from torch.nn.parallel import DistributedDataParallel as DDP            
            process_group = dist.new_group(list(range(self.world_size)))
            # set ddp configuration
            for k, v in models.items():
                # sync batchnorm 
                v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(v, process_group)
                # DDP enable
                models[k] = DDP(v, device_ids=[rank], broadcast_buffers=True)
        return models
    
    def set_pose_net(self, cfg):
        if self.pose_model == 'mono':
            pose_net = MonoPoseNet(cfg)
        elif self.pose_model == 'prior':
            pose_net = PriorPoseNet(cfg)
        elif self.pose_model == 'concat':
            pose_net = ConcatPoseNet(cfg)
        else:
            raise NotImplementedError(f"Not implemented pose model: {self.pose_model}")
        return pose_net
        
    def prepare_dataset(self, cfg, rank):
        if rank == 0:
            print('### Preparing Datasets')
        if self.mode == 'train':
            self.set_train_dataloader(cfg, rank)
            if rank == 0:
                self.set_val_dataloader(cfg)
        if self.mode == 'eval':
            self.set_eval_dataloader(cfg)
        
    def process_batch(self, inputs, rank):
        """
        Pass a minibatch through the network and generate images, depth maps, and losses.
        """
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(rank) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(rank)   

        outputs = self.train_inference(inputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses  
    
    def eval_inference(self, inputs):  # 只在单卡进行推理
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(0)  

        # pre-calculate inverse of the extrinsic matrix
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}           
        depth_feats = self.predict_depth(inputs)
        for cam in range(self.num_cams):                  
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])
        self.compute_depth_maps(inputs, outputs)
        return outputs

    def train_inference(self, inputs):
        """
        This function sets dataloader for validation in training.
        """          
        # pre-calculate inverse of the extrinsic matrix
        inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])
        
        # init dictionary 
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}  

        pose_pred = self.predict_pose(inputs)

        depth_feats = self.predict_depth(inputs)
        for cam in range(self.num_cams):       
            outputs[('cam', cam)].update(pose_pred[('cam', cam)])             
            outputs[('cam', cam)].update(depth_feats[('cam', cam)])

        if self.syn_visualize:
            outputs['disp_vis'] = depth_feats['disp_vis']
            
        self.compute_depth_maps(inputs, outputs)
        return outputs

    def predict_pose(self, inputs):      
        """
        This function predicts poses.
        - 'gt':    GT pose directly
        - 'mono' / 'prior': 统一走 Pose.compute_pose (cam 0 预测 + 外参传播)
        """
        if self.pose_model == 'gt':
            return self.pose.compute_pose_from_gt(inputs)
        else:  # 'mono', 'prior', 'concat'
            net = self.models['pose_net']
            if (self.mode != 'train') and self.ddp_enable:
                net = self.models['pose_net'].module
            pred = self.pose.compute_pose(net, inputs)

            # ---- Diagnostic: always check for outliers, print raw data ----
            with torch.no_grad():
                gt = self.pose.compute_pose_from_gt(inputs)
                for f_i in self.frame_ids[1:]:
                    T_gt = gt[('cam', 0)][('cam_T_cam', 0, f_i)]
                    t_gt = T_gt[:, :3, 3]
                    gt_norm = t_gt.norm(dim=1).mean().item()

                    if gt_norm > 5.0:
                        T_pred = pred[('cam', 0)][('cam_T_cam', 0, f_i)]
                        t_pred = T_pred[:, :3, 3]
                        fname = inputs.get('filename', 'N/A')
                        idx_val = inputs.get('idx', 'N/A')

                        # Raw world poses for cam 0
                        pose_t = inputs['pose'][:, 0, :, :]      # [B,4,4]
                        pose_other = inputs['pose_bwd'][:, 0, :, :] if f_i < 0 else inputs['pose_fwd'][:, 0, :, :]

                        print(f'  [OUTLIER] fid={f_i} | '
                              f't_pred={t_pred.norm(dim=1).mean():.4f} t_gt={gt_norm:.4f} | '
                              f'idx={idx_val}, filename={fname}')
                        print(f'    pose_t[:3,3]     = {pose_t[0, :3, 3].tolist()}')
                        print(f'    pose_other[:3,3] = {pose_other[0, :3, 3].tolist()}')
                        world_diff = (pose_t[0, :3, 3] - pose_other[0, :3, 3]).norm().item()
                        print(f'    world_coord_diff = {world_diff:.4f}m')
                        print(f'    t_gt_xyz = {t_gt[0].tolist()}')

            return pred

    
    def predict_depth(self, inputs):
        """
        This function predicts disparity maps.
        """                  
        net = self.models['depth_net']
        if (self.mode != 'train') and self.ddp_enable: 
            net = self.models['depth_net'].module

        if self.depth_model == 'dino':
            return net(inputs)
   
        depth_feats = {}
        for cam in range(self.num_cams):
            input_depth = inputs[('color_aug', 0, 0)][:, cam, ...]
            depth_feats[('cam', cam)] = net(input_depth)

        return depth_feats
    
    def compute_depth_maps(self, inputs, outputs):     
        """
        This function computes depth map for each viewpoint.
        """                  
        for cam in range(self.num_cams):
            for scale in self.scales: # [0]
                disp = outputs[('cam', cam)][('disp', scale)]  # 输出的视差
                outputs[('cam', cam)][('depth', scale)] = self.disp_to_depth(disp) # 视差转深度图
                if self.aug_depth:
                    disp = outputs[('cam', cam)][('disp', scale, 'aug')]
                    outputs[('cam', cam)][('depth', scale, 'aug')] = self.disp_to_depth(disp) 
    
    def to_depth(self, disp_in, K_in):        
        """
        This function of VFDepth transforms disparity value into depth map while multiplying the value with the focal length.
        E2Depth remove this because we used focal length messesage in fusion block. 
        """
        min_disp = 1/self.max_depth
        max_disp = 1/self.min_depth
        disp_range = max_disp-min_disp

        disp_in = F.interpolate(disp_in, [self.height, self.width],mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in 
        depth = 1/disp
        return depth * K_in[:, 0:1, 0:1].unsqueeze(2)/self.focal_length_scale
    
    def disp_to_depth(self, disp):
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp = F.interpolate(disp, [self.height, self.width],mode='bilinear', align_corners=False)
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1 / scaled_disp
        return depth
        
    def compute_losses(self, inputs, outputs):
        """
        This function computes losses.
        """          
        losses = 0
        loss_fn = defaultdict(list)
        loss_mean = defaultdict(float)

        # generate image and compute loss per cameara
        for cam in range(self.num_cams):  
            # 每一个相机单独计算loss
            self.pred_cam_imgs(inputs, outputs, cam)
            cam_loss, loss_dict = self.losses(inputs, outputs, cam)
            
            losses += cam_loss  
            for k, v in loss_dict.items():
                loss_fn[k].append(v)

        losses /= self.num_cams
        for k in loss_fn.keys():
            loss_mean[k] = sum(loss_fn[k]) / float(len(loss_fn[k]))

        loss_mean['total_loss'] = losses

        # Pose metrics (only when pose_model != 'gt' and GT is available)
        if self.pose_model != 'gt' and 'pose' in inputs:
            pose_metrics = self.compute_pose_metrics(inputs, outputs)
            loss_mean.update(pose_metrics)
        
        return loss_mean 

    def compute_pose_metrics(self, inputs, outputs):
        """
        Compute pose error for front camera (cam 0) vs GT.
        Returns rotation error in degrees and translation error in meters.
        """
        gt_pose = self.pose.compute_pose_from_gt(inputs)
        rot_errors = []
        trans_errors = []
        
        for frame_id in self.frame_ids[1:]:
            T_pred = outputs[('cam', 0)][('cam_T_cam', 0, frame_id)]
            T_gt = gt_pose[('cam', 0)][('cam_T_cam', 0, frame_id)]
            
            # Rotation error: geodesic distance
            R_diff = T_pred[:, :3, :3].transpose(1, 2) @ T_gt[:, :3, :3]
            trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
            angle_rad = torch.acos(((trace - 1) / 2).clamp(-1, 1))
            rot_errors.append(angle_rad.mean())
            
            # Translation error: L2
            t_err = (T_pred[:, :3, 3] - T_gt[:, :3, 3]).norm(dim=1)
            trans_errors.append(t_err.mean())
        
        n = len(rot_errors)
        return {
            'pose/rot_err_deg': sum(e.item() for e in rot_errors) / n * 180 / 3.14159265,
            'pose/trans_err_m': sum(e.item() for e in trans_errors) / n,
        }

    def pred_cam_imgs(self, inputs, outputs, cam):
        """
        This function renders projected images using camera parameters and depth information.
        """                  
        rel_pose_dict = self.pose.compute_relative_cam_poses(inputs, outputs, cam)
        self.view_rendering(inputs, outputs, cam, rel_pose_dict)