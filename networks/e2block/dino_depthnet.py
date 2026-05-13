from collections import OrderedDict
import os
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.blocks import upsample, conv2d, pack_cam_feat, unpack_cam_feat
from utils.misc import get_proj_dir, load_cfg
from datasets import get_train_dataset
from networks import DINOv3
from networks.e2block.fusion import CVT3DNet
from networks.e2block.depth_decoder import EdgeRefineDecoder

MODEL_NAME = "dinov3_vits16"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureAdapter(nn.Module):
    def __init__(self, 
                 dim_in=384,
                 dim_out=256,
                 feat_num=3,
                 ):
        super().__init__()
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
                nn.GroupNorm(8, dim_out),
                nn.ReLU(inplace=True)
            )
            for _ in range(feat_num)
        ])
    
    def forward(self, feats):
        layer_feats = []
        for f, proj in zip(feats, self.projects):
            f_proj = proj(f)  # [B_cam, C_l, H_i, W_i]
            layer_feats.append(f_proj)
        return layer_feats


class DinoDepthNet(nn.Module):
    """
    ThisDepth Depth estimation network using DINOv3 as backbone.
    """
    def __init__(self, cfg):
        super(DinoDepthNet, self).__init__()
        self.read_config(cfg)
        proj_dir = get_proj_dir()
        DINOv3_REPO = os.path.join(proj_dir, 'networks/dinov3')
        weight_path = glob.glob(os.path.join(DINOv3_REPO, "weights/dinov3_"+self.dino_model+str(self.dino_patch_size)+"_*"))[0]
        self.encoder = DINOv3(self.dino_model, img_size=min(self.height, self.width), weights=weight_path, freeze=self.freeze)

        self.adapter = FeatureAdapter(
            dim_in=self.dino_feat_dim,
            dim_out=self.fusion_feat_in_dim,
            feat_num=len(self.intermediate_layer_idx)
        )
        self.fusion_net = CVT3DNet(cfg, self.fusion_feat_in_dim, self.fusion_feat_out_dim)
        self.decoder = EdgeRefineDecoder(self.fusion_feat_out_dim)

    def forward(self, inputs):
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam', cam)] = {}

        sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(self.num_cams)], 1)
        B, K, C, H ,W = sf_images.shape
        
        packed_input = pack_cam_feat(sf_images)  # 变成[b*6,3,384, 640]
        packed_feats = self.encoder_forward(packed_input)
        packed_feats_agg = self.adapter(packed_feats)  # [b*6, 256, 48, 80]
        feats_agg = [unpack_cam_feat(f, B, K) for f in packed_feats_agg]
        # fusion_net, backproject each feature into the 3D voxel space
        fusion_dict = self.fusion_net(inputs, feats_agg)   #  [6, 128, 48, 80]
        packed_depth_outputs = self.decoder(fusion_dict['proj_feat'], H, W)    # torch.Size([6, 1, 384, 640])
        # # 1 6 1 384 640    
        depth_outputs = unpack_cam_feat(packed_depth_outputs, B, K)
        
        for cam in range(K):
            for k in depth_outputs.keys():
                outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]

        return outputs # [1,6,1,384,640]

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def encoder_forward(self, inputs):
        # packed images for surrounding view
        outputs = []
        feats = self.encoder.get_intermediate_layers(inputs, n=self.intermediate_layer_idx, return_class_token=False)  # 输出[5 8 11]层，因为第二层和第五层差距不大
        # ADD DPT Head 的Resize层, 能够提供多尺度的输出
        for i, x in enumerate(feats):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1],self.height//self.dino_patch_size, self.width//self.dino_patch_size))  # [6 384 24 40]
            outputs.append(x)
        return outputs
    
    def train(self, mode = True):
        super().train(mode)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        return self
    
    def collect_wavelet_features(self):
        vis = {}
        for name in ['refinenet4', 'refinenet3', 'refinenet2', 'refinenet1']:
            block = getattr(self.scratch, name)
            if block.use_wavelet and hasattr(block, 'wavelet_edge'):
                w = block.wavelet_edge
                if hasattr(w, '_vis_cache'):
                    vis[name] = w._vis_cache
        return vis