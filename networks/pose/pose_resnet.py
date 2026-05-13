# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict
from networks import ResnetEncoder

class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]  # 前三位 表示旋转
        translation = out[..., 3:]  # 后三位 表示位移

        return axisangle, translation  # [B,N,1,3]  


class MonoPoseNet(nn.Module):
    """
    Pytorch module for a pose network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """
    def __init__(self, cfg):
        super(MonoPoseNet, self).__init__()
        num_layers = cfg['model']['num_layers']  # 18
        pretrained = cfg['model']['weights_init']   # True
        
        self.pose_encoder = ResnetEncoder(num_layers, pretrained, num_input_images = 2)
        del self.pose_encoder.encoder.fc # For ddp training
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, 
                                        num_input_features=1, 
                                        num_frames_to_predict_for=1)
                                        
    def forward(self, inputs, frame_ids, cam):
        pose_inputs = [inputs['color_aug', f_i, 0][:,cam,...] for f_i in frame_ids]
        input_images = torch.cat(pose_inputs, 1)
        pose_feature = [self.pose_encoder(input_images)]
        axis_angle, translation = self.pose_decoder(pose_feature)
        return axis_angle, torch.clamp(translation, -4.0, 4.0) # for DDAD dataset