"""
geometry pose.py
"""
import numpy as np
import torch
import torch.nn as nn
from utils.transforms import axis_angle_to_matrix


def vec_to_matrix(rot_angle, trans_vec, invert=False):
    """
    This function transforms rotation angle and translation vector into 4x4 matrix.
    """
    # initialize matrices
    b, _, _ = rot_angle.shape
    R_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)
    T_mat = torch.eye(4).repeat([b, 1, 1]).to(device=rot_angle.device)

    R_mat[:, :3, :3] = axis_angle_to_matrix(rot_angle).squeeze(1)
    t_vec = trans_vec.clone().contiguous().view(-1, 3, 1)

    if invert == True:
        R_mat = R_mat.transpose(1,2)
        t_vec = -1 * t_vec

    T_mat[:, :3,  3:] = t_vec

    if invert == True:
        P_mat = torch.matmul(R_mat, T_mat)
    else :
        P_mat = torch.matmul(T_mat, R_mat)
    return P_mat  # [1,4,4]


class Pose:
    """
    Class for multi-camera pose calculation 
    """
    def __init__(self, cfg):
        self.read_config(cfg)
        
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def compute_pose(self, net, inputs):
        """
        This function computes multi-camera posse in accordance with the network structure.
        """
        if self.pose_model == 'single':
            pose = {}
            for cam in range(self.num_cams):
                pose[('cam', cam)] = self.get_single_pose(net, inputs, cam)        
        else:
             # 只用前向相机 (cam 0) 预测姿态
            pose_cam0 = self.get_single_pose(net, inputs, cam=0)
            # 通过外参传播到所有相机
            pose = self.distribute_pose(pose_cam0, inputs['extrinsics'], inputs['extrinsics_inv'])
        return pose
    
    def get_single_pose(self, net, inputs, cam):
        """
        This function computes pose for a single camera.
        """
        output = {}
        for f_i in self.frame_ids[1:]:  # [0, -1, 1]
            # To maintain ordering we always pass frames in temporal order
            frame_ids = [-1, 0] if f_i < 0 else [0, 1]  # 计算前两张或者后两张，参照点分别为第一张和第二张，保证极性是一致的。
            axisangle, translation = net(inputs, frame_ids, cam)  
            output[('cam_T_cam', 0, f_i)] = vec_to_matrix(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))  # [1, 4, 4]          
        return output
        
    def distribute_pose(self, poses, exts, exts_inv):
        """
        This function distrubutes pose to each camera by using the canonical pose and camera extrinsics.
        (default: reference camera 0)
        """
        outputs = {}
        for cam in range(self.num_cams):
            outputs[('cam',cam)] = {}
        # Refernce camera(canonical)
        ref_ext = exts[:, 0, ...]
        ref_ext_inv = exts_inv[:, 0, ...]
        for f_i in self.frame_ids[1:]:
            ref_T = poses['cam_T_cam', 0, f_i].float() # canonical pose      
            # Relative cameras(canonical)            
            for cam in range(self.num_cams):
                cur_ext = exts[:,cam,...]
                cur_ext_inv = exts_inv[:,cam,...]                
                cur_T = cur_ext_inv @ ref_ext @ ref_T @ ref_ext_inv @ cur_ext
                outputs[('cam',cam)][('cam_T_cam', 0, f_i)] = cur_T            
        return outputs 
    
    def compute_relative_cam_poses(self, inputs, outputs, cam):
        """
        This function computes spatio & spatio-temporal transformation for images from different viewpoints.
        """
        ref_ext = inputs['extrinsics'][:, cam, ...]  # 相机外参
        target_view = outputs[('cam', cam)]  # 对应相机的所有输出信息（disp,depth,pose）
    
        rel_pose_dict = {} # 相对姿态字典
        # precompute the relative pose 空间计算
        if self.spatio:  
            # current time step (spatio)
            for cur_index in self.rel_cam_list[cam]:  # [1,2] [0,3] [0,4] [1,5] [2,5] [3,4] 在misc中定义的; 相机0对应的是相机1和2；
                # for partial surround view training
                if cur_index >= self.num_cams:
                    continue

                cur_ext_inv = inputs['extrinsics_inv'][:, cur_index, ...]
                rel_pose_dict[(0, cur_index)] = torch.matmul(cur_ext_inv, ref_ext)  # TODO:我总感觉这里写错了

        if self.spatio_temporal:  
            # different time step (spatio-temporal)  时空计算（其实就是计算相邻两帧的外参*变换矩阵）
            for frame_id in self.frame_ids[1:]:                 
                for cur_index in self.rel_cam_list[cam]:
                    # for partial surround view training
                    if cur_index >= self.num_cams:
                        continue

                    T = target_view[('cam_T_cam', 0, frame_id)]  # 对应的变换矩阵
                    # assuming that extrinsic doesn't change
                    rel_ext = rel_pose_dict[(0, cur_index)]  # 参考相机的变换外参
                    rel_pose_dict[(frame_id, cur_index)] = torch.matmul(rel_ext, T) # using matmul speed up  参考相机的变换矩阵
        return rel_pose_dict
    
    def compute_pose_from_gt(self, inputs):
        """
            使用 GT 姿态 (world<-cam) 直接构造每个相机的相对位姿矩阵 cam_T_cam(0 -> frame_id)。

            约定：
            - inputs['pose']      : [B, 6, 4, 4], world <- cam at time t
            - inputs['pose_fwd']  : [B, 6, 4, 4], world <- cam at time t+1
            - inputs['pose_bwd']  : [B, 6, 4, 4], world <- cam at time t-1

            输出格式保持与原 VFDepth 一致：
            - pose[('cam', cam)][('cam_T_cam', 0,  1)] : cam_t -> cam_t+1
            - pose[('cam', cam)][('cam_T_cam', 0, -1)] : cam_t -> cam_t-1

            NOTE: 使用 float64 计算以避免大世界坐标（~60000m）下 float32
            精度不足导致的相对位姿计算误差。计算完成后转回 float32。
        """
        # 用 float64 做矩阵求逆和乘法，避免大坐标下的精度损失
        poses_t   = inputs['pose'].to(dtype=torch.float64)      # [B,6,4,4]
        poses_fwd = inputs['pose_fwd'].to(dtype=torch.float64)   # [B,6,4,4]
        poses_bwd = inputs['pose_bwd'].to(dtype=torch.float64)   # [B,6,4,4]

        B, num_cams, _, _ = poses_t.shape

        pose_out = {}
        for cam in range(num_cams):
            pose_out[('cam', cam)] = {}
            T_wc_t   = poses_t[:,   cam, ...]   # world <- cam_t
            T_wc_fwd = poses_fwd[:, cam, ...]   # world <- cam_t+1
            T_wc_bwd = poses_bwd[:, cam, ...]   # world <- cam_t-1
            # T_cam_rel_fwd = (world<-cam_t+1)^(-1) @ (world<-cam_t) 相对位姿：cam_t -> cam_t+1
            T_wc_fwd_inv   = torch.inverse(T_wc_fwd)
            T_cam_rel_fwd  = torch.matmul(T_wc_fwd_inv, T_wc_t)   # [B,4,4]
            # T_cam_rel_bwd = (world<-cam_t-1)^(-1) @ (world<-cam_t) 相对位姿：cam_t -> cam_t-1
            T_wc_bwd_inv   = torch.inverse(T_wc_bwd)
            T_cam_rel_bwd  = torch.matmul(T_wc_bwd_inv, T_wc_t)   # [B,4,4]

            # 转回 float32 供下游使用
            pose_out[('cam', cam)][('cam_T_cam', 0,  1)] = T_cam_rel_fwd.float()
            pose_out[('cam', cam)][('cam_T_cam', 0, -1)] = T_cam_rel_bwd.float()
        return pose_out