"""
geometry view_rendering.py
"""

# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Projection(nn.Module):
    """
    把像素坐标 + 深度 ⇄ 相机坐标系下的 3D 点
    """
    def __init__(self, batch_size, height, width, device):
        super().__init__()
        self.batch_size = batch_size 
        self.width = width   # 640
        self.height = height  # 384
        
        # initialize img point grid
        img_points = np.meshgrid(range(width), range(height), indexing='xy')
        img_points = torch.from_numpy(np.stack(img_points, 0)).float()
        img_points = torch.stack([img_points[0].view(-1), img_points[1].view(-1)], 0).repeat(batch_size, 1, 1)
        img_points = img_points.to(device)  #【1,2,384*640】
        
        self.to_homo = torch.ones([batch_size, 1, width*height]).to(device) # 【1,1,384*640】
        self.homo_points = torch.cat([img_points, self.to_homo], 1)  #【1,3,384*640】 避免每次 forward 重建像素网格，提高效率 

    def backproject(self, invK, depth):  # 内参的逆和深度图
        """
        This function back-projects 2D image points to 3D.
        """
        depth = depth.view(self.batch_size, 1, -1) # [1,1, 384*640]

        points3D = torch.matmul(invK[:, :3, :3], self.homo_points) # [1, 3, 384*640] matmul只看最后两位
        points3D = depth*points3D #【1,3,245760】每一个对应位置相乘
        return torch.cat([points3D, self.to_homo], 1) #数据齐次化，最后形状为【1,4,245760】为3D点云，表示对应的xyz1坐标
    
    def reproject(self, K, points3D, T):  
        """
        This function reprojects transformed 3D points to 2D image coordinate.
        K: [1,4,4]
        point3D: [1,4,245760]
        T: [1,4,4]
        """
        # project points  [1,3,4] @ [1, 4, 245760]=[1,3,245760] 表示新相机的投影矩阵x,y,z
        points2D = (K @ T)[:,:3, :] @ points3D

        # normalize projected points for grid sample function
        norm_points2D = points2D[:, :2, :]/(points2D[:, 2:, :] + 1e-7)  # 归一化后得到像素坐标 [1,2,384*640]
        norm_points2D = norm_points2D.view(self.batch_size, 2, self.height, self.width) # [1,2,384,640]
        norm_points2D = norm_points2D.permute(0, 2, 3, 1) # [1,384,640,2]

        norm_points2D[..., 0 ] /= self.width - 1 # 
        norm_points2D[..., 1 ] /= self.height - 1  
        norm_points2D = (norm_points2D-0.5)*2
        return norm_points2D        

    def forward(self, depth, T, bp_invK, rp_K):
        cam_points = self.backproject(bp_invK, depth)
        pix_coords = self.reproject(rp_K, cam_points, T)
        return pix_coords

class ViewRendering(nn.Module):
    """
    Class for rendering images from given camera parameters and pixel wise depth information
    """
    def __init__(self, cfg, rank):
        super().__init__()
        self.read_config(cfg)
        self.rank = rank
        self.project = self.init_project_imgs(rank)      
            
    def read_config(self, cfg):    
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)
                
    def init_project_imgs(self, rank):
        project_imgs = {}
        project_imgs = Projection(
                self.batch_size, self.height, self.width, rank)
        return project_imgs    
    
    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features. 
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1,2,3), keepdim=True) / (mask.sum(dim=(1,2,3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1,2,3), keepdim=True) / (c*h*w) # GPT认为这里应该用masked_count,计算有误
        return mean, torch.sqrt(var + 1e-16)     
    
    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1,3,1,1)

            mask_sum = mask.sum(dim=(-3,-2,-1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()   

    def get_virtual_image(self, src_img, src_mask, tar_depth, tar_invK, src_K, T, scale=0):
        """
        This function warps source image to target image using backprojection and reprojection process. 
        把src_img基于目标像素深度tar_depth反投影到目标视图并 grid_sample 获取 img_warped 与 mask_warped。
        """
        # do reconstruction for target from source   
        pix_coords = self.project(tar_depth, T, tar_invK, src_K)
        
        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear', 
                                    padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest', 
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, 
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped

    def get_virtual_depth(self, src_depth, src_mask, src_invK, src_K, tar_depth, tar_invK, tar_K, T, min_depth, max_depth, scale=0):
        """
        This function backward-warp source depth into the target coordinate.
        src -> target
        """       
        # transform source depth
        b, _, h, w = src_depth.size()    
        src_points = self.project.backproject(src_invK, src_depth)
        src_points_warped = torch.matmul(T[:, :3, :], src_points)
        src_depth_warped = src_points_warped.reshape(b, 3, h, w)[:, 2:3, :, :]

        # reconstruct depth: backward-warp source depth to the target coordinate
        pix_coords = self.project(tar_depth, torch.inverse(T), tar_invK, src_K)  
        depth_warped = F.grid_sample(src_depth_warped, pix_coords, mode='bilinear', 
                                        padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_depth = torch.isnan(depth_warped)
        depth_warped[inf_depth] = 2.0
        inf_regions = torch.isnan(mask_warped)
        mask_warped[inf_regions] = 0
        
        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1, pix_coords < -1).sum(dim=1, keepdim=True) > 0

        # range handling
        valid_depth_min = (depth_warped > min_depth)
        depth_warped[~valid_depth_min] = min_depth
        valid_depth_max = (depth_warped < max_depth)
        depth_warped[~valid_depth_max] = max_depth
        return depth_warped, (~invalid_mask).float() * mask_warped * valid_depth_min * valid_depth_max        
        
    def forward(self, inputs, outputs, cam, rel_pose_dict):
        # predict images for each scale(default = scale 0 only)
        source_scale = 0
        
        # ref inputs
        ref_color = inputs['color', 0, source_scale][:,cam, ...]   # 【1,3,384,640】全是归一化后的值  
        ref_mask = inputs['mask'][:, cam, ...] #【1,1,384,640】1和0
        ref_K = inputs[('K', source_scale)][:,cam, ...] #【1,4,4】标准内参
        ref_invK = inputs[('inv_K', source_scale)][:,cam, ...]  
        
        # output
        target_view = outputs[('cam', cam)] 
        
        for scale in self.scales:  # [0]         
            ref_depth = target_view[('depth', scale)] # 深度值        
            for frame_id in self.frame_ids[1:]:  # [0, -1, 1]              
                # for temporal learning
                T = target_view[('cam_T_cam', 0, frame_id)]  # 相对位姿变换矩阵 【1,4,4】
                src_color = inputs['color', frame_id, source_scale][:, cam, ...] 
                src_mask = inputs['mask'][:, cam, ...]
                warped_img, warped_mask = self.get_virtual_image(
                    src_color, 
                    src_mask, 
                    ref_depth, 
                    ref_invK, 
                    ref_K, 
                    T, 
                    source_scale
                )
                
                if self.intensity_align: # True  强度对齐
                    warped_img = self.get_norm_image_single(
                        ref_color, 
                        ref_mask,
                        warped_img, 
                        warped_mask
                    )
                
                target_view[('color', frame_id, scale)] = warped_img
                target_view[('color_mask', frame_id, scale)] = warped_mask            

            # spatio-temporal learning  根据姿态计算
            if self.spatio or self.spatio_temporal:

                if self.spatio_temporal:
                    frame_ids_for_overlap = self.frame_ids     # [-1,0,1]
                else:
                    frame_ids_for_overlap = [0]

                for frame_id in frame_ids_for_overlap:
                    overlap_img = torch.zeros_like(ref_color)  # 重叠区域的图像
                    overlap_mask = torch.zeros_like(ref_mask)  # 重叠区域的mask
                    
                    for cur_index in self.rel_cam_list[cam]:
                        # for partial surround view training
                        if cur_index >= self.num_cams: 
                            continue

                        src_color = inputs['color', frame_id, source_scale][:, cur_index, ...]
                        src_mask = inputs['mask'][:, cur_index, ...]
                        src_K = inputs[('K', source_scale)][:, cur_index, ...]                        
                        
                        rel_pose = rel_pose_dict[(frame_id, cur_index)]
                        warped_img, warped_mask = self.get_virtual_image(
                            src_color, 
                            src_mask, 
                            ref_depth, 
                            ref_invK, 
                            src_K,
                            rel_pose, # 外参计算的结果
                            source_scale
                        )

                        if self.intensity_align:
                            warped_img = self.get_norm_image_single(
                                ref_color, 
                                ref_mask,
                                warped_img, 
                                warped_mask
                            )
                        # assuming no overlap between warped images
                        overlap_img = overlap_img + warped_img
                        overlap_mask = overlap_mask + warped_mask
                    
                    target_view[('overlap', frame_id, scale)] = overlap_img # 将相邻相机的图像映射到参考相机后的重建结果。
                    target_view[('overlap_mask', frame_id, scale)] = overlap_mask
                    
            # depth augmentation at a novel view
            if self.aug_depth:
                tform_depth = []
                tform_mask = []

                aug_ext = inputs['extrinsics_aug'][:, cam, ...]
                aug_ext_inv = torch.inverse(aug_ext)                
                aug_K, aug_invK = ref_K, ref_invK
                aug_depth = target_view[('depth', scale, 'aug')]

                for i, curr_index in enumerate(self.rel_cam_list[cam] + [cam]):
                    # for partial surround view training
                    if curr_index >= self.num_cams: 
                        continue

                    src_ext = inputs['extrinsics'][:, curr_index, ...]                        
                    
                    src_depth = outputs[('cam', curr_index)][('depth', scale)]
                    src_mask = inputs['mask'][:, curr_index, ...]                
                    src_invK = inputs[('inv_K', source_scale)][:,curr_index, ...]
                    src_K = inputs[('K', source_scale)][:,curr_index, ...]

                    # current view to the novel view
                    rel_pose = torch.matmul(aug_ext_inv, src_ext)
                    warp_depth, warp_mask = self.get_virtual_depth(
                        src_depth, 
                        src_mask, 
                        src_invK, 
                        src_K,
                        aug_depth, 
                        aug_invK, 
                        aug_K, 
                        rel_pose,
                        self.min_depth,
                        self.max_depth
                    )

                    tform_depth.append(warp_depth)
                    tform_mask.append(warp_mask)

                target_view[('tform_depth', scale)] = tform_depth
                target_view[('tform_depth_mask', scale)] = tform_mask

        outputs[('cam', cam)] = target_view