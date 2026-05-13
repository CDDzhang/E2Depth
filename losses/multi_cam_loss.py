# Copyright (c) 2023 42dot. All rights reserved.
import torch
import torch.nn.functional as F
from utils.transforms import matrix_to_euler_angles 

from losses.loss_util import compute_photometric_loss, compute_masked_loss
from losses.single_cam_loss import SingleCamLoss


class MultiCamLoss(SingleCamLoss):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """
    def __init__(self, cfg, rank):
        super(MultiCamLoss, self).__init__(cfg, rank)
    
    def compute_spatio_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        """
        This function computes spatial loss.
        ref_mask 表示数据集本身的采集遮挡区域
        """        
        # self occlusion mask * overlap region mask
        
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color',0, 0][:,cam, ...]         
        }        
        spatio_loss = compute_photometric_loss(**loss_args)

        spatio_mask = ref_mask * target_view[('overlap_mask', 0, scale)]
        target_view[('overlap_mask', 0, scale)] = spatio_mask         
        return compute_masked_loss(spatio_loss, spatio_mask) 

    def compute_spatio_soft_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None):
        loss_args = {
            'pred': target_view[('overlap', 0, scale)],
            'target': inputs['color',0, 0][:,cam, ...]         
        }        
        spatio_photo = compute_photometric_loss(**loss_args)

        # --- masks ---
        valid = ref_mask.float()                                # 硬门控：无效区域权重=0

        overlap = target_view[('overlap_mask', 0, scale)].float()
        overlap = (overlap > 0).float()  # 二值化

        # 对 overlap 做轻微平滑，避免硬边导致梯度断裂/孔洞
        overlap_soft = F.avg_pool2d(overlap, kernel_size=3, stride=1, padding=1).clamp(0., 1.)

        # soft 权重（仅在 valid 内生效），给不可见区域留少量梯度保底
        eps = 0.1
        weight = valid * (eps + (1.0 - eps) * overlap_soft)     # [B,1,H,W]

        # 稳定的加权均值（比 sum/sum 更稳，不依赖极小分母）
        spatio_loss = (spatio_photo * weight).sum() / (weight.sum() + 1e-8)

        # 注意：不再覆写 target_view[('overlap_mask', 0, scale)]，避免把乘积写回去
        return spatio_loss

    def compute_spatio_tempo_loss(self, inputs, target_view, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None):
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in self.frame_ids[1:]:

            pred_mask = ref_mask * target_view[('overlap_mask', frame_id, scale)]
            pred_mask = pred_mask * reproj_loss_mask 

            loss_args = {
                'pred': target_view[('overlap', frame_id, scale)],
                'target': inputs['color',0, 0][:,cam, ...]
            } 

            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)

        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)    

        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses on all the True values of masks.
        spatio_tempo_loss, _ = torch.min(spatio_tempo_losses, dim=1, keepdim=True)
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)
        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask) 

    
    def compute_pose_con_loss(self, inputs, outputs, cam=None, scale=None, ref_mask=None, reproj_loss_mask=None) :
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """        
        ref_output = outputs[('cam', 0)]
        ref_ext = inputs['extrinsics'][:, 0, ...]
        ref_ext_inv = inputs['extrinsics_inv'][:, 0, ...]
   
        cur_output = outputs[('cam', cam)]
        cur_ext = inputs['extrinsics'][:, cam, ...]
        cur_ext_inv = inputs['extrinsics_inv'][:, cam, ...] 
        
        trans_loss = 0.
        angle_loss = 0.
     
        for frame_id in self.frame_ids[1:]:
            ref_T = ref_output[('cam_T_cam', 0, frame_id)]
            cur_T = cur_output[('cam_T_cam', 0, frame_id)]    

            cur_T_aligned = ref_ext_inv@cur_ext@cur_T@cur_ext_inv@ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:,:3,:3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:,:3,:3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:,:3,3] - cur_T_aligned[:,:3,3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff
        
        pose_loss = (trans_loss + 10 * angle_loss) / len(self.frame_ids[1:])
        return pose_loss
    
    def forward(self, inputs, outputs, cam):        
        loss_dict = {}
        cam_loss = 0. # loss across the multi-scale
        target_view = outputs[('cam', cam)]
        for scale in self.scales:
            kargs = {
                'cam': cam,
                'scale': scale,
                'ref_mask': inputs['mask'][:,cam,...]
            }
                          
            reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
            
            spatio_loss = self.compute_spatio_soft_loss(inputs, target_view, **kargs) if self.spatio else 0 
            # spatio_loss = self.compute_spatio_loss(inputs, target_view, **kargs) if self.spatio else 0 
            kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
         
            spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs) if self.spatio_temporal else 0

            # pose consistency loss
            
            pose_loss = self.compute_pose_con_loss(inputs, outputs, **kargs) if (self.pose_consist and cam != 0) else 0

            cam_loss += reprojection_loss
            cam_loss += self.disparity_smoothness * smooth_loss / (2 ** scale) # 0.001         
            cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss   #  0.03   0.1             
            cam_loss += self.pose_loss_coeff* pose_loss   # 0
            
            ##########################
            # for logger
            ##########################
            if scale == 0:
                loss_dict['reproj_loss'] = reprojection_loss.item()
                loss_dict['smooth'] = smooth_loss.item() 
                loss_dict['spatio_loss'] = spatio_loss.item() if self.spatio else 0
                loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item() if self.spatio_temporal else 0
                loss_dict['pose'] = pose_loss.item() if self.pose_consist and cam != 0 else 0
                
                # log statistics
                self.get_logs(loss_dict, target_view, cam)                        
        
        cam_loss /= len(self.scales)
        return cam_loss, loss_dict