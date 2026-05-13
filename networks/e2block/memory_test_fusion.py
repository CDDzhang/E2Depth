# Copyright (c) 2023 42dot. All rights reserved.
"""
depth_fusion--version3 (memory-optimized)

Key optimization in backproject_into_voxel:
  - Old: warp all 6 cameras -> store 6 full [B,C,N] tensors -> then fuse
  - New: warp 1 camera -> immediately scatter non-overlap -> keep only overlap slice -> delete full tensor
  - Saves ~1GB by not holding 6 full voxel tensors simultaneously
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transforms import axis_angle_to_matrix

from networks.blocks import conv2d, conv1d
from utils.visualize import aug_depth_params
from utils.torch_utils import format_size, track_memory

import math
import torch

import gc

class CVT3DNet(nn.Module):
    """
    Surround-view fusion module that estimates a single 3D feature using surround-view images.
    """
    def __init__(self, 
                 cfg, 
                 feat_in_dim=128, 
                 feat_out_dim=[64, 128, 256, 256],
                 feat_size=[24, 40]):
        super(CVT3DNet, self).__init__()
        self.read_config(cfg) 
        self.img_h, self.img_w = feat_size
        self.eps = 1e-6

        self.num_layers = len(self.intermediate_layer_idx)

        self.in_dim = feat_in_dim
        self.out_dim = feat_out_dim
    
        self.build_voxel_and_pixel_girds()

        self.voxel_feat_dim = self.voxel_pre_dim[-1]  # 64

        self.cam_embed_dim = 8
        self.cam_embed = nn.Embedding(self.num_cams, self.cam_embed_dim)

        self.pos_embed_dim = 8
        self.pos_encoder = nn.Linear(4, self.pos_embed_dim)

        self.cam_token_in_dim = feat_in_dim + self.pos_embed_dim + self.cam_embed_dim

        self.cam_value_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.cam_token_in_dim),
                nn.Linear(self.cam_token_in_dim, self.voxel_feat_dim),
            )
            for _ in range(self.num_layers)
        ])
        self.cam_score_proj = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.cam_token_in_dim),
                nn.Linear(self.cam_token_in_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),  
            )
            for _ in range(self.num_layers)
        ])

        self.overlap_chunk = getattr(self, "overlap_chunk", 60000)

        gn_groups = getattr(self, "smooth_gn_groups", 8)
        self.post_smooth = nn.Sequential(
            nn.Conv3d(self.voxel_feat_dim, self.voxel_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=self.voxel_feat_dim),
            nn.GELU(),
            nn.Conv3d(self.voxel_feat_dim, self.voxel_feat_dim, kernel_size=3, padding=1, bias=False),
        )

        self.depth_attn = nn.Sequential(
            nn.Conv3d(self.voxel_feat_dim, self.voxel_feat_dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(self.voxel_feat_dim // 4, 1, kernel_size=1, bias=True),
        )

        self.reduce_dim = nn.ModuleList([
            nn.Sequential(
                *conv2d(self.voxel_feat_dim, out_dim, kernel_size=3, stride=1).children(),
                *conv2d(out_dim, out_dim, kernel_size=3, stride=1).children(),
            )
            for out_dim in feat_out_dim
        ])

    
    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def type_check(self, sample_tensor):
        d_dtype, d_device = sample_tensor.dtype, sample_tensor.device
        if (self.voxel_pts.dtype != d_dtype) or (self.voxel_pts.device != d_device):
            self.voxel_pts = self.voxel_pts.to(device=d_device, dtype=d_dtype)
            self.pixel_grid = self.pixel_grid.to(device=d_device, dtype=d_dtype)
            self.depth_grid = self.depth_grid.to(device=d_device, dtype=d_dtype)
            self.pixel_ones = self.pixel_ones.to(device=d_device, dtype=d_dtype) 
        return d_dtype, d_device

    def create_voxel_grid(self, str_p, end_p, v_size):
        grids = [torch.linspace(str_p[i], end_p[i], v_size[i]) for i in range(3)]
        x_dim, y_dim, z_dim = v_size
        grids[0] = grids[0].view(1, 1, 1, 1, x_dim)
        grids[1] = grids[1].view(1, 1, 1, y_dim, 1)
        grids[2] = grids[2].view(1, 1, z_dim, 1, 1)
        grids = [grid.expand(self.batch_size, 1, z_dim, y_dim, x_dim) for grid in grids]
        return torch.cat(grids, 1)

    def create_pixel_grid(self, batch_size, height, width):
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, axis=0).unsqueeze(0).view(1, 2, height * width)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        ones = torch.ones(batch_size, 1, height * width)
        pix_coords = torch.cat([pix_coords, ones], 1)
        return pix_coords
    
    def create_depth_grid(self, batch_size, n_pixels, n_depth_bins, depth_bins):
        depth_layers = []
        for d in depth_bins:  
            depth_layer = torch.ones((1, n_pixels)) * d 
            depth_layers.append(depth_layer)  
        depth_layers = torch.cat(depth_layers, dim=0).view(1, 1, n_depth_bins, n_pixels)
        depth_layers = depth_layers.expand(batch_size, 3, n_depth_bins, n_pixels)
        return depth_layers
    
    def build_voxel_and_pixel_girds(self):
        self.voxel_end_p = [self.voxel_str_p[i] + self.voxel_unit_size[i] * (self.voxel_size[i] - 1) for i in range(3)]
        voxel_grid = self.create_voxel_grid(self.voxel_str_p, self.voxel_end_p, self.voxel_size)       
        b, _, self.z_dim, self.y_dim, self.x_dim = voxel_grid.size()
        self.n_voxels = self.z_dim * self.y_dim * self.x_dim  
        ones = torch.ones(self.batch_size, 1, self.n_voxels)
        self.voxel_pts = torch.cat([voxel_grid.view(b, 3, self.n_voxels), ones], dim=1) 
        self.num_pix = self.img_h * self.img_w  
        self.pixel_grid = self.create_pixel_grid(self.batch_size, self.img_h, self.img_w)
        self.pixel_ones = torch.ones(self.batch_size, 1, self.proj_d_bins, self.num_pix)  
        depth_bins = torch.linspace(self.proj_d_str, self.proj_d_end, self.proj_d_bins)
        self.depth_grid = self.create_depth_grid(self.batch_size, self.num_pix, self.proj_d_bins, depth_bins)    

    def calculate_sample_pixel_coords(self, K, v_pts, w_dim, h_dim):       
        cam_points = torch.matmul(K[:, :3, :3], v_pts) 
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        if not torch.all(torch.isfinite(pix_coords)):
            pix_coords = torch.clamp(pix_coords, min=-w_dim*2, max=w_dim*2)
        pix_coords = pix_coords.view(self.batch_size, 2, self.n_voxels, 1)
        pix_coords = pix_coords.permute(0, 2, 3, 1)   
        pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (w_dim - 1)
        pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (h_dim - 1)
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def calculate_valid_mask(self, mask_img, pix_coords, v_pts_local):
        mask_selfocc = (F.grid_sample(mask_img, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True) > 0.5)
        mask_depth = (v_pts_local[:, 2:3, :] > 0) 
        pix_coords_mask = pix_coords.permute(0, 3, 1, 2)
        mask_oob = ~(torch.logical_or(pix_coords_mask > 1, pix_coords_mask < -1).sum(dim=1, keepdim=True) > 0)
        valid_mask = mask_selfocc.squeeze(-1) * mask_depth * mask_oob.squeeze(-1)
        return valid_mask
    
    def precompute_voxel_projection(self, input_mask, intrinsics, extrinsics_inv):
        pix_coords_list = []
        voxel_mask_list = []
        x_norms = []
        y_norms = []
        z_norms = []
        depth_norms = []     
        for cam in range(self.num_cams):
            mask_img = input_mask[:, cam, ...]
            mask_img = F.interpolate(mask_img, [self.img_h, self.img_w], mode='bilinear', align_corners=True)
            ext_inv_mat = extrinsics_inv[:, cam, :3, :].to(dtype=self.dtype)
            v_pts_local = torch.matmul(ext_inv_mat, self.voxel_pts)
            K_mat = intrinsics[:, cam, :, :].to(dtype=self.dtype)
            pix_coords = self.calculate_sample_pixel_coords(K_mat, v_pts_local, self.img_w, self.img_h)
            valid_mask = self.calculate_valid_mask(mask_img, pix_coords, v_pts_local)

            depth = v_pts_local[:, 2:3, :].clamp(min=self.proj_d_str, max=self.proj_d_end)
            log_d = torch.log(depth + 1e-6)
            log_min = math.log(self.proj_d_str + 1e-6)
            log_max = math.log(self.proj_d_end + 1e-6)
            depth_norm = (log_d - log_min) / (log_max - log_min + 1e-6)
            depth_norm = depth_norm.clamp(0.0, 1.0)

            v_global = self.voxel_pts[:, :3, :]
            x = v_global[:, 0:1, :]
            y = v_global[:, 1:2, :]
            z = v_global[:, 2:3, :]
            x_min, x_max = self.voxel_str_p[0], self.voxel_end_p[0]
            y_min, y_max = self.voxel_str_p[1], self.voxel_end_p[1]
            z_min, z_max = self.voxel_str_p[2], self.voxel_end_p[2]
            x_norm = ((x - x_min) / (x_max - x_min + 1e-6)) * 2 - 1
            y_norm = ((y - y_min) / (y_max - y_min + 1e-6)) * 2 - 1
            z_norm = ((z - z_min) / (z_max - z_min + 1e-6)) * 2 - 1

            pix_coords_list.append(pix_coords)
            depth_norms.append(depth_norm)
            voxel_mask_list.append(valid_mask)
            x_norms.append(x_norm)
            y_norms.append(y_norm)
            z_norms.append(z_norm)

        # =============================================
        # 新增: 预计算 overlap/non-overlap mask 和 overlap indices
        # 这些在所有 feature level 之间共享
        # =============================================
        valid_masks = torch.stack([m.squeeze(1) for m in voxel_mask_list], dim=1)  # [B, K, N]
        voxel_mask_count = valid_masks.float().sum(dim=1)  # [B, N]
        non_overlap_mask = (voxel_mask_count == 1)   # [B, N]
        overlap_mask = (voxel_mask_count >= 2)        # [B, N]
        
        B = non_overlap_mask.shape[0]
        N = self.n_voxels
        BN = B * N
        
        # 预计算 overlap 的 flat indices（所有level共享）
        ov_flat = overlap_mask.reshape(-1)  # [B*N]
        ov_indices = torch.nonzero(ov_flat, as_tuple=False).squeeze(1) if ov_flat.any() else torch.tensor([], dtype=torch.long, device=ov_flat.device)
        
        per_cam = {
            "pix_coords": pix_coords_list,
            "valid_mask": voxel_mask_list,
            "x_norm": x_norms,
            "y_norm": y_norms,           
            "z_norm": z_norms,
            "depth_norm": depth_norms,
            # 新增预计算结果
            "non_overlap_mask": non_overlap_mask,
            "overlap_mask": overlap_mask,
            "ov_indices": ov_indices,
        }
        return per_cam

    def backproject_into_voxel(self, level_idx, feats_agg, per_cam):
        """
        Memory-optimized version:
        对每个相机：warp → 立刻处理non-overlap scatter → 只保留overlap部分 → 释放完整tensor
        
        显存节省: 从 6×[B,Cin,N] 降到 峰值1×[B,Cin,N] + 6×[M,Cin]
        其中 M = overlap体素数 ≈ 15-20% of N
        """
        B, K, C, H, W = feats_agg.shape
        N = self.n_voxels
        BN = B * N
        device = feats_agg.device

        cam_ids = torch.arange(self.num_cams, device=device).long()
        cam_embed_base = self.cam_embed(cam_ids).to(dtype=self.dtype)  # [K, 8]

        # 从预计算结果中取出 masks 和 indices
        non_overlap_mask = per_cam["non_overlap_mask"]  # [B, N]
        overlap_mask = per_cam["overlap_mask"]          # [B, N]
        ov_indices = per_cam["ov_indices"]              # [M_total] flat indices
        M_total = ov_indices.numel()

        # 输出初始化
        out = feats_agg.new_zeros((B, self.voxel_feat_dim, N))
        out_bn = out.permute(0, 2, 1).reshape(BN, self.voxel_feat_dim)  # [B*N, 64]

        # overlap 部分的稀疏特征存储: 只存 [M, Cin] 而非 [B, Cin, N]
        Cin = C + self.pos_embed_dim  # 128 + 8 = 136
        ov_feat_list = []    # 6 × [M_total, Cin]
        ov_valid_list = []   # 6 × [M_total] bool

        # =============================================
        # 逐相机处理: warp → non-overlap scatter → 保留overlap slice → 释放
        # =============================================
        for cam in range(self.num_cams):
            feats_img = feats_agg[:, cam, ...]  # [B, C, H, W]
            pix_coords = per_cam["pix_coords"][cam]
            valid_mask = per_cam["valid_mask"][cam].squeeze(1)  # [B, N]

            x_norm     = per_cam["x_norm"][cam]
            y_norm     = per_cam["y_norm"][cam]
            z_norm     = per_cam["z_norm"][cam]
            depth_norm = per_cam["depth_norm"][cam]

            # Warp: [B, C, N]
            feat_warped = F.grid_sample(
                feats_img, pix_coords,
                mode='bilinear', padding_mode='zeros', align_corners=True
            ).squeeze(-1)  # [B, C, N]

            # Positional encoding: [B, pos_embed_dim, N]
            pos_raw = torch.cat([x_norm, y_norm, z_norm, depth_norm], dim=1)  # [B, 4, N]
            pos_encoded = self.pos_encoder(pos_raw.permute(0, 2, 1)).permute(0, 2, 1)  # [B, 8, N]

            # 拼接: [B, Cin, N]
            feat_full = torch.cat([feat_warped, pos_encoded], dim=1)
            feat_full = feat_full * valid_mask.unsqueeze(1).float()

            del feat_warped, pos_encoded, pos_raw  # 释放中间变量

            # -----------------------------------------
            # 1) Non-overlap: 立刻 scatter 到 out，不存储
            # -----------------------------------------
            pick = (valid_mask & non_overlap_mask).reshape(-1)  # [B*N]
            if pick.any():
                idx = torch.nonzero(pick, as_tuple=False).squeeze(1)  # [M_non]
                feat_bn = feat_full.permute(0, 2, 1).reshape(BN, -1)[idx]  # [M_non, Cin]
                cam_emb = cam_embed_base[cam].view(1, -1).expand(feat_bn.size(0), -1)  # [M_non, 8]
                token = torch.cat([feat_bn, cam_emb], dim=1)  # [M_non, Cin+8]
                v = self.cam_value_proj[level_idx](token)      # [M_non, 64]
                out_bn[idx] = v
                del feat_bn, cam_emb, token, v

            # -----------------------------------------
            # 2) Overlap: 只提取 overlap 体素的特征（很小）
            # -----------------------------------------
            if M_total > 0:
                ov_feat = feat_full.permute(0, 2, 1).reshape(BN, -1)[ov_indices]  # [M_total, Cin]
                ov_valid = valid_mask.reshape(-1)[ov_indices]                      # [M_total] bool
                ov_feat = ov_feat * ov_valid.unsqueeze(1).float()
                ov_feat_list.append(ov_feat)
                ov_valid_list.append(ov_valid)
            
            # -----------------------------------------
            # 3) 释放完整的 [B, Cin, N] tensor
            # -----------------------------------------
            del feat_full

        # =============================================
        # Overlap CVA: 在稀疏的 overlap 体素上做 attention
        # =============================================
        if M_total > 0:
            chunk = int(self.overlap_chunk)
            for s in range(0, M_total, chunk):
                e = min(s + chunk, M_total)
                ov_idx = ov_indices[s:e]  # global flat indices for scatter
                m = e - s

                vals = []
                scrs = []
                vmsk = []

                for cam in range(self.num_cams):
                    cam_feat = ov_feat_list[cam][s:e]      # [m, Cin]
                    cam_valid = ov_valid_list[cam][s:e]     # [m] bool

                    cam_emb = cam_embed_base[cam].view(1, -1).expand(m, -1)  # [m, 8]
                    token = torch.cat([cam_feat, cam_emb], dim=1)  # [m, Cin+8]

                    v = self.cam_value_proj[level_idx](token)          # [m, 64]
                    sc = self.cam_score_proj[level_idx](token).squeeze(-1)  # [m]

                    sc = sc.masked_fill(~cam_valid, -1e9)
                    v = v * cam_valid.unsqueeze(1).float()

                    vals.append(v)
                    scrs.append(sc)
                    vmsk.append(cam_valid)

                V = torch.stack(vals, dim=1)   # [m, K, 64]
                S = torch.stack(scrs, dim=1)   # [m, K]
                Mk = torch.stack(vmsk, dim=1)  # [m, K] bool

                attn = F.softmax(S, dim=1)
                attn = attn * Mk.float()
                attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)

                fused = (attn.unsqueeze(-1) * V).sum(dim=1)  # [m, 64]
                out_bn[ov_idx] = fused

            # 释放 overlap 特征
            del ov_feat_list, ov_valid_list

        # Reshape back
        out = out_bn.view(B, N, self.voxel_feat_dim).permute(0, 2, 1).contiguous()  # [B, 64, N]

        # =============================================
        # 3D Voxel Smoothing (residual)
        # =============================================
        v3d = out.view(B, self.voxel_feat_dim, self.z_dim, self.y_dim, self.x_dim)
        v3d = v3d + self.post_smooth(v3d)
        out = v3d.view(B, self.voxel_feat_dim, N).contiguous()

        return out

    def project_voxel_into_image(self, level_idx, voxel_feat, inv_K, extrinsics):
        b, feat_dim, _ = voxel_feat.size()
        voxel_feat = voxel_feat.view(b, feat_dim, self.z_dim, self.y_dim, self.x_dim) 
        
        proj_feats = []
        for cam in range(self.num_cams):
            cam_points = torch.matmul(inv_K[:, cam, :3, :3], self.pixel_grid)
            cam_points = self.depth_grid * cam_points.view(self.batch_size, 3, 1, self.num_pix)
            cam_points = torch.cat([cam_points, self.pixel_ones], dim=1)
            cam_points = cam_points.view(self.batch_size, 4, -1)
            points = torch.matmul(extrinsics[:, cam, :3, :], cam_points)
            grid = points.permute(0, 2, 1) 
            for i in range(3):
                v_length = self.voxel_end_p[i] - self.voxel_str_p[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_str_p[i]) / v_length * 2. - 1.
            grid = grid.view(self.batch_size, self.proj_d_bins, self.img_h, self.img_w, 3)            
            proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            depth_logits = self.depth_attn(proj_feat)
            depth_weights = F.softmax(depth_logits, dim=2)
            proj_feat = (proj_feat * depth_weights).sum(dim=2)
            
            proj_feat = self.reduce_dim[level_idx](proj_feat)
            proj_feats.append(proj_feat)
        return proj_feats

    def scale_intrinsics(self, K, down_scale):
        K_feat = K.clone()
        K_feat[..., 0, 0] /= down_scale     
        K_feat[..., 1, 1] /= down_scale     
        K_feat[..., 0, 2] /= down_scale   
        K_feat[..., 1, 2] /= down_scale    
        inv_K_feat = torch.inverse(K_feat)
        return K_feat, inv_K_feat

    def forward(self, inputs, feats_agg):
        mask = inputs['mask']
        K0 = inputs[('K', 0)]
        extrinsics = inputs['extrinsics']
        extrinsics_inv = inputs['extrinsics_inv']
        K, inv_K = self.scale_intrinsics(K0, down_scale=self.dino_patch_size)
        fusion_dict = {}
        for cam in range(self.num_cams):
            fusion_dict[('cam', cam)] = {}

        sample_tensor = feats_agg[0][0, 0, ...]
        self.dtype, self.device = self.type_check(sample_tensor)

        K, inv_K = K.to(dtype=self.dtype), inv_K.to(dtype=self.dtype)
        extrinsics = extrinsics.to(dtype=self.dtype)
        extrinsics_inv = extrinsics_inv.to(dtype=self.dtype)

        with torch.no_grad():
            per_cam = self.precompute_voxel_projection(mask, K, extrinsics_inv)
        
        proj_feats_list = []
        for i, f in enumerate(feats_agg):
            voxel_feat = self.backproject_into_voxel(i, f, per_cam)
            proj_feats = self.project_voxel_into_image(i, voxel_feat, inv_K, extrinsics)
            proj_feats = torch.stack(proj_feats, 1)
            proj_feats_list.append(proj_feats)
            
        fusion_dict['proj_feat'] = proj_feats_list
        return fusion_dict


# ============================================================
# Benchmark (和原版相同的测试代码)
# ============================================================
BATCH_SIZE = 1
NUM_CAMS = 6
IMG_H, IMG_W = 384, 640

CVT3DNET_CFG = {
    'model': {
        'batch_size': BATCH_SIZE,
        'num_cams': NUM_CAMS,
    },
    'depth_fusion': {
        'voxel_str_p': [-50.0, -50.0, -1.0],
        'voxel_unit_size': [1.0, 1.0, 0.75],
        'voxel_size': [100, 100, 20],
        'voxel_pre_dim': [64],
        'proj_d_bins': 50,
        'proj_d_str': 1.0,
        'proj_d_end': 75.0,
        'dino_patch_size': 16,
        'intermediate_layer_idx': [2, 5, 8, 11],
    },
}

def get_memory_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def make_intrinsics(B, device):
    K = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, NUM_CAMS, 1, 1)
    K[:, :, 0, 0] = 600.0
    K[:, :, 1, 1] = 600.0
    K[:, :, 0, 2] = IMG_W / 2
    K[:, :, 1, 2] = IMG_H / 2
    return K

def make_extrinsics(B, device):
    ext = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, NUM_CAMS, 1, 1)
    angles = [0, 60, -60, 120, -120, 180]
    for i, ang in enumerate(angles):
        rad = math.radians(ang)
        ext[:, i, 0, 0] = math.cos(rad)
        ext[:, i, 0, 2] = math.sin(rad)
        ext[:, i, 2, 0] = -math.sin(rad)
        ext[:, i, 2, 2] = math.cos(rad)
        ext[:, i, 0, 3] = math.sin(rad) * 2
        ext[:, i, 2, 3] = math.cos(rad) * 2
    return ext

def make_mask(B, device):
    return torch.ones(B, NUM_CAMS, 1, IMG_H, IMG_W, device=device)

def test_cvt3dnet(device):
    print("\n--- Testing CVT3DNet (Hier + CVA @ 1/16, memory-optimized) ---")
    
    B = BATCH_SIZE
    num_levels = 4
    feat_h, feat_w = IMG_H // 16, IMG_W // 16

    model = CVT3DNet(
        CVT3DNET_CFG,
        feat_in_dim=128,
        feat_out_dim=[64, 128, 256, 256],
        feat_size=[feat_h, feat_w],
    ).to(device).train()

    K = make_intrinsics(B, device)
    ext = make_extrinsics(B, device)
    ext_inv = torch.inverse(ext)
    mask = make_mask(B, device)

    inputs = {
        'mask': mask,
        ('K', 0): K,
        'extrinsics': ext,
        'extrinsics_inv': ext_inv,
    }

    feats_agg = [
        torch.randn(B, NUM_CAMS, 128, feat_h, feat_w, device=device)
        for _ in range(num_levels)
    ]

    # Warmup
    reset_memory()
    with torch.no_grad():
        _ = model(inputs, feats_agg)

    # Measure
    reset_memory()
    out = model(inputs, feats_agg)
    proj_feats = out['proj_feat']
    loss = sum(p.sum() for p in proj_feats)
    loss.backward()
    mem = get_memory_mb()

    print(f"  Peak Memory: {mem:.1f} MB")
    del model, feats_agg, out
    reset_memory()
    return mem

def main():
    assert torch.cuda.is_available(), "Requires CUDA GPU!"
    device = torch.device('cuda')

    print("=" * 55)
    print("CVT3DNet Memory Benchmark (optimized)")
    print(f"  B={BATCH_SIZE}, Cams={NUM_CAMS}")
    print(f"  Image: {IMG_H}x{IMG_W}")
    print(f"  CVT3DNet: 1/16 -> 24x40, C=128, 4 levels")
    print("=" * 55)

    mem = test_cvt3dnet(device)

    print("\n" + "=" * 55)
    print(f"  CVT3DNet (optimized): {mem:.1f} MB")
    print("=" * 55)

if __name__ == "__main__":
    main()