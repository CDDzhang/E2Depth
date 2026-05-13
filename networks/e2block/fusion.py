# Copyright (c) 2023 42dot. All rights reserved.
"""
depth_fusion--version3
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
    
    v3 improvements:
      1. Depth-bin attention replaces heavy 3200→out reduce_dim (kept from v2)
      2. 3D voxel smoothing restored (from v1) for spatial continuity
      3. Lightweight pos_encoder: single Linear(4→8) instead of MLP(4→32→16)
      4. cam_embed_dim = 8 (balanced between v1=4 and v2=16)
    """
    def __init__(self, 
                 cfg, 
                 feat_in_dim=128, 
                 feat_out_dim=[64, 128, 256, 256],
                 feat_size=[24, 40]):
        super(CVT3DNet, self).__init__()
        self.read_config(cfg) 
        self.img_h, self.img_w = feat_size  # [24, 40]
        self.eps = 1e-6

        self.num_layers = len(self.intermediate_layer_idx)

        self.in_dim = feat_in_dim
        self.out_dim = feat_out_dim
    
        self.build_voxel_and_pixel_girds()

        # 体素特征最终维度
        self.voxel_feat_dim = self.voxel_pre_dim[-1]  # 64

        # =============================================
        # cam_embed_dim: 8 (比v1的4稍大，但比v2的16更轻量)
        # =============================================
        self.cam_embed_dim = 8
        self.cam_embed = nn.Embedding(self.num_cams, self.cam_embed_dim)

        # =============================================
        # 改进4: 轻量 Positional encoding
        # 单层线性映射，避免 per-voxel per-camera 的 MLP 开销
        # =============================================
        self.pos_embed_dim = 8
        self.pos_encoder = nn.Linear(4, self.pos_embed_dim)

        # token维度: feat_in_dim + pos_embed_dim + cam_embed_dim
        self.cam_token_in_dim = feat_in_dim + self.pos_embed_dim + self.cam_embed_dim

        # 每一层：用一个 MLP 得到 value（体素特征），一个 MLP 得到 score（注意力权重）
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

        # overlap attention 的 chunk
        self.overlap_chunk = getattr(self, "overlap_chunk", 60000)

        # =============================================
        # 3D Voxel Smoothing (从 v1 恢复)
        # 为 non-overlap 和 overlap 区域都提供局部空间连续性
        # =============================================
        gn_groups = getattr(self, "smooth_gn_groups", 8)
        self.post_smooth = nn.Sequential(
            nn.Conv3d(self.voxel_feat_dim, self.voxel_feat_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=gn_groups, num_channels=self.voxel_feat_dim),
            nn.GELU(),
            nn.Conv3d(self.voxel_feat_dim, self.voxel_feat_dim, kernel_size=3, padding=1, bias=False),
        )

        # =============================================
        # 改进1: Depth-bin attention 替代 heavy reduce_dim
        # grid_sample后: [B, 64, 50, H, W]
        # depth_attn: 学习每个像素关注哪些depth bins → softmax加权求和 → [B, 64, H, W]
        # 然后只需 64→out_dim 的轻量conv
        # =============================================
        self.depth_attn = nn.Sequential(
            nn.Conv3d(self.voxel_feat_dim, self.voxel_feat_dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(self.voxel_feat_dim // 4, 1, kernel_size=1, bias=True),
        )  # output: [B, 1, D, H, W]

        self.reduce_dim = nn.ModuleList([
            nn.Sequential(
                *conv2d(self.voxel_feat_dim, out_dim, kernel_size=3, stride=1).children(),
                *conv2d(out_dim, out_dim, kernel_size=3, stride=1).children(),
            )
            for out_dim in feat_out_dim
        ])  # 64→out_dim, 比原来的 3200→256→256→out_dim 轻量得多

    
    def read_config(self, cfg):
        for attr in cfg.keys():
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def type_check(self, sample_tensor):
        """
        This function checks the type of the tensor, so that all the parameters share same device and dtype.
        """
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

        pix_coords = pix_coords.view(self.batch_size, 2, self.n_voxels, 1)  # b, 2,2e5,1
        pix_coords = pix_coords.permute(0, 2, 3, 1)   
        pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (w_dim - 1)
        pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (h_dim - 1)
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def calculate_valid_mask(self, mask_img, pix_coords, v_pts_local):
        """
        This function creates valid mask in voxel coordinate by projecting self-occlusion mask to 3D voxel coords. 
        """
        # compute validity mask, [b, 1, n_voxels, 1]
        mask_selfocc = (F.grid_sample(mask_img, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True) > 0.5)
        # discard points behind the camera, [b, 1, n_voxels]
        mask_depth = (v_pts_local[:, 2:3, :] > 0) 
        # compute validity mask, [b, 1, n_voxels, 1]
        pix_coords_mask = pix_coords.permute(0, 3, 1, 2)
        mask_oob = ~(torch.logical_or(pix_coords_mask > 1, pix_coords_mask < -1).sum(dim=1, keepdim=True) > 0)
        valid_mask = mask_selfocc.squeeze(-1) * mask_depth * mask_oob.squeeze(-1)
        return valid_mask
    
    def precompute_voxel_projection(self, input_mask, intrinsics, extrinsics_inv):
        pix_coords_list = []  # 体素投到相机后的 2D 像素坐标
        voxel_mask_list = []  # 该 voxel 是否被此相机"看到"

        x_norms = []
        y_norms = []
        z_norms = []
        depth_norms = []     
        for cam in range(self.num_cams):
            mask_img = input_mask[:, cam, ...]
            mask_img = F.interpolate(mask_img, [self.img_h, self.img_w], mode='bilinear', align_corners=True)

            ext_inv_mat = extrinsics_inv[:, cam, :3, :].to(dtype=self.dtype)  # [b,3,4] 
            v_pts_local = torch.matmul(ext_inv_mat, self.voxel_pts)

            K_mat = intrinsics[:, cam, :, :].to(dtype=self.dtype)
            pix_coords = self.calculate_sample_pixel_coords(K_mat, v_pts_local, self.img_w, self.img_h)

            valid_mask = self.calculate_valid_mask(mask_img, pix_coords, v_pts_local)

            depth = v_pts_local[:, 2:3, :].clamp(min=self.proj_d_str, max=self.proj_d_end)  # [B,1,N]
            log_d = torch.log(depth + 1e-6)
            log_min = math.log(self.proj_d_str + 1e-6)   # log(1)
            log_max = math.log(self.proj_d_end + 1e-6)   # log(75)
            depth_norm = (log_d - log_min) / (log_max - log_min + 1e-6)
            depth_norm = depth_norm.clamp(0.0, 1.0)

            v_global = self.voxel_pts[:, :3, :]  # [B, 3, N]
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

        per_cam = {
            "pix_coords": pix_coords_list,   # [B, n_voxels, 1, 2] * 6 
            "valid_mask": voxel_mask_list,   # [B, 1, n_voxels] * 6 
            "x_norm": x_norms,           # [B, 1, n_voxels] * 6
            "y_norm": y_norms,           
            "z_norm": z_norms,           # [B, 1, n_voxels] * 6
            "depth_norm": depth_norms
        }
        return per_cam

    def backproject_into_voxel(self, level_idx, feats_agg, per_cam):
        """
        feats_agg: [B, K, C, H, W]
        return:   [B, C_voxel(=64), N]
        
        改进4: 位置编码通过 pos_encoder MLP 映射到更高维表示
        """
        B, K, C, H, W = feats_agg.shape
        N = self.n_voxels
        device = feats_agg.device

        # per-cam embed (改进3: 现在是16维)
        cam_ids = torch.arange(self.num_cams, device=device).long()
        cam_embed_base = self.cam_embed(cam_ids).to(dtype=self.dtype)  # [K, 16]

        # 先做 warp + 拼 pos encoding，保存每个 cam 的 [B, Cin, N] 和 valid [B,N]
        voxel_feat_list = []
        valid_list = []

        for cam in range(self.num_cams):
            feats_img = feats_agg[:, cam, ...]  # [B, C, H, W]
            pix_coords = per_cam["pix_coords"][cam]  # [B, N, 1, 2]
            valid_mask = per_cam["valid_mask"][cam].squeeze(1)  # [B, N] bool/0-1

            x_norm     = per_cam["x_norm"][cam]       # [B,1,N]
            y_norm     = per_cam["y_norm"][cam]
            z_norm     = per_cam["z_norm"][cam]
            depth_norm = per_cam["depth_norm"][cam]

            feat_warped = F.grid_sample(
                feats_img, pix_coords,
                mode='bilinear', padding_mode='zeros', align_corners=True
            )  # [B, C, N, 1]

            feat_warped = feat_warped.squeeze(-1)  # [B, C, N]

            # =============================================
            # 改进4: 用 pos_encoder MLP 替代直接拼接原始坐标
            # =============================================
            pos_raw = torch.cat([x_norm, y_norm, z_norm, depth_norm], dim=1)  # [B, 4, N]
            pos_raw = pos_raw.permute(0, 2, 1)  # [B, N, 4]
            pos_encoded = self.pos_encoder(pos_raw)  # [B, N, 16]
            pos_encoded = pos_encoded.permute(0, 2, 1)  # [B, 16, N]

            feat_warped = torch.cat([feat_warped, pos_encoded], dim=1)  # [B, C+16, N]
            feat_warped = feat_warped * valid_mask.unsqueeze(1).float()  # mask

            voxel_feat_list.append(feat_warped)  # [B, Cin, N], Cin = C + 16
            valid_list.append(valid_mask)        # [B, N]

        # valid_masks: [B, K, N]
        valid_masks = torch.stack(valid_list, dim=1).contiguous()
        voxel_mask_count = valid_masks.sum(dim=1)  # [B, N]

        non_overlap_mask = (voxel_mask_count == 1)
        overlap_mask     = (voxel_mask_count >= 2)

        # 输出初始化
        out = feats_agg.new_zeros((B, self.voxel_feat_dim, N))

        # -------------------------
        # 1) non-overlap: 直接取唯一可见相机的 value（无需 attention）
        # -------------------------
        BN = B * N
        non_flat = non_overlap_mask.reshape(-1)  # [B*N]
        if non_flat.any():
            for cam in range(self.num_cams):
                cam_valid = valid_list[cam]                    # [B,N]
                pick = (cam_valid & non_overlap_mask).reshape(-1)  # [B*N]
                if not pick.any():
                    continue

                idx = torch.nonzero(pick, as_tuple=False).squeeze(1)  # [M]
                # gather feat: [B,Cin,N] -> [B*N,Cin] -> [M,Cin]
                feat_bn = voxel_feat_list[cam].permute(0, 2, 1).reshape(BN, -1)[idx]  # [M, Cin]
                cam_emb = cam_embed_base[cam].view(1, -1).expand(feat_bn.size(0), -1)  # [M, D_cam]
                token = torch.cat([feat_bn, cam_emb], dim=1)  # [M, Cin + D_cam]

                v = self.cam_value_proj[level_idx](token)     # [M, 64]

                # scatter 回 out
                out_bn = out.permute(0, 2, 1).reshape(BN, self.voxel_feat_dim)  # [B*N,64]
                out_bn[idx] = v
                out = out_bn.view(B, N, self.voxel_feat_dim).permute(0, 2, 1).contiguous()

        # -------------------------
        # 2) overlap: 只在 overlap voxel 上做 attention（可 chunk）
        # -------------------------
        ov_flat = overlap_mask.reshape(-1)  # [B*N]
        if ov_flat.any():
            ov_idx_all = torch.nonzero(ov_flat, as_tuple=False).squeeze(1)  # [M_all]
            M_all = ov_idx_all.numel()
            chunk = int(self.overlap_chunk)

            out_bn = out.permute(0, 2, 1).reshape(BN, self.voxel_feat_dim)  # [B*N,64]

            for s in range(0, M_all, chunk):
                ov_idx = ov_idx_all[s:s+chunk]  # [m]

                vals = []
                scrs = []
                vmsk = []

                for cam in range(self.num_cams):
                    cam_valid_flat = valid_list[cam].reshape(-1)[ov_idx]  # [m] bool
                    feat_bn = voxel_feat_list[cam].permute(0, 2, 1).reshape(BN, -1)[ov_idx]  # [m,Cin]
                    feat_bn = feat_bn * cam_valid_flat.unsqueeze(1).float()

                    cam_emb = cam_embed_base[cam].view(1, -1).expand(feat_bn.size(0), -1)  # [m,D_cam]
                    token = torch.cat([feat_bn, cam_emb], dim=1)  # [m, Cin+D_cam]

                    v = self.cam_value_proj[level_idx](token)  # [m,64]
                    sc = self.cam_score_proj[level_idx](token).squeeze(-1)  # [m]

                    # mask invalid
                    sc = sc.masked_fill(~cam_valid_flat, -1e9)
                    v = v * cam_valid_flat.unsqueeze(1).float()

                    vals.append(v)
                    scrs.append(sc)
                    vmsk.append(cam_valid_flat)

                V = torch.stack(vals, dim=1)   # [m,K,64]
                S = torch.stack(scrs, dim=1)   # [m,K]
                M = torch.stack(vmsk, dim=1)   # [m,K] bool

                attn = F.softmax(S, dim=1)     # [m,K]
                attn = attn * M.float()
                attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-6)

                fused = (attn.unsqueeze(-1) * V).sum(dim=1)  # [m,64]

                out_bn[ov_idx] = fused

            out = out_bn.view(B, N, self.voxel_feat_dim).permute(0, 2, 1).contiguous()  # [B,64,N]

        # -------------------------
        # 3) continuity: Voxel 3D Smoothing (residual)
        # -------------------------
        v3d = out.view(B, self.voxel_feat_dim, self.z_dim, self.y_dim, self.x_dim)
        v3d = v3d + self.post_smooth(v3d)
        out = v3d.view(B, self.voxel_feat_dim, N).contiguous()

        return out

    def project_voxel_into_image(self, level_idx, voxel_feat, inv_K, extrinsics):
        """
        改进1: 用 depth-bin attention 替代 heavy reduce_dim
        
        原来: grid_sample → [B, 64, 50, H, W] → reshape [B, 3200, H, W] → 3层conv(3200→256→256→out)
        现在: grid_sample → [B, 64, 50, H, W] → depth_attn → [B, 64, H, W] → 2层conv(64→out)
        """
        b, feat_dim, _ = voxel_feat.size()
        voxel_feat = voxel_feat.view(b, feat_dim, self.z_dim, self.y_dim, self.x_dim) 
        
        proj_feats = []
        for cam in range(self.num_cams):
            # construct 3D point grid for each view
            cam_points = torch.matmul(inv_K[:, cam, :3, :3], self.pixel_grid)
            cam_points = self.depth_grid * cam_points.view(self.batch_size, 3, 1, self.num_pix)
            cam_points = torch.cat([cam_points, self.pixel_ones], dim=1) # [b, 4, n_depthbins, n_pixels]
            cam_points = cam_points.view(self.batch_size, 4, -1) # [b, 4, n_depthbins * n_pixels]
            
            # apply extrinsic: local 3D point -> global coordinate
            points = torch.matmul(extrinsics[:, cam, :3, :], cam_points)

            # 3D grid_sample
            grid = points.permute(0, 2, 1) 
            
            for i in range(3):
                v_length = self.voxel_end_p[i] - self.voxel_str_p[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_str_p[i]) / v_length * 2. - 1.
                
            grid = grid.view(self.batch_size, self.proj_d_bins, self.img_h, self.img_w, 3)            
            proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            # proj_feat: [B, 64, 50, H, W]

            # =============================================
            # 改进1: Depth-bin attention
            # =============================================
            depth_logits = self.depth_attn(proj_feat)          # [B, 1, 50, H, W]
            depth_weights = F.softmax(depth_logits, dim=2)     # softmax over depth bins
            proj_feat = (proj_feat * depth_weights).sum(dim=2) # [B, 64, H, W]
            
            # 轻量 conv: 64 → out_dim
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
        """
        inputs: input dict including camera intrinsics and extrinsics
        feats_agg: multi-layers of surrounding view features, [B, n_cam, C, H, W] * num_layers
        
        保持原始的每层独立 pipeline：backproject → project, 跨层交互交给 decoder
        """
        mask = inputs['mask']
        K0 = inputs[('K', 0)]
        extrinsics = inputs['extrinsics']
        extrinsics_inv = inputs['extrinsics_inv']
        K, inv_K = self.scale_intrinsics(K0, down_scale=self.dino_patch_size)
        fusion_dict = {}
        for cam in range(self.num_cams):
            fusion_dict[('cam', cam)] = {}

        # device, dtype check
        sample_tensor = feats_agg[0][0, 0, ...]
        self.dtype, self.device = self.type_check(sample_tensor)

        K, inv_K = K.to(dtype=self.dtype), inv_K.to(dtype=self.dtype)
        extrinsics = extrinsics.to(dtype=self.dtype)
        extrinsics_inv = extrinsics_inv.to(dtype=self.dtype)
        mem_stats = {}
        with torch.no_grad():
            per_cam = self.precompute_voxel_projection(mask, K, extrinsics_inv)
        
        proj_feats_list = []
        for i, f in enumerate(feats_agg):
            voxel_feat = self.backproject_into_voxel(i, f, per_cam)
            proj_feats = self.project_voxel_into_image(i, voxel_feat, inv_K, extrinsics)
            proj_feats = torch.stack(proj_feats, 1) # [B, n_cam, fusion_out_dim, 24, 40]
            proj_feats_list.append(proj_feats)
            
        fusion_dict['proj_feat'] = proj_feats_list
        return fusion_dict




BATCH_SIZE = 2
NUM_CAMS = 6
IMG_H, IMG_W = 384, 640

# CVT3DNet config dict
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
        'intermediate_layer_idx': [2, 5, 8, 11],  # 4 levels
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
    """生成合理的相机内参 [B, 6, 4, 4]"""
    K = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, NUM_CAMS, 1, 1)
    fx, fy = 600.0, 600.0
    cx, cy = IMG_W / 2, IMG_H / 2
    K[:, :, 0, 0] = fx
    K[:, :, 1, 1] = fy
    K[:, :, 0, 2] = cx
    K[:, :, 1, 2] = cy
    return K
 
def make_extrinsics(B, device):
    """生成6个相机的外参 [B, 6, 4, 4]，模拟环视相机"""
    import math
    ext = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, NUM_CAMS, 1, 1)
    angles = [0, 60, -60, 120, -120, 180]  # 6个相机朝向
    for i, ang in enumerate(angles):
        rad = math.radians(ang)
        ext[:, i, 0, 0] = math.cos(rad)
        ext[:, i, 0, 2] = math.sin(rad)
        ext[:, i, 2, 0] = -math.sin(rad)
        ext[:, i, 2, 2] = math.cos(rad)
        # 加一点平移
        ext[:, i, 0, 3] = math.sin(rad) * 2
        ext[:, i, 2, 3] = math.cos(rad) * 2
    return ext
 
def make_mask(B, device):
    """自遮挡mask [B, 6, 1, H, W]"""
    return torch.ones(B, NUM_CAMS, 1, IMG_H, IMG_W, device=device)

def test_cvt3dnet(device):
    print("\n--- Testing CVT3DNet (Hier + CVA @ 1/16) ---")
    
    B = BATCH_SIZE
    num_levels = 4
    feat_h, feat_w = IMG_H // 16, IMG_W // 16  # 24, 40
    
    model = CVT3DNet(
        CVT3DNET_CFG,
        feat_in_dim=128,
        feat_out_dim=[64, 128, 256, 256],
        feat_size=[feat_h, feat_w],
    ).to(device).train()
    
    # 准备inputs dict
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
    
    # feats_agg: list of 4 levels, each [B, 6, 128, 24, 40]
    feats_agg = [
        torch.randn(B, NUM_CAMS, 128, feat_h, feat_w, device=device)
        for _ in range(num_levels)
    ]
    
    # Warmup
    reset_memory()
    with torch.no_grad():
        _ = model(inputs, feats_agg)
    
    # Measure (forward + backward)
    reset_memory()
    out = model(inputs, feats_agg)
    proj_feats = out['proj_feat']  # list of [B, 6, out_dim, 24, 40]
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
    
    print("=" * 50)
    print("Memory Benchmark: VFNet vs CVT3DNet")
    print(f"  B={BATCH_SIZE}, Cams={NUM_CAMS}")
    print(f"  Image: {IMG_H}x{IMG_W}")
    print(f"  CVT3DNet: 1/16 -> 24x40,  C=128, 4 levels")
    print("=" * 50)
    
    mem_cvt = test_cvt3dnet(device)
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  CVT3D  (Hier+CVA 1/16):  {mem_cvt:8.1f} MB")
    print("=" * 50)
 
 
if __name__ == "__main__":
    main()