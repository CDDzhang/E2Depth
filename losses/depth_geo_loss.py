import torch
import torch.nn.functional as F

from .multi_cam_loss import MultiCamLoss

_REL_CAM_DICT = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1, 5], 4: [2, 5], 5: [3, 4]}


class GeoMultiCamLoss(MultiCamLoss):
    """
    MultiCamLoss + cross-camera geometric depth consistency (no overlap mask),
    optimized for compute:
      1) geo loss computed on downsampled depth (default 1/4)
      2) only compute for selected reference cameras (default [0])
      3) CVCDepth-style margin gating for robustness
    """

    def __init__(self, cfg, rank):
        super(GeoMultiCamLoss, self).__init__(cfg, rank)

        if isinstance(self.geo_ref_cams, int):
            self.geo_ref_cams = [self.geo_ref_cams]

        # safety clamps
        self.geo_min_depth = float(cfg.get("eval", {}).get("eval_min_depth", 1e-3))
        self.geo_max_depth = float(cfg.get("eval", {}).get("eval_max_depth", 200.0))

    @staticmethod
    def _charbonnier(x: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sqrt(x * x + eps * eps)

    @staticmethod
    def _get_3x3(K_or_invK: torch.Tensor) -> torch.Tensor:
        # supports [B,?,4,4] or [B,?,3,3]
        return K_or_invK[..., :3, :3] if K_or_invK.shape[-1] == 4 else K_or_invK

    @staticmethod
    def _scale_K_3x3(K: torch.Tensor, s: float) -> torch.Tensor:
        """
        Scale intrinsics when we downsample image by factor s (s = 1/downsample).
        K: [B,3,3]
        """
        Ks = K.clone()
        Ks[:, 0, 0] *= s  # fx
        Ks[:, 1, 1] *= s  # fy
        Ks[:, 0, 2] *= s  # cx
        Ks[:, 1, 2] *= s  # cy
        return Ks
    
    @staticmethod
    def _scale_invK_3x3(invK: torch.Tensor, s: float) -> torch.Tensor:
        """
        Scale inverse intrinsics for downsampled resolution.
        invK: [B, 3, 3], s = new_size / old_size (e.g. 0.25 for 1/4 downsample)
        """
        invK_s = invK.clone()
        invK_s[:, 0, :] /= s   # 1/fx' = 1/(fx*s), -cx/(fx*s)
        invK_s[:, 1, :] /= s   # 1/fy' = 1/(fy*s), -cy/(fy*s)
        return invK_s

    def _downsample_depth_and_mask(self, depth, mask=None):
        """
        depth: [B,1,H,W]
        mask:  [B,1,H,W] optional
        returns depth_ds, mask_ds at H',W'
        """
        if self.geo_downsample <= 1:
            return depth, mask

        ds = self.geo_downsample
        H, W = depth.shape[-2:]
        H2, W2 = H // ds, W // ds

        depth_ds = F.interpolate(depth, size=(H2, W2), mode='bilinear', align_corners=True)
        if mask is None:
            return depth_ds, None

        # for mask, use nearest to keep binary-ish
        mask_ds = F.interpolate(mask.float(), size=(H2, W2), mode='nearest')
        return depth_ds, mask_ds

    def compute_geo_depth_consistency_pair(self, inputs, outputs, cam: int, cam_p: int, scale: int = 0):
        """
        cam -> cam_p geometric depth consistency (computed on downsampled depth).
        Valid pixels decided only by: z>0 & in-bounds (plus optional dataset mask).
        """

        ref_view = outputs[('cam', cam)]
        tgt_view = outputs[('cam', cam_p)]

        d_c = ref_view[('depth', scale)].clamp(self.geo_min_depth, self.geo_max_depth)   # [B,1,H,W]
        d_p = tgt_view[('depth', scale)].clamp(self.geo_min_depth, self.geo_max_depth)   # [B,1,H,W]
        B, _, H, W = d_c.shape
        device = d_c.device

        # optional dataset masks (NOT overlap mask)
        m_c = inputs['mask'][:, cam, ...].to(device) if (self.geo_use_dataset_mask and 'mask' in inputs) else None
        m_p = inputs['mask'][:, cam_p, ...].to(device) if (self.geo_use_dataset_mask and 'mask' in inputs) else None

        # downsample depth & mask to save compute
        d_c_ds, m_c_ds = self._downsample_depth_and_mask(d_c, m_c)
        d_p_ds, m_p_ds = self._downsample_depth_and_mask(d_p, m_p)

        B, _, Hs, Ws = d_c_ds.shape

        # intrinsics (scaled for downsampled resolution)
        K_full   = inputs[('K', 0)][:, cam_p, ...].to(device)
        invK_full = inputs[('inv_K', 0)][:, cam, ...].to(device)
        K_p = self._get_3x3(K_full)
        invK_c = self._get_3x3(invK_full)

        # scale factor s = Hs/H = 1/downsample (assuming integer)
        s = float(Ws) / float(W)  # should equal Hs/H
        K_p = self._scale_K_3x3(K_p, s)
        invK_c = self._scale_invK_3x3(invK_c, s)

        # extrinsics
        T_c2ego = inputs['extrinsics'][:, cam, ...].to(device)          # [B,4,4]
        T_ego2p = inputs['extrinsics_inv'][:, cam_p, ...].to(device)    # [B,4,4]

        # pixel grid at downsampled resolution
        ys, xs = torch.meshgrid(
            torch.arange(Hs, device=device),
            torch.arange(Ws, device=device),
            indexing='ij'
        )
        xs = xs.float().view(1, 1, Hs, Ws).expand(B, 1, Hs, Ws)
        ys = ys.float().view(1, 1, Hs, Ws).expand(B, 1, Hs, Ws)
        ones = torch.ones_like(xs)

        pix = torch.cat([xs, ys, ones], dim=1).view(B, 3, -1)  # [B,3,HWs]

        rays = invK_c @ pix                      # [B,3,HWs]
        d_flat = d_c_ds.view(B, 1, -1)           # [B,1,HWs]
        Xc = rays * d_flat                       # [B,3,HWs]

        Xc_h = torch.cat([Xc, torch.ones((B, 1, Xc.shape[-1]), device=device)], dim=1)  # [B,4,HWs]

        # cam c -> ego -> cam p
        Xego_h = T_c2ego @ Xc_h
        Xp_h = T_ego2p @ Xego_h
        Xp = Xp_h[:, :3, :]                      # [B,3,HWs]

        x = Xp[:, 0:1, :]
        y = Xp[:, 1:2, :]
        z = Xp[:, 2:3, :]

        # geometry-only validity
        in_front = (z > 1e-6)

        fx = K_p[:, 0:1, 0:1]
        fy = K_p[:, 1:2, 1:2]
        cx = K_p[:, 0:1, 2:3]
        cy = K_p[:, 1:2, 2:3]

        u = fx * (x / z.clamp(min=1e-6)) + cx
        v = fy * (y / z.clamp(min=1e-6)) + cy

        in_bounds = (u >= 0) & (u <= (Ws - 1)) & (v >= 0) & (v <= (Hs - 1))
        valid = (in_front & in_bounds).float()  # [B,1,HWs]

        # optional dataset mask (downsampled)
        if self.geo_use_dataset_mask and m_c_ds is not None and m_p_ds is not None:
            valid = valid * (m_c_ds.view(B, 1, -1) > 0).float() * (m_p_ds.view(B, 1, -1) > 0).float()

        if valid.sum() < 1:
            return torch.tensor(0.0, device=device)

        # sample target depth at projected coords
        u_norm = 2.0 * (u / (Ws - 1.0)) - 1.0
        v_norm = 2.0 * (v / (Hs - 1.0)) - 1.0
        grid = torch.cat([u_norm, v_norm], dim=1)  # [B,2,HWs]
        grid = grid.permute(0, 2, 1).view(B, Hs, Ws, 2)

        d_p_s = F.grid_sample(
            d_p_ds, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        ).view(B, 1, -1)

        valid = valid * (d_p_s > 0).float()
        if valid.sum() < 1:
            return torch.tensor(0.0, device=device)

        # compute robust diff
        if self.geo_use_inv:
            inv_c = 1.0 / d_flat.clamp(min=1e-6)
            inv_p = 1.0 / d_p_s.clamp(min=1e-6)
            diff = inv_c - inv_p
        else:
            diff = d_flat - d_p_s

        err = self._charbonnier(diff, self.geo_charb_eps)

        # margin gating (CVCDepth-style)
        if self.geo_use_margin:
            gate = (err < self.geo_margin).float()
            valid = valid * gate

        if valid.sum() < 1:
            return torch.tensor(0.0, device=device)

        loss = (err * valid).sum() / (valid.sum() + 1e-8)
        return loss

    def compute_geo_depth_consistency(self, inputs, outputs, cam: int, scale: int = 0):
        cams_p = _REL_CAM_DICT.get(cam, [])
        if len(cams_p) == 0:
            return torch.tensor(0.0, device=outputs[('cam', cam)][('depth', scale)].device)

        total = 0.0
        cnt = 0
        for cam_p in cams_p:
            total = total + self.compute_geo_depth_consistency_pair(inputs, outputs, cam, cam_p, scale)
            cnt += 1
        return total / max(cnt, 1)

    def forward(self, inputs, outputs, cam):
        """
        Same base losses as MultiCamLoss, plus optimized geo depth consistency.
        Only computed for cams in self.geo_ref_cams to reduce compute.
        """
        loss_dict = {}
        cam_loss = 0.0
        target_view = outputs[('cam', cam)]
        output_device = target_view[('depth', 0)].device

        for scale in self.scales:
            kargs = {'cam': cam, 'scale': scale, 'ref_mask': inputs['mask'][:, cam, ...]}

            reprojection_loss = self.compute_reproj_loss(inputs, target_view, **kargs)
            smooth_loss = self.compute_smooth_loss(inputs, target_view, **kargs)
            spatio_loss = self.compute_spatio_soft_loss(inputs, target_view, **kargs)

            kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
            spatio_tempo_loss = self.compute_spatio_tempo_loss(inputs, target_view, **kargs)

            if self.pose_model == 'fsm' and cam != 0:
                pose_loss = self.compute_pose_con_loss(inputs, outputs, **kargs)
            else:
                pose_loss = torch.tensor(0.0, device=output_device)

            # optional extras from base
            if hasattr(self, 'spatial_depth_consistency_loss_weight'):
                spatial_depth_consistency_loss = self.compute_spatial_depth_consistency_loss(inputs, target_view, **kargs)
            if hasattr(self, 'sp_tp_recon_con_loss_weight'):
                sp_tp_recon_con_loss = self.compute_sp_tp_recon_con_loss(inputs, target_view, **kargs)
            if hasattr(self, 'spatial_depth_aug_smoothness'):
                spatial_depth_aug_smooth_loss = self.compute_spatial_depth_aug_smooth_loss(inputs, target_view, **kargs)

            # geo loss only on selected reference cams
            geo_loss = torch.tensor(0.0, device=output_device)
            if self.geo_consist and (cam in self.geo_ref_cams):
                geo_loss = self.compute_geo_depth_consistency(inputs, outputs, cam=cam, scale=scale)

            cam_loss = cam_loss + reprojection_loss
            cam_loss = cam_loss + self.disparity_smoothness * smooth_loss / (2 ** scale)
            cam_loss = cam_loss + self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss
            cam_loss = cam_loss + self.pose_loss_coeff * pose_loss

            if hasattr(self, 'spatial_depth_consistency_loss_weight'):
                cam_loss = cam_loss + self.spatial_depth_consistency_loss_weight * spatial_depth_consistency_loss
            if hasattr(self, 'sp_tp_recon_con_loss_weight'):
                cam_loss = cam_loss + self.sp_tp_recon_con_loss_weight * sp_tp_recon_con_loss
            if hasattr(self, 'spatial_depth_aug_smoothness'):
                cam_loss = cam_loss + self.spatial_depth_aug_smoothness * spatial_depth_aug_smooth_loss

            cam_loss = cam_loss + self.geo_coeff * geo_loss

            if scale == 0:
                loss_dict['reproj_loss'] = float(reprojection_loss.detach())
                loss_dict['spatio_loss'] = float(spatio_loss.detach())
                loss_dict['spatio_tempo_loss'] = float(spatio_tempo_loss.detach())
                loss_dict['smooth'] = float(smooth_loss.detach())
                if self.pose_model == 'fsm' and cam != 0:
                    loss_dict['pose'] = float(pose_loss.detach())
                loss_dict['geo_depth_loss'] = float(geo_loss.detach()) * self.geo_coeff

                self.get_logs(loss_dict, target_view, cam)

        cam_loss = cam_loss / len(self.scales)
        return cam_loss, loss_dict
