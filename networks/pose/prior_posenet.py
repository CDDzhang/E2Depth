"""
Prior-guided pose network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import ResnetEncoder


# ============================================================
# Pose representation utilities (GT decomposition only, not in gradient path)
# ============================================================

def _rot_to_axis_angle_safe(R):
    """R: [B, 3, 3] -> aa: [B, 3].  Used only inside @torch.no_grad."""
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    w = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=1)
    small = theta < 1e-4
    axis = w / (2 * theta.sin().unsqueeze(1) + 1e-9)
    aa = axis * theta.unsqueeze(1)
    aa[small] = 0.5 * w[small]
    return aa


def _mat_to_pose_vec(T):
    """T: [B,4,4] -> aa: [B,3], t: [B,3].  Used only inside @torch.no_grad."""
    return _rot_to_axis_angle_safe(T[:, :3, :3]), T[:, :3, 3]


class PoseFiLM(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on pose prior.
    
    output = feat * (1 + gamma) + beta
    
    Initialized so that gamma=0, beta=0 → identity transform when prior is zero.
    """
    def __init__(self, feat_dim, pose_dim=6, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.to_gamma = nn.Linear(hidden, feat_dim)
        self.to_beta = nn.Linear(hidden, feat_dim)
        
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)
    
    def forward(self, feat, pose_6d):
        h = self.mlp(pose_6d)
        gamma = self.to_gamma(h).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(h).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta


# ============================================================
# Pose decoder: multi-scale features → full 6DoF (direct output)
# ============================================================

class PoseRefineDecoder(nn.Module):
    """
    Fuses layer3 + layer4 from ResNet to predict a 6DoF residual (delta)
    on top of the pose prior, scaled by 0.01.

    out_conv is zero-initialized so that delta = 0 at step 0, i.e. the
    network output exactly equals the prior at initialization.
    """
    def __init__(self, num_ch_enc):
        super().__init__()
        ch3 = num_ch_enc[-2]
        ch4 = num_ch_enc[-1]
        
        self.squeeze4 = nn.Sequential(nn.Conv2d(ch4, 256, 1), nn.ReLU(True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(ch3, 256, 1), nn.ReLU(True))
        
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
        )
        self.out_conv = nn.Conv2d(256, 6, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, feat3, feat4):
        f4 = self.squeeze4(feat4)
        f3 = self.squeeze3(feat3)
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode='bilinear', align_corners=True)
        out = self.fuse(torch.cat([f3, f4_up], dim=1))
        out = self.out_conv(out)
        return 0.01 * out.mean(dim=[2, 3])  # GAP → [B, 6], same scale as MonoPoseNet


# ============================================================
# Main: PriorPoseNet  (v2 — no compose)
# ============================================================

class PriorPoseNet(nn.Module):
    """
    GT-prior conditioned pose network, front camera only.
    
    v3 key changes (residual + validity flag):
      - Residual parameterization in 6D vector space:
            aa_out = aa_prior + delta_aa,  t_out = t_prior + delta_t
        Decoder out_conv is zero-initialized, so the output equals the
        prior at step 0. Frame-to-frame rotations are small, so additive
        composition in vector space differs from manifold composition
        only by second-order terms. No axis-angle <-> matrix round-trips.
      - FiLM conditioning input is 7D: [aa_prior, t_prior, valid_flag].
        valid_flag = 1 when a prior is present, 0 when the prior is
        dropped (curriculum dropout) or unavailable (eval fallback).
        This disambiguates "prior missing" from "near-zero motion".
        When the prior is dropped, prior = 0 and delta alone carries the
        full pose, which is consistent with the residual formulation.
      - Curriculum noise: noise sigma and dropout ramp 0 -> target over
        noise_warmup_steps, then stay at target level
      - Output interface identical to MonoPoseNet
    """
    def __init__(self, cfg):
        super(PriorPoseNet, self).__init__()
        num_layers = cfg['model'].get('num_layers', 18)
        pretrained = cfg['model'].get('weights_init', True)
        
        self.pose_encoder = ResnetEncoder(num_layers, pretrained, num_input_images=2)
        del self.pose_encoder.encoder.fc
        
        ch4 = self.pose_encoder.num_ch_enc[-1]
        self.post_film_norm = nn.LayerNorm([ch4])
        self.pose_film = PoseFiLM(feat_dim=ch4, pose_dim=7)  # 6D prior + validity flag
        self.pose_decoder = PoseRefineDecoder(self.pose_encoder.num_ch_enc)
        
        self.noise_rot_sigma = cfg.get('pose', {}).get('noise_rot_sigma', 0.002)
        self.noise_trans_sigma = cfg.get('pose', {}).get('noise_trans_sigma', 0.1)
        self.gt_drop_prob = cfg.get('pose', {}).get('gt_drop_prob', 0.1)
        self.trans_clamp = cfg.get('pose', {}).get('trans_clamp', 4.0)
        
        # frame_ids from config
        self.frame_ids = cfg['training']['frame_ids']

        # ---- Curriculum noise schedule ----
        self.noise_warmup_steps = cfg.get('pose', {}).get('noise_warmup_steps', 1000)
        self.register_buffer('_train_step', torch.tensor(0, dtype=torch.long))

    def _get_curriculum_ratio(self):
        """
        Returns a float in [0, 1] indicating the current noise level.
        0 at step 0, linearly reaches 1 at noise_warmup_steps, stays 1 after.
        """
        if self.noise_warmup_steps <= 0:
            return 1.0
        return min(self._train_step.item() / self.noise_warmup_steps, 1.0)

    @torch.no_grad()
    def _prepare_prior(self, T_gt, B, device, has_prior):
        """
        Prepare 6D pose prior and validity flag from GT 4x4 matrix.
        Entirely detached — no gradient flows through the prior.
        
        Training: GT + curriculum noise + curriculum dropout.
          - Noise sigma ramps from 0 → target over warmup_steps
          - Dropout prob ramps from 0 → gt_drop_prob over warmup_steps
          - valid = 0 for dropped samples, 1 otherwise
        Eval: GT directly with valid = 1, or zeros with valid = 0
        when no prior is available (has_prior=False).
        """
        aa_gt, t_gt = _mat_to_pose_vec(T_gt)

        if not has_prior:
            valid = torch.zeros(B, 1, device=device)
            return torch.zeros_like(aa_gt), torch.zeros_like(t_gt), valid

        if self.training:
            ratio = self._get_curriculum_ratio()
            
            cur_rot_sigma = self.noise_rot_sigma * ratio
            cur_trans_sigma = self.noise_trans_sigma * ratio
            cur_drop_prob = self.gt_drop_prob * ratio
            
            aa_prior = aa_gt + torch.randn_like(aa_gt) * cur_rot_sigma
            t_prior = t_gt + torch.randn_like(t_gt) * cur_trans_sigma
            
            drop = (torch.rand(B, 1, device=device) < cur_drop_prob).float()
            aa_prior = aa_prior * (1 - drop)
            t_prior = t_prior * (1 - drop)
            valid = 1.0 - drop
        else:
            aa_prior = aa_gt
            t_prior = t_gt
            valid = torch.ones(B, 1, device=device)
        
        return aa_prior, t_prior, valid
    
    def forward(self, inputs, frame_ids, cam):
        """
        Interface identical to MonoPoseNet.
        
        Returns:
            axisangle:   [B, 1, 1, 3]
            translation: [B, 1, 1, 3]
        """
        # Advance curriculum step counter (only during training)
        if self.training:
            self._train_step += 1

        B = inputs[('color_aug', 0, 0)].shape[0]
        device = inputs[('color_aug', 0, 0)].device

        if frame_ids == [-1, 0]:
            frame_id = -1
            invert = True
        else:
            frame_id = 1
            invert = False

        pose_inputs = [inputs[('color_aug', f_i, 0)][:, cam, ...] for f_i in frame_ids]
        img_pair = torch.cat(pose_inputs, dim=1)

        # ---- Prepare GT prior (detached, no gradient) ----
        has_prior = self.training and 'pose' in inputs
        if has_prior:
            gt_T = self._get_gt_pose_from_inputs(inputs, cam, frame_id)
            T_gt_for_net = torch.inverse(gt_T) if invert else gt_T
        else:
            T_gt_for_net = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

        # ---- Encode + FiLM hint + Decode ----
        feats = self.pose_encoder(img_pair)

        aa_prior, t_prior, valid = self._prepare_prior(T_gt_for_net, B, device, has_prior)
        prior_in = torch.cat([aa_prior, t_prior, valid], dim=1)   # [B, 7], detached
        feats[-1] = self.pose_film(feats[-1], prior_in)
        B_f, C, H, W = feats[-1].shape
        feats[-1] = self.post_film_norm(feats[-1].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # ---- Residual 6DoF output: prior + delta (vector-space compose) ----
        # Dropped/unavailable prior is exactly zero, so delta alone carries
        # the full pose in that mode; otherwise delta refines the prior.
        delta_6d = self.pose_decoder(feats[-2], feats[-1])    # [B, 6], zero at init
        aa_out = aa_prior + delta_6d[:, :3]
        t_out = t_prior + delta_6d[:, 3:]

        axisangle   = aa_out.unsqueeze(1).unsqueeze(1)   # [B, 1, 1, 3]
        translation = t_out.unsqueeze(1).unsqueeze(1)    # [B, 1, 1, 3]

        return axisangle, torch.clamp(translation, -self.trans_clamp, self.trans_clamp)

    @staticmethod
    def _get_gt_pose_from_inputs(inputs, cam, frame_id):
        """
        Extract GT relative pose cam_T_cam(cam, frame_id) from inputs.
        Uses float64 for precision with large world coordinates.
        """
        T_wc_t = inputs['pose'][:, cam, ...].double()
        if frame_id > 0:
            T_wc_other = inputs['pose_fwd'][:, cam, ...].double()
        else:
            T_wc_other = inputs['pose_bwd'][:, cam, ...].double()
        return (torch.inverse(T_wc_other) @ T_wc_t).float()

    @staticmethod
    def propagate_to_all_cameras(T_front, inputs, frame_ids):
        """Propagate front camera pose to all cameras via extrinsics."""
        E = inputs['extrinsics']
        E_inv = inputs['extrinsics_inv']
        num_cams = E.shape[1]
        E_0 = E[:, 0, ...]
        E_0_inv = E_inv[:, 0, ...]
        
        pose_dict = {}
        for cam_i in range(num_cams):
            pose_dict[('cam', cam_i)] = {}
            E_c = E[:, cam_i, ...]
            E_c_inv = E_inv[:, cam_i, ...]
            for fid in frame_ids[1:]:
                T_f = T_front[fid]
                if cam_i == 0:
                    pose_dict[('cam', cam_i)][('cam_T_cam', 0, fid)] = T_f
                else:
                    pose_dict[('cam', cam_i)][('cam_T_cam', 0, fid)] = \
                        E_c_inv @ E_0 @ T_f @ E_0_inv @ E_c
        return pose_dict
