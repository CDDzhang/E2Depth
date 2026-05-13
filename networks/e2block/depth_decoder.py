"""V4 - EdgeRefineDecoder
Identical to V1 except:
  - WaveletEdgeBlock only on refinenet1 & refinenet2 (V1 uses all four)
Everything else same as V1:
  - Wavelet AFTER resConfUnit2
  - Full resolution operation
  - Softplus gating with learnable alpha/beta
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DWTConv2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        k_LL = torch.tensor([[1.,  1.],
                             [1.,  1.]])
        k_LH = torch.tensor([[-1., -1.],
                             [ 1.,  1.]])
        k_HL = torch.tensor([[-1.,  1.],
                             [-1.,  1.]])
        k_HH = torch.tensor([[ 1., -1.],
                             [-1.,  1.]])

        self.register_buffer("weight_LL", k_LL[None, None, ...] / 2.0)
        self.register_buffer("weight_LH", k_LH[None, None, ...] / 2.0)
        self.register_buffer("weight_HL", k_HL[None, None, ...] / 2.0)
        self.register_buffer("weight_HH", k_HH[None, None, ...] / 2.0)

    def _depthwise_conv(self, x, weight):
        B, C, H, W = x.shape
        w = weight.repeat(C, 1, 1, 1)
        return F.conv2d(x, w, bias=None, stride=2, padding=0, groups=C)

    def forward(self, x):
        LL = self._depthwise_conv(x, self.weight_LL)
        LH = self._depthwise_conv(x, self.weight_LH)
        HL = self._depthwise_conv(x, self.weight_HL)
        HH = self._depthwise_conv(x, self.weight_HH)
        return LL, LH, HL, HH


class WaveletEdgeBlock(nn.Module):
    """
    Wavelet-based edge enhancement — identical to V1.
    Full resolution DWT → upsample → high_proj → softplus gating.
    """
    def __init__(self, features, alpha_init=1.0, beta_init=0.5):
        super().__init__()
        self._enable_vis_cache = False  # 默认关闭
        self.dwt = DWTConv2d(features)

        # 3C → C
        self.high_proj = nn.Conv2d(features * 3, features, kernel_size=1, bias=True)

        # edge magnitude → sigmoid attention (1x1 keeps spatial sharpness)
        self.edge_proj = nn.Conv2d(1, 1, kernel_size=1, bias=True)

        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        
    

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = H % 2
        pad_w = W % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        LL, LH, HL, HH = self.dwt(x)  # all [B, C, H/2, W/2]

        # Upsample subbands back to full resolution
        LH_up = F.interpolate(LH, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        HL_up = F.interpolate(HL, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        HH_up = F.interpolate(HH, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        # Concat and project high-freq at full resolution
        high_cat = torch.cat([LH_up, HL_up, HH_up], dim=1)  # [B, 3C, H, W]
        high = self.high_proj(high_cat)                        # [B, C, H, W]

        # Edge magnitude and attention
        edge_mag = high.pow(2).mean(dim=1, keepdim=True).sqrt()  # [B, 1, H, W]
        edge_att = torch.sigmoid(self.edge_proj(edge_mag))       # [B, 1, H, W]

        # Softplus gating: always > 0, smooth gradient flow
        out = x + F.softplus(self.alpha * edge_att - self.beta * (1.0 - edge_att)) * high

        if pad_h != 0 or pad_w != 0:
            out = out[..., :H, :W]

        # Visualization cache
        if self._enable_vis_cache:
            self._vis_cache = {
                'input': x[..., :H, :W].detach(),
                'output': out.detach(),
                'LL': LL.detach(),
                'LH': LH_up[..., :H, :W].detach(),
                'HL': HL_up[..., :H, :W].detach(),
                'HH': HH_up[..., :H, :W].detach(),
                'high_freq': high[..., :H, :W].detach(),
                'edge_att': edge_att[..., :H, :W].detach(),
            }

        return out


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)
        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block, optionally with wavelet edge enhancement."""

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None,
        use_wavelet=False
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.use_wavelet = use_wavelet

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        if self.use_wavelet:
            self.wavelet_edge = WaveletEdgeBlock(features)
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        # resConfUnit2 first to refine features,
        # then wavelet enhances edges on clean features
        output = self.resConfUnit2(output)
        if self.use_wavelet:
            output = self.wavelet_edge(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)
        return output


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


def _make_fusion_block(features, use_bn, size=None, use_wavelet=False):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
        use_wavelet=use_wavelet,
    )


class ResizeBlock(nn.Module):
    def __init__(self, channels, scale=None, dilation=1):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=True
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.scale is not None and self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=True)
        x = self.act(self.conv(x))
        return x


class EdgeRefineDecoder(nn.Module):
    def __init__(
        self, 
        in_channels=[64, 128, 256, 256],
        features=128, 
        use_bn=False, 
    ):
        super(EdgeRefineDecoder, self).__init__()
        self._enable_vis_cache = False  # 默认关闭
        self.resize_layers = nn.ModuleList([
            ResizeBlock(in_channels[0], scale=4, dilation=1),
            ResizeBlock(in_channels[1], scale=2, dilation=1),
            ResizeBlock(in_channels[2], scale=None, dilation=1),
            ResizeBlock(in_channels[3], scale=None, dilation=2)
        ])
        
        self.scratch = _make_scratch(
            in_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        # Only refinenet1 and refinenet2 use wavelet edge enhancement
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, use_wavelet=True)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, use_wavelet=True)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, use_wavelet=False)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, use_wavelet=False)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0)
        )
        
    def set_vis_cache(self, enable=True):
        self._enable_vis_cache = enable
        for name in ['refinenet1', 'refinenet2', 'refinenet3', 'refinenet4']:
            block = getattr(self.scratch, name)
            if block.use_wavelet and hasattr(block, 'wavelet_edge'):
                block.wavelet_edge._enable_vis_cache = enable
        return self
    
    def forward(self, out_features, img_h, img_w):
        out = []
        for i, x in enumerate(out_features):
            B, num_cam, C, H, W = x.shape
            x = x.view(B * num_cam, C, H, W)
            x = self.resize_layers[i](x)
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn, size=layer_1_rn.shape[2:])  
        
        if self._enable_vis_cache:
            self._layer_rn_cache = {
                'path_1': path_1.detach(),
                'path_2': path_2.detach(),
                'path_3': path_3.detach(),
                'path_4': path_4.detach(),
            }
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(img_h), int(img_w)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        out = torch.sigmoid(out)
        return {('disp', 0): out}
    
    def collect_wavelet_features(self):
        vis = {}
        for name in ['refinenet4', 'refinenet3', 'refinenet2', 'refinenet1']:
            block = getattr(self.scratch, name)
            if block.use_wavelet and hasattr(block, 'wavelet_edge'):
                w = block.wavelet_edge
                if hasattr(w, '_vis_cache'):
                    vis[name] = w._vis_cache
        return vis