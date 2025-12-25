import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock_priors(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_priors, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class Priors_Attention(nn.Module):
    """使用 CLIP 特征指导对 out_dec_level1 的自适应增强。
    → 结合 Transformer 和 ResNet 的长程、短程特征建模能力，学习到“区域自适应”的增强掩膜和特征。"""
    def __init__(self, num_blocks=[4,6,6,8],num_refinement_blocks = 4,heads = [1,2,4,8], ffn_expansion_factor = 2.66, bias = False, LayerNorm_type = 'WithBias'):
        super(Priors_Attention, self).__init__()

        nf = 32
        self.conv_last2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last3 = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        # self.conv_last3 = nn.Conv2d(nf, 6, 3, 1, 1, bias=True)
        self.pool=nn.AdaptiveAvgPool2d(1)
        kh=3
        self.kh=kh
        kw=kh
        
        self.mapp_space1=nn.Linear(512, 256,bias=False)
        self.mapp_space2=nn.Linear(256, nf*4,bias=False)
        self.mapp_space11=nn.Linear(512, 256,bias=False)
        self.mapp_space22=nn.Linear(256, nf*4,bias=False)
        
        self.mapp_position1=nn.Conv2d(2, nf, 3, 1, 1, bias=False)
        self.mapp_position2=nn.Conv2d(nf, nf, 1, 1, 0, bias=False)
        self.mapp_position11=nn.Conv2d(2, nf, 3, 1, 1, bias=False)
        self.mapp_position22=nn.Conv2d(nf, nf, 1, 1, 0, bias=False)

        self.mapping11=nn.Linear(nf*4+nf, nf*1,bias=False)
        self.mapping12=nn.Linear(nf*1, nf*1,bias=False)
        self.mapp11=nn.Conv2d(nf*5+nf, nf*kh*kw, 3,1,1,bias=False)
        self.mapp12=nn.Conv2d(nf*kh*kw, nf*kh*kw,3,1,1,bias=False)
        self.mapp1=nn.Conv2d(nf, 1, 3, 1, 1, bias=False)
        self.refine11=nn.Conv2d(nf*2, nf, 1, 1, 0, bias=False)
        
        self.r1=nn.Conv2d(3+3, nf, 3, 1, 1, bias=False)
        self.r2=nn.Conv2d(nf, nf, 3, 2, 1, bias=False)
        self.r22=nn.Conv2d(nf, nf, 3, 2, 1, bias=False)
        
        self.r1_decoder=nn.Conv2d(nf, nf*4, 3, 1, 1, bias=False)
        self.r2_decoder=nn.Conv2d(nf, nf*4, 3, 1, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.r1norm=nn.InstanceNorm2d(nf, affine=True)
        self.r2norm=nn.InstanceNorm2d(nf, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.r1_mask=nn.Conv2d(nf+nf*5, nf, 3, 1, 1, bias=False)
        self.r2_mask=nn.Conv2d(nf, 1, 3, 1, 1, bias=False)
        
        self.transformer = nn.Sequential(*[TransformerBlock_priors(dim=nf, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.resnet=ResidualBlock_noBN(nf)

    def forward(self, clip_condition_list, inp_img, out_dec_level1=None):
        """
        inp_img: 输入的原始退化图片
        out_dec_level1: 初始恢复图像
        """
        ### 这里获得“原始图”的 CLIP 特征。
        feat_clip_normal=clip_condition_list[0].float()
        
        # ### 把“初步增强图”和“原始图”拼接，经过两次卷积 + InstanceNorm 下采样，得到低分辨率特征 out_noise2。这一部分提取出联合特征。
        # out_noise2=self.r22(self.r2norm(self.r2(self.lrelu(self.r1norm(self.r1(torch.cat([out_dec_level1,inp_img],dim=1)))))))
        # batch_size=out_noise2.shape[0]
        # height=out_noise2.shape[2]
        # width=out_noise2.shape[3]

        # ##### position embedding 生成坐标嵌入，给每个像素分配位置编码，使网络具有空间意识。
        # xs = torch.linspace(-1, 1, steps=height)
        # ys = torch.linspace(-1, 1, steps=width)
        # grid_x, grid_y = torch.meshgrid(xs, ys)
        # grid_x=grid_x.unsqueeze(dim=0).unsqueeze(dim=0)
        # grid_y=grid_y.unsqueeze(dim=0).unsqueeze(dim=0)
        # grid=torch.cat([grid_x,grid_y],dim=1).repeat(batch_size,1,1,1).to(out_noise2.device)
        # position_embedding1=self.lrelu(self.mapp_position1(grid))
        # position_embedding1=self.mapp_position2(position_embedding1)
        

        # ### fusion1：融合 CLIP 特征，生成掩膜（mask）：结合图像特征 + CLIP 特征 + 位置信息 → 输出一个掩膜图 mask_input ∈ [0,1]。
        # ### 这个掩膜控制 Transformer 与 ResNet 特征的融合比例。
        # feat_clip_normal1=self.lrelu(self.mapp_space1(feat_clip_normal))
        # feat_clip_normal1=self.mapp_space2(feat_clip_normal1)
        
        # mask_input1=feat_clip_normal1.view(batch_size,-1,1,1).repeat(1,1,height,width)
        # mask_input=torch.cat([out_noise2, mask_input1, position_embedding1],dim=1)
        # mask_input=self.r2_mask(self.lrelu(self.r1_mask(mask_input)))
        # mask_input=nn.Sigmoid()(mask_input)

        # ### 融合长程与短程特征
        # feature_long=self.transformer(out_noise2)
        # feature_short=self.resnet(out_noise2)
        # out_noise3 = feature_long * mask_input + feature_short * (1-mask_input)

        
        ### 第二阶段条件增强（通道融合 + 空间卷积核预测）
        height=inp_img.shape[2]
        width=inp_img.shape[3]
        batch_size=inp_img.shape[0]
        feap=self.pool(inp_img).view(batch_size,-1)

        ##### position embedding
        # ##### position embedding 生成坐标嵌入，给每个像素分配位置编码，使网络具有空间意识。
        xs = torch.linspace(-1, 1, steps=height)
        ys = torch.linspace(-1, 1, steps=width)
        grid_x, grid_y = torch.meshgrid(xs, ys)
        grid_x=grid_x.unsqueeze(dim=0).unsqueeze(dim=0)
        grid_y=grid_y.unsqueeze(dim=0).unsqueeze(dim=0)
        grid=torch.cat([grid_x,grid_y],dim=1).repeat(batch_size,1,1,1).to(inp_img.device)
        position_embedding2=self.lrelu(self.mapp_position11(grid))
        position_embedding2=self.mapp_position22(position_embedding2)

        ### fusion2 ：通道注意力
        feat_clip_normal2=self.lrelu(self.mapp_space11(feat_clip_normal))
        feat_clip_normal2=self.mapp_space22(feat_clip_normal2)
        
        f1=self.mapping11(torch.cat([feap, feat_clip_normal2],dim=1))
        f1=self.mapping12(self.lrelu(f1)).view(batch_size,-1,1,1)
        feature1=nn.Sigmoid()(f1)
        feature1=feature1*inp_img
        ### 通道注意力完成，注意力图 Mc 是 feature1=nn.Sigmoid()(f1)，out_noise3是f^
        
        ### 把 CLIP 条件展开成与空间一致的张量并拼接位置编码，对应公式 Tc(g)
        input_condition2=feat_clip_normal2.view(batch_size,-1,1,1).repeat(1,1,height,width)
        input_condition=torch.cat([inp_img, input_condition2, position_embedding2],dim=1)      # torch.Size([4, 128, 320, 320])  128+32+32=192
        
        ### 通过卷积生成动态卷积核参数 g1，对应公式（5），公式（5）得到的 Cp 是卷积核参数，其变量名称是g1。
        g1=self.mapp11(input_condition)
        g1=self.mapp12(self.lrelu(g1))                                                         # torch.Size([4, 288, 320, 320])
        
        ### 用 kernel2d_conv（自适应卷积）对特征做空间可变滤波。对应公式（6）的第一个。利用卷积核 Cp 对 f^进行卷积
        fea1=kernel2d_conv(inp_img, g1, self.kh)
        
        ### 经非线性映射并生成空间掩码，然后与 out_noise3 做逐像素位置的加权，对应公式（6）的第二个，然后乘以掩码
        fea1=self.mapp1(self.lrelu(fea1))
        fea1=nn.Sigmoid()(fea1)*inp_img
        
        ### 与之前的通道自适应融合（feature1）拼接并经 1×1 卷积融合，残差跳连。最终得到 f- ：feature1
        # import ipdb; ipdb.set_trace()
        feature1=self.refine11(torch.cat([feature1,fea1],dim=1))+inp_img
        return feature1
        feature1=self.r1_decoder(feature1)
        feature1=self.lrelu(self.pixel_shuffle(feature1))
        feature1=self.r2_decoder(feature1)
        feature1=self.lrelu(self.pixel_shuffle(feature1))
        
        feature_noise=self.conv_last2(feature1)
        feature_noise=self.conv_last3(self.lrelu(feature_noise))
        
        # if feature_noise.shape[1]==6:
        #     feature_noise_rgb = feature_noise[:, 3:, :, :]
        # else:
        #     feature_noise_rgb = feature_noise
        # import ipdb; ipdb.set_trace()
        out_noise4=torch.clamp(out_dec_level1,min=0.0,max=1.0)+feature_noise
        return out_noise4


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    """
    feat_in	[N, C, H, W]	输入特征图
    kernel	[N, C, H, W]	每个像素点对应的动态卷积核权重（flatten 形式）
    ksize	int	            卷积核大小（例如3、5、7）
    “空间可变卷积（Spatially Variant Convolution）”:让每个像素使用自己独立的卷积核来进行特征变换。
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, -1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out

        
        
##########-------------------------------------上面是Priors部分，下面是YOLO网络-------------------------------------########



#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from .deform_conv_v2 import DeformConv2D
import clip


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv

        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


class dConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=3,d = 3, g=4, act=True,):  # ch_in, ch_out, kernel, stride, padding, groups
        super(dConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, d, g, bias=False,)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class C3TR(CSPLayer):
    # C3 module with TransformerBlock()
    def __init__(self, in_channels, out_channels, n=1, shortcut=True,  e=0.5):
        super().__init__(in_channels, out_channels, n, shortcut,  e)
        c_ = int(out_channels * e)
        self.m = TransformerBlock(c_, c_, 4, n)
        
    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.conv3(x)
        return x


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(c)
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)

        self.layernorm2 = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, 4 * c, bias=False)
        self.fc2 = nn.Linear(4 * c, c, bias=False)

        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x1 = x
        x = self.layernorm1(x)
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x1  # <--- return two outputs, which is not kind to DDP

        x2 = x
        x = self.layernorm2(x)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.fc2(x)) + x2

        return x

class TransformerBlock(nn.Module):
    # Vision Transformer
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).unsqueeze(0).transpose(0, 3).squeeze(3)
        return self.tr(p + self.linear(p)).unsqueeze(3).transpose(0, 3).reshape(b, self.c2, w, h)
# from .fixed_datknet import TransformerBlock, TransformerLayer

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels   = int(wid_mul * 64)  # 64
        base_depth      = max(round(dep_mul * 3), 1)  # 3

        self.stem = Focus(3, base_channels, ksize=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )
        
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )
        self.deconv = DeformConv2D(512, 512, kernel_size=3, padding=1, modulation=True)
        self.swt1 = C3TR(512, 512)
        
        num_blocks = [4,6,6,8]
        num_refinement_blocks = 4
        heads = [1,2,4,8]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        
        self.priors_attention=Priors_Attention(num_blocks=num_blocks, num_refinement_blocks=num_refinement_blocks,
                                    heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias = bias, LayerNorm_type=LayerNorm_type)
        # self.clip_model, preprocess = clip.load("ViT-B/32", device=device)
        self.clip_model, preprocess = clip.load("ViT-B/32")
        # CLIP标准化参数
        self.register_buffer("mean_feature", torch.FloatTensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("std_feature", torch.FloatTensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))
        # ImageNet标准化参数（用于还原输入）
        self.register_buffer("mean_imagenet", torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std_imagenet", torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))


    def forward(self, x):
        '''
        这里除了Together原作者添加的部分（两个可变形卷积一个注意力机制），其余网络结果和传输顺序都和yolox一样。self.dark2-5的结构都一样。
        '''
        outputs = {}
        real_low=x
        x = self.stem(x)
        
        # import ipdb; ipdb.set_trace()
        # #******添加的部分*****
        mean_feature = self.mean_feature
        std_feature  = self.std_feature
        mean_imagenet = self.mean_imagenet
        std_imagenet = self.std_imagenet

        # 正确的预处理流程：
        # 1. 将ImageNet标准化的输入还原到[0,1]
        real_low = real_low * std_imagenet + mean_imagenet
        # 2. clamp到[0,1]范围（防止数值溢出）
        real_low = torch.clamp(real_low, min=0, max=1)
        # 3. 应用CLIP标准化
        real_low = (real_low - mean_feature) / std_feature
        real_low_clip = F.interpolate(real_low, size=(224, 224), mode='bilinear', align_corners=False)
        image_features2 = self.clip_model.encode_image(real_low_clip)
        image_features2n = image_features2/image_features2.norm(dim=-1, keepdim=True)
        
        clip_condition_list=[]
        clip_condition_list.append(image_features2n)
        x = self.priors_attention.forward(clip_condition_list, x)
        # self.priors_attention.forward(clip_condition_list, x)
        # #******添加的部分*****
        outputs["stem"] = x    # torch.Size([4, 32, 320, 320])
        
        x = self.dark2(x)
        outputs["dark2"] = x   # torch.Size([4, 64, 160, 160])
        x = self.dark3(x)      
        outputs["dark3"] = x   # torch.Size([4, 128, 80, 80])
        x = self.dark4(x)
        outputs["dark4"] = x   # torch.Size([4, 256, 40, 40])
        x = self.dark5(x)
                               # torch.Size([4, 512, 40, 40])
        x1 = self.deconv(x)    # x1.shape = torch.Size([4, 512, 20, 20])
        x = self.deconv(x1)    # x.shape = torch.Size([4, 512, 20, 20])
        x = self.swt1(x)       # x.shape = torch.Size([4, 512, 20, 20])        
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out


if __name__ == '__main__':
    print(CSPDarknet(1, 1))