#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchvision.datasets import ImageFolder
from modelNer13 import MultiscaleNet as myNet
import torch.nn.functional as F
from einops import rearrange
import numbers
from utils import PatchExpand, FinalPatchExpand_X4, STMBlock
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=2):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.ReLU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out =self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()


        # Multiscale Block
        self.safm = SAFM(dim)
        # Feedforward layer

    def forward(self, x):
        x = self.safm(x) + x
        return x
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, branch_ratio=0.5):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Sequential(nn.Conv2d(gc, gc, square_kernel_size, padding=1),
                                       )
        self.block_2 = STMBlock(hidden_dim=64,
                                drop_path=16,
                                norm_layer=nn.LayerNorm,
                                ssm_ratio=2,
                                d_state=16,
                                mlp_ratio=0,
                                dt_rank="auto")
        self.expand_layers = PatchExpand(input_resolution=None,
                                                         dim=64,
                                                         dim_scale=2,
                                                         norm_layer=nn.LayerNorm)
        self.concat_layers = nn.Linear(128, 64)

        self.split_indexes = (gc, gc )
        self.relu = nn.ReLU()


    def forward(self, x):
        x_id, x_hw= torch.split(x, self.split_indexes, dim=1)
        x_id = self.block_2(self.concat_layers(self.expand_layers(x_id)))
        x_id = x_id.permute(0, 3, 1, 2)
        xr = torch.cat(
            (x_id, self.dwconv_hw(x_hw)),
            dim=1 )
        xr = self.relu(xr)

        return xr
class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv = InceptionDWConv2d(64, band_kernel_size=3)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SemanticSelectionAttention(nn.Module):
    def __init__(self, in_channels):
        super(SemanticSelectionAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.global_avg_pool(x).view(b, c)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        return x


class SPFUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPFUnit, self).__init__()
        self.conv_block1 = ConvolutionBlock(in_channels, out_channels)
        self.conv_block2 = ConvolutionBlock(out_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention = SemanticSelectionAttention(out_channels)

    def forward(self, Fi, Fi_SPF_minus_1):
        Fi_SPF = self.conv_block1(Fi)
        Fi_SPF = self.conv_block2(Fi_SPF_minus_1)

        if Fi_SPF_minus_1.size() != Fi_SPF.size():
            Fi_SPF_minus_1 = self.upsample(Fi_SPF_minus_1) if Fi_SPF_minus_1.size(2) < Fi_SPF.size(
                2) else self.downsample(Fi_SPF_minus_1)

        Fi_concat = torch.cat((Fi_SPF_minus_1, Fi_SPF), dim=1)
        Fi_att = self.attention(Fi_concat)

        Fi_SPF_out = Fi_SPF * Fi_att
        return Fi_SPF_out
def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )



class SemanticSelectionAttention(nn.Module):
    def __init__(self, in_channels):
        super(SemanticSelectionAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.global_avg_pool(x).view(b, c)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.sigmoid(x).view(b, c, 1, 1)
        return x

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
class Dautoencoder(nn.Module):
    def __init__(self, in_channels,in_channels2, out_channels,padding=False,device=device):
        super(Dautoencoder, self).__init__()
        self.myNet=myNet(device=device)
        pad_mult = 1 if padding else 0


        self.autoencoder = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
            # decoder
            nn.Upsample(size=3, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=8, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=15, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=32, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=63, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=127, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=1,
                      padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Upsample(size=56, mode='bilinear'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      stride=1, padding=1)
    )

    def forward(self, Fi, Fi_SPF_minus_1):
        outputnet=self.myNet(Fi)
        output = self.autoencoder(outputnet)
        return output

def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)

    )
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class CCM(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        self.ccm = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.ccm(x)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = self.gamma * out + x
        return out
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
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
class LayerNormS(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNormS, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(to_3d(x), h, w)
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

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
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()




        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):

        x = self.ffn(x)

        return x
class Student(nn.Module):
    def __init__(self, out_channels):
        super(Student, self).__init__()
        pad_mult = 1

        self.dt1=InceptionDWConv2d(128)
        self.dt2 = InceptionDWConv2d(128)
        self.dt3 = InceptionDWConv2d(128)
        self.dt4 = InceptionDWConv2d(128)

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=3 * pad_mult),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            CCM(out_channels, 2.0),
        )


        self.decoderM = nn.Sequential(


            nn.Upsample(size=32, mode='bicubic'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(size=64, mode='bicubic'),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            CCM(out_channels, 2.0),

        )
        self.decoderM2 = nn.Sequential(

            AttBlock(out_channels, 2),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

        self.decoderAEG = nn.Sequential(


            nn.Conv2d(in_channels=out_channels, out_channels=64, kernel_size=16),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=4, mode='bicubic'),
            self.dt1,
            nn.Dropout(0.2),
            nn.Upsample(size=8, mode='bicubic'),
            self.dt2,
            nn.Dropout(0.2),

            nn.Upsample(size=32, mode='bicubic'),
            self.dt3,
            nn.Dropout(0.2),
            nn.Upsample(size=64, mode='bicubic'),
            self.dt4,
            nn.Dropout(0.2),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Dropout(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),

        )
        self.decoderAEG2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)


        # 解码器
        x2 = self.decoder(x1)



        x3 = self.decoderM(x2)



        x3o = self.decoderM2(x3)+x3

        # x6 = self.att2(x1)

        x4 = self.decoderAEG(x2)

        # x41 = x6*x4
        #
        x4o=self.decoderAEG2(x4)

        return x1, x4o , x3o
class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
