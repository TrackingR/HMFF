import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.tbsi_track.base_backbone import BaseBackbone
from lib.models.tbsi_track.utils import combine_tokens, recover_tokens
from lib.models.layers.fusion_modules import CBAMLayer
from lib.models.layers.MDIM_layer import TBSILayer
from lib.models.layers.adapter import Bi_direct_adapter
class MSA(nn.Module):
    def __init__(self, dim=768, num=8, qkv=False, attn_drop_msa=0., proj_drop_msa=0.):
        super().__init__()
        self.num = num
        head_dim = dim // num
        self.scale_msa = head_dim ** -0.5

        self.qkv_msa = nn.Linear(dim, dim * 3, bias=qkv)
        self.attn_drop_msa = nn.Dropout(attn_drop_msa)
        self.proj_msa = nn.Linear(dim, dim)
        self.proj_drop_msa = nn.Dropout(proj_drop_msa)

    def forward(self, x, x_c, return_attention=False):
        B, N, C = x.shape
        # x_c = torch.cat([x_v, x_i], dim=1)
        B2, N2, C2 = x_c.shape
        qkv = self.qkv_msa(x).reshape(B, N, 3, self.num, C // self.num).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv_msa(x_c).reshape(B2, N2, 3, self.num, C2 // self.num).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        attn1 = (q @ k.transpose(-2, -1)) * self.scale_msa
        attn2 = (q @ k2.transpose(-2, -1)) * self.scale_msa
        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn1 = self.attn_drop_msa(attn1)
        attn2 = self.attn_drop_msa(attn2)
        x1 = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)
        # x = torch.cat(x1, x2, dim=2)
        x = x1 + x2
        x = self.proj_msa(x)
        x = self.proj_drop_msa(x)

        if return_attention:
            return x, attn
        return x
