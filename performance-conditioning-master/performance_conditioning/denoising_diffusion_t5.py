import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import librosa
from datetime import datetime
from scipy.io.wavfile import write as wav_write
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T, utils
from onsets_and_frames.midi_utils import *
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
# from onsets_and_frames.mel import melspectrogram, melspectrogram_this, melspectrogram_src
# from onsets_and_frames.mel import melspectrogram_this

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA
from onsets_and_frames.constants import *
# from accelerate import Accelerator
import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder
import math

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions



def spectral_reconstruction_loss(x, G_x, specs, eps=1e-5, factor=16.):
    L = 0
    for i in range(6, 12):
        s = 2**i
        alpha_s = (s/2)**0.5
        # melspec = MelSpectrogram(sample_rate=sr, n_fft=s, hop_length=s//4, n_mels=64, wkwargs={"device": device}).to(device)
        melspec = specs[i - 6]
        S_x = melspec(x.squeeze(1))
        S_G_x = melspec(G_x.squeeze(1))
        f_bins = S_x.shape[-2]
        t_bins = S_x.shape[-1]
        # print('t bins f bins', t_bins, f_bins)
        loss = (S_x-S_G_x).abs().sum() + alpha_s*(((torch.log(S_x.abs()+eps)-torch.log(S_G_x.abs()+eps))**2).sum(dim=-2)**0.5).sum()
        L += loss #/ (f_bins * t_bins)
    # print('recon loss', LAMBDA_REC * L.item())
    return L / factor

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img, meanstd=None):
    if exists(meanstd):
        mn, std = meanstd
        return (img - mn) / std
    return img * 2 - 1

def unnormalize_to_zero_to_one(t, meanstd=None):
    if exists(meanstd):
        mn, std = meanstd
        return t * std + mn
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None, factor=2):
    return nn.Sequential(
        nn.Upsample(scale_factor=factor, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

# def Downsample(dim, dim_out = None, kernel=4, stride=2, padding=1):
#     return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)
def Downsample(dim, dim_out = None, kernel=4, stride=2, padding=1):
    return nn.Conv1d(dim, default(dim_out, dim), kernel, stride, padding)

class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.g = nn.Parameter(torch.ones(1, dim, 1))
#
#     def forward(self, x):
#         eps = 1e-5 if x.dtype == torch.float32 else 1e-3
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) * (var + eps).rsqrt() * self.g

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = LayerNorm(dim)
#
#     def forward(self, x):
#         x = self.norm(x)
#         return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):

        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # print('sinu emb x', emb.shape, x.shape)
        emb = x[:, None] * emb[None, :]
        # print('emb shape', emb.shape)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, acti_relu=False):
        super().__init__()

        if groups == 'instance':
            print('using instance norm, no ws')
            self.norm = nn.InstanceNorm1d(dim_out)
            self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        elif exists(groups):
            print('group norm with ws, groups:', groups)
            self.norm = nn.GroupNorm(groups, dim_out) #if exists(groups) else None
            # print('self.norm', self.norm)
            self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding = 1)# if exists(groups) else nn.Conv1d(dim, dim_out, 3, padding = 1)
        else:
            print('no norm')
            self.norm = None
            self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)



        self.act = nn.SiLU() if not acti_relu else nn.LeakyReLU(negative_slope=0.1)


    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        if exists(self.norm):
            x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, id_emb_dim=None, acti_relu=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.id_mlp = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(id_emb_dim, dim_out * 2)
        ) if exists(id_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups, acti_relu=acti_relu)
        self.block2 = Block(dim_out, dim_out, groups = groups, acti_relu=acti_relu)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, id_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        id_scale_shift = None
        if hasattr(self, 'id_mlp') and exists(self.id_mlp) and exists(id_emb):
            id_emb = self.id_mlp(id_emb)
            id_emb = rearrange(id_emb, 'b c -> b c 1')
            id_scale_shift = id_emb.chunk(2, dim = 1)
            # print('resnet block id_scale_shift', id_scale_shift[0].shape, id_scale_shift[1].shape)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h, scale_shift=id_scale_shift)

        return h + self.res_conv(x)

class ResnetBlockV2(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, id_emb_dim=None, acti_relu=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim + id_emb_dim, dim_out * 2)
        )

        self.block1 = Block(dim, dim_out, groups = groups, acti_relu=acti_relu)
        self.block2 = Block(dim_out, dim_out, groups = groups, acti_relu=acti_relu)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, id_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # print('before cat', time_emb.shape, id_emb.shape)
            time_and_id = torch.cat((time_emb, id_emb), dim=1)
            time_emb = self.mlp(time_and_id)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) y -> b h c y', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c y -> b (h c) y', h = self.heads, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) y -> b h c y', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h y d -> b (h d) y', y = w)
        return self.to_out(out)


### google code
# model


# Copyright 2022 The Music Spectrogram Diffusion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5.1.1 Transformer model."""


class FiLMLayer(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.linear = nn.Linear(cond_features, 2 * features)

    def forward(self, x, cond):
        scale_shift = self.linear(cond).unsqueeze(0)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=-1)
        x = (1. + scale) * x + shift
        return x


class T5Config:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int
    # Activation dtypes.
    dtype = torch.float32
    emb_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    head_dim: int = 64
    mlp_dim: int = 2048
    # Activation functions are retrieved from Flax.
    mlp_activations = ('relu',)
    dropout_rate: float = 0.1
    max_decoder_noise_time: float = 2e4
    decoder_cross_attend_style: str = 'sum_cross_attends'
    position_encoding: str = 'fixed'
    context_positions: str = 'regular'


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerFiLMDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).

    # Examples::
    #     >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    #     >>> memory = torch.rand(10, 32, 512)
    #     >>> tgt = torch.rand(20, 32, 512)
    #     >>> out = decoder_layer(tgt, memory)
    #
    # Alternatively, when ``batch_first`` is ``True``:
    #     >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
    #     >>> memory = torch.rand(32, 10, 512)
    #     >>> tgt = torch.rand(32, 20, 512)
    #     >>> out = decoder_layer(tgt, memory)
    # """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, d_time: int = 512,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerFiLMDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.film1 = FiLMLayer(d_model, d_time)
        self.film2 = FiLMLayer(d_model, d_time)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerFiLMDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, t_cond: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.film1(self.norm1(x), t_cond), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.film2(self.norm3(x), t_cond))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TransformerFilmEncoderLayer(Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, d_id: int = 512,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerFilmEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        self.film1 = FiLMLayer(d_model, d_id)
        self.film2 = FiLMLayer(d_model, d_id)
        print('film encoder layer init')

    def __setstate__(self, state):
        super(TransformerFilmEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, id_cond: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        assert self.norm_first

        ###### from decoer code
        # x = tgt
        # if self.norm_first:
        #     x = x + self._sa_block(self.film1(self.norm1(x), t_cond), tgt_mask, tgt_key_padding_mask)
        #     x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
        #     x = x + self._ff_block(self.film2(self.norm3(x), t_cond))
        # else:
        #     x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        #     x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
        #     x = self.norm3(x + self._ff_block(x))

        # return x

        ######

        x = src
        # print('film encoder layer forward id cond shape', id_cond.shape)
        x = x + self._sa_block(self.film1(self.norm1(x), id_cond), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.film2(self.norm2(x), id_cond))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class DiffusionTransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    # Examples::
    #     >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    #     >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    #     >>> memory = torch.rand(10, 32, 512)
    #     >>> tgt = torch.rand(20, 32, 512)
    #     >>> out = transformer_decoder(tgt, memory)
    # """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None, time_dim=512, sinu_emb_dim=512, n_ids=None):
        super(DiffusionTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        sinu_pos_emb = SinusoidalPosEmb(sinu_emb_dim)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_emb_dim, time_dim),
            # nn.GELU(),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        if exists(n_ids):
            id_dim = time_dim
            self.id_mlp = nn.Sequential(
                nn.Linear(n_ids, id_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(id_dim, id_dim)
            )



    def forward(self, tgt: Tensor, time_cond: Tensor, memory: Tensor, id_cond: Tensor = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # print('decoder tgt tcond memory idees', tgt.shape, time_cond.shape, memory.shape, id_cond.shape)


        output = tgt

        # print('time shape', time_cond.shape)
        time_cond = self.time_mlp(time_cond)
        # print('time mlp shape', time_cond.shape)
        if exists(id_cond):
            # print('id shape', id_cond.shape)
            id_cond = self.id_mlp(id_cond)
            # print('id mlp shape', id_cond.shape)
            time_cond = torch.cat((time_cond, id_cond), dim=-1)
            # print('id shapes', id_cond.shape, time_cond.shape)

        # print('decoder shapes', output.shape, time_cond.shape)

        for mod in self.layers:
            output = mod(output, time_cond, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class DiffusionTransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=False, id_dim=512, n_ids=10):
        super(DiffusionTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor

        self.id_mlp = nn.Sequential(
            nn.Linear(n_ids, id_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(id_dim, id_dim)
        )

    def forward(self, src: Tensor, id_cond: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src
        id_cond = self.id_mlp(id_cond)
        # print('film encoder id cond shape', id_cond.shape)
        for mod in self.layers:
            output = mod(output, id_cond, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output

class DiffusionTransformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).

    Examples::
        # >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        # >>> src = torch.rand((10, 32, 512))
        # >>> tgt = torch.rand((20, 32, 512))
        # >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, d_time: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = True,
                 device=None, dtype=None, n_keys=N_KEYS, n_instruments=15, n_mels=128, n_ids=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DiffusionTransformer, self).__init__()
        print('using dtime', d_time, 'using activation', activation)
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerFiLMDecoderLayer(d_model, nhead, dim_feedforward, 2 * d_time if exists(n_ids) else d_time, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = DiffusionTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, n_ids=n_ids, time_dim=d_time)

        # self.note_embedding = nn.Linear(2 * n_instruments * n_keys, d_model)        #
        # self.spec_embedding = nn.Linear(n_mels, d_model)
        # self.final_linear = nn.Linear(d_model, n_mels)
        self.note_embedding = nn.Conv1d(2 * n_instruments * n_keys, d_model, 7, padding=3)
        self.spec_embedding = nn.Conv1d(n_mels, d_model, 7, padding=3)

        self.pe = PositionalEncoding(d_model=d_model)

        self.final_linear = nn.Conv1d(d_model, n_mels, 1, padding=0)
        self.gelu = nn.GELU()

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    # x, t, x_note_cond, idees = ids, ids_choice = zero_choice.squeeze(-1) if exists(ids) and zero_out_cond else None
    def forward(self, tgt: Tensor, t_cond: Tensor, src: Tensor, idees: Tensor = None, ids_choice: Tensor = None, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                zero_ids=False) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            # >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        # print('diffusion transformer', 'src tgt', src.shape, tgt.shape)
        src = self.note_embedding(src).permute((2, 0, 1))
        src = self.pe(src)
        # print('forward pe')
        # print('after pe src shape', src.shape)
        tgt = self.spec_embedding(tgt).permute((2, 0, 1))
        # print('tgt shape', src.shape)
        tgt = self.pe(tgt)
        # print('after pe tgt shape', tgt.shape)
        # print('diffusion transformer after embed', 'src tgt', src.shape, tgt.shape)

        # src = self.gelu(src)
        # tgt = self.gelu(tgt)
        # print('diffusion transformer after activation', 'src tgt', src.shape, tgt.shape)


        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # print('tgt tcond memory idees', tgt.shape, t_cond.shape, memory.shape, idees.shape)
        if zero_ids:
            idees = 0. * idees
            print('zero idees')
        output = self.decoder(tgt, t_cond, memory, idees, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # print('out shape', output.shape)
        output = self.final_linear(output.permute(1, 2, 0))
        # print('out shape after final', output.shape)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class DiffusionFilmTransformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).

    Examples::
        # >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        # >>> src = torch.rand((10, 32, 512))
        # >>> tgt = torch.rand((20, 32, 512))
        # >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, d_time: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = True,
                 device=None, dtype=None, n_keys=N_KEYS, n_instruments=15, n_mels=128, n_ids=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DiffusionFilmTransformer, self).__init__()
        print('using d_time', d_time, 'using activation', activation)
        d_id = d_time
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerFilmEncoderLayer(d_model, nhead, dim_feedforward, d_id,
                                                        dropout,
                                                        activation, layer_norm_eps, batch_first, norm_first,
                                                        **factory_kwargs)
            encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.encoder = DiffusionTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, id_dim=d_id, n_ids=n_ids)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerFiLMDecoderLayer(d_model, nhead, dim_feedforward, 2 * d_time if exists(n_ids) else d_time, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
            decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.decoder = DiffusionTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm, n_ids=n_ids, time_dim=d_time)

        # self.note_embedding = nn.Linear(2 * n_instruments * n_keys, d_model)        #
        # self.spec_embedding = nn.Linear(n_mels, d_model)
        # self.final_linear = nn.Linear(d_model, n_mels)
        self.note_embedding = nn.Conv1d(2 * n_instruments * n_keys, d_model, 7, padding=3)
        self.spec_embedding = nn.Conv1d(n_mels, d_model, 7, padding=3)

        self.pe = PositionalEncoding(d_model=d_model)

        self.final_linear = nn.Conv1d(d_model, n_mels, 1, padding=0)
        self.gelu = nn.GELU()

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    # x, t, x_note_cond, idees = ids, ids_choice = zero_choice.squeeze(-1) if exists(ids) and zero_out_cond else None
    def forward(self, tgt: Tensor, t_cond: Tensor, src: Tensor, idees: Tensor = None, ids_choice: Tensor = None, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                zero_ids=False) -> Tensor:
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\cdot\text{num\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\cdot\text{num\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            # >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """
        # print('diffusion transformer', 'src tgt', src.shape, tgt.shape)
        src = self.note_embedding(src).permute((2, 0, 1))
        src = self.pe(src)
        # print('forward pe')
        # print('after pe src shape', src.shape)
        tgt = self.spec_embedding(tgt).permute((2, 0, 1))
        # print('tgt shape', src.shape)
        tgt = self.pe(tgt)
        # print('after pe tgt shape', tgt.shape)
        # print('diffusion transformer after embed', 'src tgt', src.shape, tgt.shape)

        # src = self.gelu(src)
        # tgt = self.gelu(tgt)
        # print('diffusion transformer after activation', 'src tgt', src.shape, tgt.shape)


        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        if zero_ids:
            idees = 0. * idees
            print('zero idees')

        memory = self.encoder(src, idees, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # print('tgt tcond memory idees', tgt.shape, t_cond.shape, memory.shape, idees.shape)

        output = self.decoder(tgt, t_cond, memory, idees, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # print('out shape', output.shape)
        output = self.final_linear(output.permute(1, 2, 0))
        # print('out shape after final', output.shape)
        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    print('betas')
    for b in range(300):
        print(b, betas[b])
    return torch.clip(betas, 0, 0.999)


def linear_map_to_minus_one_one(left, right):
    a = 2 / (right - left)
    b = (-right -left) / (right - left)

    c = (right - left) / 2
    d = (right + left) / 2
    return lambda x: a * x + b, lambda x: c * x + d

def get_ab(left, right):
    a = 2 / (right - left)
    b = (-right - left) / (right - left)
    return a, b

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
            custom_loss=None,
            channels=229,
            self_condition=True, norm_factor=1., use_log=False, use_strict_log=False, mstd=None,
            clip_min=1e-5, clip_max=1e5,
            specs=None, wave_model=False,
            schedule_s=0.008, wave_scale=1.
    ):
        super().__init__()
        # assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # assert not model.learned_sinusoidal_cond

        self.model = model
        self.channels = channels
        self.self_condition = self_condition

        self.image_size = image_size
        self.norm_factor = norm_factor
        self.mstd = mstd
        self.wave_scale = wave_scale
        if self.mstd:
            print('mstd setting norm factor to 1')
            self.norm_factor = 1.
        self.objective = objective
        self.specs = specs
        self.wave_model = wave_model
        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            print('using cosine schedule with scheulde:', schedule_s)
            betas = cosine_beta_schedule(timesteps, s=schedule_s)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.custom_loss = custom_loss
        self.use_log = use_log
        self.use_strict_log = use_strict_log
        if self.use_strict_log:
            self.clip_min, self.clip_max = clip_min, clip_max
            self.scale_forward, self.scale_backward = linear_map_to_minus_one_one(np.log(self.clip_min), np.log(self.clip_max))
        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_note_cond = None, clip_x_start = False, idees=None, zero_ids=False):
        model_output = self.model(x, t, x_note_cond, idees=idees, zero_ids=zero_ids)
        # maybe_clip = partial(torch.clamp, min = -1., max = 1.) if overlap_interpolate or clip_x_start else identity
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_note_cond = None, clip_denoised = False, ids=None):
        preds = self.model_predictions(x, t, x_note_cond, idees=ids)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_note_cond = None, clip_denoised = False, no_cond=-1, ids=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long) if isinstance(t, int) else t
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, ids=ids, x_note_cond = x_note_cond, clip_denoised = clip_denoised,
                                                                          )
        noise = torch.randn_like(x) if (not isinstance(t, int)) or t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, midi, timesteps=None, return_x_start=False, apply_exp=True, update_mel=False, update_mel_matrix=None, ids=None, init_x=None,
                      CLIP_DENOISED=False):
        batch, device = shape[0], self.betas.device
        if not exists(init_x):
            img = torch.randn(shape, device=device)
        else:
            assert self.wave_model
            init_x = init_x * self.wave_scale
            img = init_x
        x_start = None
        timesteps = default(timesteps, self.num_timesteps)
        prev_beta = self.betas[-1].item()
        for t in tqdm(reversed(range(self.num_timesteps - timesteps, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = midi
            img, x_start = self.p_sample(img, t, self_cond, ids=ids, clip_denoised=CLIP_DENOISED)
            s = torch.quantile(torch.abs(img), 0.99)
        img = self.scale_backward(img)
        if apply_exp:
            img = torch.exp(img)
        x_start = self.scale_backward(x_start)
        if apply_exp:
            x_start = torch.exp(x_start)
        if not return_x_start:
            return img
        else:
            return img, x_start

    @torch.no_grad()
    def p_sample_loop_cfg(self, shape, midi, timesteps=None, return_x_start=False, apply_exp=True, cfg_weight=2., exp_weight=False, no_cond=-1., ids=None,
                          CLIP_DENOISED=False):
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None
        timesteps = default(timesteps, self.num_timesteps)
        for t in tqdm(reversed(range(self.num_timesteps - timesteps, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img_cat = torch.cat((img, img), dim=0)
            midi_cat = torch.cat((midi, no_cond * torch.ones_like(midi).cuda()), dim=0)
            ids_cat = torch.cat((ids, torch.zeros_like(ids).cuda()), dim=0)
            img_out_cat, x_start_out_cat = self.p_sample(img_cat, t, midi_cat, ids=ids_cat, clip_denoised=CLIP_DENOISED)
            if exp_weight:
                img_out_cat, x_start_out_cat = torch.exp(self.scale_backward(img_out_cat)), torch.exp(self.scale_backward(x_start_out_cat))
            img1, img2 = torch.chunk(img_out_cat, chunks=2, dim=0)
            x_start1, x_start2 = torch.chunk(x_start_out_cat, chunks=2, dim=0)
            if t > 0:
                img = img2 + cfg_weight * (img1 - img2)
                x_start = x_start2 + cfg_weight * (x_start1 - x_start2)
                s = torch.quantile(torch.abs(img), 0.95)
            else:
                img, x_start = img1, x_start1
            if exp_weight:
                img = self.scale_forward(torch.log(torch.clamp(img, min=self.clip_min, max=self.clip_max)))
                x_start = self.scale_forward(torch.log(torch.clamp(x_start, min=self.clip_min, max=self.clip_max)))

        img = self.scale_backward(img)
        x_start = self.scale_backward(x_start)
        if apply_exp:
            img = torch.exp(img)
            x_start = torch.exp(x_start)
        if not return_x_start:
            return img
        else:
            return img, x_start

    @torch.no_grad()
    def ddim_sample(self, shape, midi, clip_denoised = False, ids=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = midi
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised, idees=ids)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        img = self.scale_backward(img)
        x_start = self.scale_backward(x_start)
        return img

    @torch.no_grad()
    def ddim_sample_cfg(self, shape, midi, ids=None, clip_denoised=True, cfg=1.5, overlap=False, t_begin=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        if exists(t_begin):
            total_timesteps = t_begin
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            if cfg != 1:
                self_cond = torch.cat((midi, torch.zeros_like(midi, dtype=midi.dtype, device=device)), dim=0)
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                pred_noise, x_start, *_ = self.model_predictions(torch.cat((img, img), dim=0), torch.cat((time_cond, time_cond), dim=0), self_cond,
                                                                 clip_x_start=clip_denoised,
                                                                 idees=torch.cat((ids, ids), dim=0) if ids is not None else None
                                                                 )
                pred_noise1, pred_noise2 = torch.chunk(pred_noise, 2, dim=0)
                x_start1, x_start2 = torch.chunk(x_start, 2, dim=0)
                pred_noise = pred_noise2 + cfg * (pred_noise1 - pred_noise2)
            else:
                self_cond = midi
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond,
                                                                 clip_x_start=clip_denoised,
                                                                 idees=ids if ids is not None else None
                                                                 )
            x_start = self.predict_start_from_noise(img, time_cond, pred_noise)
            if overlap:
                interpolant = x_start
                b = interpolant.shape[0]
                for i_b in range(1, b):
                    overlap1 = interpolant[i_b - 1, :, -overlap:]
                    overlap2 = interpolant[i_b, :, : overlap]
                    blend_weight = torch.linspace(1, 0, overlap, device='cuda').unsqueeze(0).unsqueeze(0)  # .repeat((1, model_output.shape[1], 1))
                    blended = blend_weight * overlap1 + (1 - blend_weight) * overlap2
                    interpolant[i_b - 1, :, -overlap:] = blended
                    interpolant[i_b, :, : overlap] = blended
                torch.clamp_(interpolant, min=-1., max=1.)
            x_start = torch.clamp(x_start, min=-1., max=1.)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise


        img = self.scale_backward(img)
        x_start = self.scale_backward(x_start)
        return img

    def ddim_sample_multi_cfg(self, shape, midi, ids=None, clip_denoised=True, cfg=1.25, cfg_id=1.25,
                              overlap=False, t_begin=None, zero_ids=False, cfg_type=1,
                              ):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        if exists(t_begin):
            total_timesteps = t_begin
        assert cfg_type in [1, 2]
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None
        batch_copies = 1 + (cfg > 1) + (cfg_id > 1)
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            if batch_copies == 1:
                self_cond = midi
                id_cond = ids
            elif batch_copies == 2 and cfg > 1.:
                self_cond = torch.cat((midi, torch.zeros_like(midi, dtype=midi.dtype, device=device)), dim=0)
                id_cond = torch.cat((ids, ids), dim=0)
            elif batch_copies == 2 and cfg_id > 1.:
                self_cond = torch.cat((midi, midi), dim=0)
                id_cond = torch.cat((ids, torch.zeros_like(ids, dtype=ids.dtype, device=device)), dim=0)
            else:
                assert batch_copies == 3
                if cfg_type == 1:
                    self_cond = torch.cat((midi, torch.zeros_like(midi, dtype=midi.dtype, device=device), midi), dim=0)
                    id_cond = torch.cat((ids, ids, torch.zeros_like(ids, dtype=ids.dtype, device=device)), dim=0)
                else:
                    self_cond = torch.cat((midi, torch.zeros_like(midi, dtype=midi.dtype, device=device), torch.zeros_like(midi, dtype=midi.dtype, device=device)), dim=0)
                    id_cond = torch.cat((torch.zeros_like(ids, dtype=ids.dtype, device=device), ids, torch.zeros_like(ids, dtype=ids.dtype, device=device)), dim=0)


            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            img_cat = torch.cat((img,) * batch_copies, dim=0)
            time_cond_cat = torch.cat((time_cond,) * batch_copies, dim=0)

            pred_noise, x_start, *_ = self.model_predictions(img_cat, time_cond_cat, self_cond,
                                                             clip_x_start=clip_denoised,
                                                             idees=id_cond,
                                                             zero_ids=zero_ids
                                                             )
            if batch_copies == 3:
                pred_noise1, pred_noise2, pred_noise3 = torch.chunk(pred_noise, 3, dim=0)
                x_start1, x_start2, x_start3 = torch.chunk(x_start, 3, dim=0)
                if cfg_type == 1:
                    pred_noise = pred_noise1 + (cfg - 1.) * (pred_noise1 - pred_noise2) + (cfg_id - 1.) * (pred_noise1 - pred_noise3)
                else:
                    pred_noise = pred_noise3 + cfg * (pred_noise1 - pred_noise3) + cfg_id * (pred_noise2 - pred_noise3)
            elif batch_copies == 2:
                curr_cfg = max(cfg, cfg_id)
                pred_noise1, pred_noise2 = torch.chunk(pred_noise, 2, dim=0)
                x_start1, x_start2 = torch.chunk(x_start, 2, dim=0)
                pred_noise = pred_noise2 + curr_cfg * (pred_noise1 - pred_noise2)
            else:
                assert batch_copies == 1
            x_start = self.predict_start_from_noise(img, time_cond, pred_noise)

            if overlap:
                interpolant = x_start
                b = interpolant.shape[0]
                for i_b in range(1, b):
                    overlap1 = interpolant[i_b - 1, :, -overlap:]
                    overlap2 = interpolant[i_b, :, : overlap]
                    blend_weight = torch.linspace(1, 0, overlap, device='cuda').unsqueeze(0).unsqueeze(0)  # .repeat((1, model_output.shape[1], 1))
                    blended = blend_weight * overlap1 + (1 - blend_weight) * overlap2
                    interpolant[i_b - 1, :, -overlap:] = blended
                    interpolant[i_b, :, : overlap] = blended
                torch.clamp_(interpolant, min=-1., max=1.)

            # x_start = self.predict_start_from_noise(img, time_cond, pred_noise)
            x_start = torch.clamp(x_start, min=-1., max=1.)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise


        img = self.scale_backward(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'my_loss':
            return self.custom_loss
        elif self.loss_type == 'spectral_reconstruction':
            return lambda x, x_target: spectral_reconstruction_loss(x, x_target, specs=self.specs) #+ F.l1_loss(x, x_target)
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, midi, noise = None, save_output=False, logdir=None,
                 mel_fmin=MEL_FMIN_THIS,
                 save_hop_length=HOP_LENGTH, save_n_fft=N_FFT, save_win_length=WINDOW_LENGTH_THIS,
                 zero_out_cond=0.,
                 ids=None
                 ):
        b, c, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        timenow = datetime.now().strftime('%y%m%d-%H%M%S')

        x_note_cond = midi
        if zero_out_cond:
            zero_choice = (torch.rand((x_note_cond.shape[0], 1, 1)) > zero_out_cond).cuda()
            x_note_cond = zero_choice * x_note_cond

        q_sample_clamped_input = torch.clamp(x_start, min=-1., max=1.)  # if not exists(self.mstd) else x_start
        q_sample_choice = (torch.rand((x_start.shape[0], 1, 1)) >= 0.1).cuda()
        q_sample_clamped_input = q_sample_choice * x_start + (~q_sample_choice) * q_sample_clamped_input
        x = self.q_sample(x_start=q_sample_clamped_input, t=t, noise=noise)

        # predict and take gradient step
        model_out = self.model(x, t, x_note_cond, idees=ids, ids_choice=zero_choice.squeeze(-1) if exists(ids) and zero_out_cond else None)
        if save_output:
            augs = [
                    '_cond', not zero_out_cond or zero_choice[0].item(),
                    '_unclipped', q_sample_choice[0].item(),
                    ]
            augs = ','.join(str(elem) for elem in augs)
            if self.objective == 'pred_noise':
                test_pred = model_out[0:1, ...].detach()
                x_test_inp = x[0:1, ...].detach()
                t_test = t[0:1, ...].detach()
                test_pred = self.predict_start_from_noise(x_test_inp, t_test, test_pred)
                test_pred = test_pred[0].detach().cpu().numpy()
                noisy_version = x[0].detach().cpu().numpy()
            else:
                test_pred = model_out[0].detach().cpu().numpy()
                noisy_version = x[0].detach().cpu().numpy()
            test_pred = np.exp(np.clip(self.scale_backward(test_pred), a_min=np.log(1e-5), a_max=np.log(1e5)))

            try:
                if not self.objective == 'pred_noise':
                    inverse = librosa.feature.inverse.mel_to_audio(test_pred, sr=SAMPLE_RATE, power=1., hop_length=save_hop_length, htk=True, fmin=mel_fmin, fmax=MEL_FMAX,
                                                                   n_fft=save_n_fft, win_length=save_win_length, norm=MEL_NORM)
                    wav_write(logdir + '/test_spec/test_{}_{}_t{}_augs{}.flac'.format(self.objective, timenow, t[0].item(), augs), SAMPLE_RATE, inverse)
                    noisy_inverse = librosa.feature.inverse.mel_to_audio(noisy_version, sr=SAMPLE_RATE, power=1., hop_length=save_hop_length, htk=True, fmin=mel_fmin, fmax=MEL_FMAX,
                                                                   n_fft=save_n_fft, win_length=save_win_length, norm=MEL_NORM)
                    wav_write(logdir + '/test_spec/test_noisy_{}_{}_t{}_augs{}.flac'.format(self.objective, timenow, t[0].item(), augs), SAMPLE_RATE, noisy_inverse)

            except:
                pass

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
        model_out_regular_loss = model_out
        target_regular_loss = target
        loss = self.loss_fn(model_out_regular_loss, target_regular_loss, reduction = 'none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss_weights = extract(self.p2_loss_weight, t, loss.shape)
        loss = loss * loss_weights
        return loss.mean()

    def forward(self, img, midi, mask_alpha=None, mask_gamma=None, mel_fmin=MEL_FMIN, self_input=False,
                save_hop_length=HOP_LENGTH, save_n_fft=N_FFT, save_win_length=WINDOW_LENGTH_THIS, zero_out_cond=0.1, no_cond_val=-1,
                ids=None,
                *args, **kwargs):
        b, c, w, device, img_size = *img.shape, img.device, self.image_size
        assert w == img_size[0], f'height and width of image must be {img_size} but is {w}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long().cuda()
        img = self.scale_forward(torch.log(torch.clamp(img, min=self.clip_min, max=self.clip_max)))
        return self.p_losses(img, t, midi, mask_alpha=mask_alpha, mask_gamma=mask_gamma, mel_fmin=mel_fmin, self_input=self_input,
                             save_hop_length=save_hop_length, save_n_fft=save_n_fft, save_win_length=save_win_length,
                             zero_out_cond=zero_out_cond, no_cond_val=no_cond_val,
                             ids=ids,
                             *args, **kwargs)