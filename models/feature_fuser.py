import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange
from bisect import bisect_right
# from torchvision import models
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

c = copy.deepcopy

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4, out_channel=None):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if out_channel is None:
            out_channel = channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y = None):
        if y is None:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
        else:
            y = self.fc(y)
            b, e, c = y.size()
            y =  torch.mean(y, 1).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Decoder(nn.Module):
    "Core Decoder is a stack of N layers"

    def __init__(self, layer, N, position_im=None, position_ex=None, linear_kv=None, linear_q=None, outputEachLayer=False):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.position_im = position_im
        self.position_ex = position_ex
        self.fc_kv = linear_kv
        self.fc_q = linear_q
        self.outputEachLayer = outputEachLayer

    def forward(self, x, kv=None, mask = None, scales=None):
        "Pass the input (and mask) through each layer in turn."
        if self.fc_q is not None:
            x = self.fc_q(x)
        if self.fc_kv!=None:
            kv = self.fc_kv(kv)

        if self.position_im is not None:
            x = self.position_im(x)
                
        elif self.position_ex is not None:
            if scales is not None:
                x = self.position_ex(x, scales)

        if kv is not None and self.position_ex is not None:
            if scales is not None:
                kv = self.position_ex(kv, scales)

        if kv is None:
            kv = x
        for i,layer in enumerate(self.layers):
            x = layer(kv, x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, kv, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if kv is not None:
            return x + self.dropout(sublayer(self.norm(kv),self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x)))


class DecoderLayer(nn.Module):
    "Decoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, kv, x, mask = None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](kv, x, lambda kv, x: self.self_attn(x, kv, kv, mask))
        return self.sublayer[1](None, x, self.feed_forward)
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:, :emb.size(1)]
        else:
            emb = emb + self.pe[:, step]
        emb = self.dropout(emb)
        return emb

class ExamplarPositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_level=20):
        super(ExamplarPositionalEncoding, self).__init__()
        
        self.dim = dim
        self.max_level = max_level
        pe_h = torch.rand(max_level, dim//2)
        pe_w = torch.rand(max_level, dim//2)
        self.pe_h = nn.Parameter(pe_h)
        self.pe_w = nn.Parameter(pe_w)
        self.dropout = nn.Dropout(p=dropout)
        
        self.h_split = [12,  15,  19,  22,  25,  29,  33,  37,  41,  46,  51,  56,  63,  69, 76,  85,  97, 116, 148, 100000] # 2, 384
        self.w_split = [14,  18,  21,  25,  28,  31,  35,  40,  45,  50,  56,  62,  70,  78, 86,  96, 110, 125, 157, 100000] # 5, 683

    def forward(self, emb, scales):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """
        h_scales, w_scales = scales
        n, ex, f = emb.shape
        h_scales = h_scales.reshape(n*ex)
        w_scales = w_scales.reshape(n*ex)
        level_h = torch.tensor([bisect_right(self.h_split, int(h_scale)) for h_scale in h_scales], dtype=torch.long)
        level_h = torch.clamp(level_h, 0, self.max_level-1)
        level_w = torch.tensor([bisect_right(self.w_split, int(w_scale)) for w_scale in w_scales], dtype=torch.long)
        level_w = torch.clamp(level_w, 0, self.max_level-1)
        pe = torch.cat([self.pe_h[level_h],  self.pe_w[level_w]], dim=1)
        emb = emb + pe.reshape(n, ex, f)

        emb = self.dropout(emb)
        return emb
    
class ExamplarSA(nn.Module):
    def __init__(self, N=1, tr_token_dim=2048, d_ff=4096, d_feature=2048, h=8, dropout=0.1):
        super().__init__()
        attn = MultiHeadedAttention(h, tr_token_dim)
        ff = PositionwiseFeedForward(tr_token_dim, d_ff, dropout)
        position = ExamplarPositionalEncoding(dropout, d_feature)
        linear = nn.Linear(d_feature,tr_token_dim) if d_feature != tr_token_dim else None
        self.model = Decoder(DecoderLayer(tr_token_dim, c(attn), c(ff), dropout), N, position_ex=c(position), linear_q=c(linear))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, x, h_scales=None, w_scales=None):
        if h_scales is None or w_scales is None:
            scales = None
        else:
            scales = [h_scales, w_scales]
        return self.model(x, scales=scales)


class Fusion(nn.Module):
    def __init__(self, layer_dim=[512,1024], N=4, tr_token_dim=2048, d_ff=4096, d_kvfeature=2048, d_qfeature=2048, h=8, dropout=0.1, outputEachLayer=False):
        super(Fusion, self).__init__()
        attn = MultiHeadedAttention(h, tr_token_dim)
        ff = PositionwiseFeedForward(tr_token_dim, d_ff, dropout)
        position_im = PositionalEncoding(dropout, tr_token_dim)
        linear_kv = nn.Linear(d_kvfeature,tr_token_dim) if d_kvfeature != tr_token_dim else None
        linear_q = nn.Linear(d_qfeature,tr_token_dim) if d_qfeature != tr_token_dim else None
        
        self.model = Decoder(DecoderLayer(tr_token_dim, c(attn), c(ff), dropout), 
                            N=N, position_im=c(position_im), position_ex=None, 
                            linear_kv=c(linear_kv), linear_q=c(linear_q), outputEachLayer=outputEachLayer)
        
        self.layer_fcs = nn.ModuleList([nn.Linear(dim, d_kvfeature) if dim!=d_kvfeature else None for dim in layer_dim])
        self.layer_dim = layer_dim
        
        self.examplar_sa = ExamplarSA(1, d_kvfeature, tr_token_dim, d_kvfeature, h, dropout)
        self.channel_weight_model = SELayer(d_kvfeature, out_channel=tr_token_dim)

    def forward(self, image_features, exampler_features, h_scales=None, w_scales=None):
        for i, (exampler_feature, layer_fc) in enumerate(zip(exampler_features, self.layer_fcs)):
            assert self.layer_dim[i]==exampler_feature.size(1), "layer dim not match"
            if layer_fc is not None:
                exampler_feature = layer_fc(exampler_feature.view(exampler_feature.size(0),-1))
            else:
                exampler_feature = exampler_feature.view(exampler_feature.size(0), -1)
            exampler_features[i] = exampler_feature

        exampler_features = torch.cat(exampler_features, dim=0)
        if len(exampler_features.shape) == 2:
            exampler_features = exampler_features.unsqueeze(0)

        exampler_features = exampler_features.reshape(exampler_features.shape[0], exampler_features.shape[1], -1)
        exampler_features = self.examplar_sa(exampler_features, h_scales, w_scales)
        
        features_shape = image_features.shape
        pathes = rearrange(image_features, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=1, s2=1)
        pathes = self.model(pathes,exampler_features, scales=[h_scales, w_scales])
        image_features = rearrange(pathes, 'b (h w) (s1 s2 c) -> b c (h s1) (w s2)', s1=1, s2=1, h=features_shape[2], w=features_shape[3])
        image_features = self.channel_weight_model(image_features, exampler_features)
        return image_features