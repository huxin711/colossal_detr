import copy
import torch
import torch.nn.functional as F
from torch import nn
import math
from colossalai.registry import LAYERS, MODELS
from colossalai import nn as col_nn


@MODELS.register_module
class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=512, dropout=0.1,
                 return_intermediate=False):
        super().__init__()
        self.encoder = TransformerEncoder(d_model, dim_feedforward, num_encoder_layers, nhead, dropout)
        self.decoder = TransformerDecoder(d_model, dim_feedforward, num_decoder_layers, nhead, dropout, return_intermediate)

        self.d_model = d_model
        self.nhead = nhead


    def forward(self, src, query_embed, pos_embed):

        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)
        tgt = torch.zeros_like(query_embed)

        memory = self.encoder(src, pos_embed)  # [N,B,256]

        hs = self.decoder(tgt, memory, pos_embed, query_embed)


        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)

@LAYERS.register_module
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_encoder_layers, nhead, dropout):
        super().__init__()
        self.en_layers = num_encoder_layers
        self.layers = get_clones(TransformerEncoderLayer(d_model, nhead, dropout, dim_feedforward), num_encoder_layers)
        self.norm = col_nn.LayerNorm(d_model)

    def forward(self, src, pos):  # src,pos:[hw,B,256] mask:[B,hw]

        src = src if pos is None else (src+pos)
        for i in range(self.en_layers):
            src = self.layers[i](src).transpose(0,1)
        return self.norm(src)

@LAYERS.register_module
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super().__init__()
        self.norm_1 = col_nn.LayerNorm(d_model)
        self.norm_2 = col_nn.LayerNorm(d_model)
        self.selfAttn = MultiHeadAttention(d_model, nhead, dropout)
        self.feedForward = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout_1 = col_nn.Dropout(dropout)
        self.dropout_2 = col_nn.Dropout(dropout)
    def forward(self, x):  # [hw,B,256] [B,hw]
        x = x.transpose(0, 1)
        x1 = self.norm_1(x)
        x = x + self.dropout_1(self.selfAttn(x1, x1, x1))
        x2 = self.norm_2(x)
        out = x + self.dropout_2(self.feedForward(x2))

        return out

@LAYERS.register_module
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, dim_feedforward, num_decoder_layers, nhead, dropout, return_intermediate=True):
        super().__init__()
        self.de_layers = num_decoder_layers
        self.layers = get_clones(TransformerDecoderLayer(d_model, nhead, dropout, dim_feedforward), num_decoder_layers)
        self.norm = col_nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos, query_pos):  # src,pos:[hw,B,256] mask:[B,hw]
        # tgt = tgt if query_pos is None else (tgt + query_pos)

        intermediate = []

        for i in range(self.de_layers):
            tgt = self.layers[i](tgt, memory, pos, query_pos).transpose(0, 1)

            if self.return_intermediate:
                intermediate.append(self.norm(tgt))


        return torch.stack(intermediate)

@LAYERS.register_module
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout, dim_feedforward):
        super().__init__()
        self.self_attn_1 = MultiHeadAttention(d_model, nhead, dropout)
        self.self_attn_2 = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = col_nn.LayerNorm(d_model)
        self.norm2 = col_nn.LayerNorm(d_model)
        self.norm3 = col_nn.LayerNorm(d_model)
        self.dropout1 = col_nn.Dropout(dropout)
        self.dropout2 = col_nn.Dropout(dropout)
        self.dropout3 = col_nn.Dropout(dropout)

        self.ff = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, tgt, memory, pos, query_pos):
        memory = memory.transpose(0, 1)
        tgt1 = tgt if query_pos is None else (tgt + query_pos)
        tgt1 = tgt1.transpose(0, 1)
        x2 = self.norm1(tgt1)
        tgt = tgt + self.dropout1(self.self_attn_1(x2, x2, tgt))
        x2 = self.norm2(tgt)
        tgt = tgt + self.dropout2(self.self_attn_2(x2+query_pos, memory+pos, memory))
        x2 = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.ff(x2))
        return tgt

@LAYERS.register_module
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(MultiHeadAttention, self).__init__()
        if d_model % nhead != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (d_model, nhead))

        self.nhead = nhead  # 8
        self.attention_head_size = int(d_model / nhead)  # 16  每个注意力头的维度
        self.all_head_size = int(self.nhead * self.attention_head_size)
        self.query = col_nn.Linear(d_model, self.all_head_size)  # 128, 128
        self.key = col_nn.Linear(d_model, self.all_head_size)
        self.value = col_nn.Linear(d_model, self.all_head_size)
        self.dropout = col_nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        # x'shape = [bs, seqlen, hid_size]
        new_x_shape = x.size()[:-1] + (self.nhead, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, q,k,v):
        mixed_query_layer = self.query(q)
        mixed_key_layer = self.key(k)
        mixed_value_layer = self.value(v)
        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]

        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

@LAYERS.register_module
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear_1 = col_nn.Linear(d_model, dim_feedforward)
        self.ff_drop = col_nn.Dropout(dropout)
        self.linear_2 = col_nn.Linear(dim_feedforward, d_model)
    def forward(self, x):

        x = self.ff_drop(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

def get_clones(module, num_encoder_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num_encoder_layers)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        return_intermediate=True,
    )
