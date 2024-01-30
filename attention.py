import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from entmax_bisect import entmax_bisect


class Attention(nn.Module):
    def forward(self, query, key, value, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class SparseAttention(nn.Module):
    def forward(self, query, key, value, alpha, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        p_attn = entmax_bisect(scores, alpha, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

