"""字符级 GPT：手写注意力，只依赖 PyTorch。运行 python train.py --help。"""

import argparse
from array import array
import hashlib
import math
from pathlib import Path
import signal
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
PRESETS = {
    "tiny": dict(context=128, width=128, layers=4, heads=4, dropout=0.1),
    "small": dict(context=256, width=384, layers=6, heads=6, dropout=0.1),
}


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["width"]
        self.heads = cfg["heads"]
        if width % self.heads:
            raise ValueError("width 必须能被 heads 整除。")
        self.head_dim = width // self.heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(cfg["dropout"])
        mask = torch.tril(torch.ones(cfg["context"], cfg["context"], dtype=torch.bool))
        # mask 可以重建，无需写入 checkpoint。
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B,T,C] -> [B,H,T,D]，每个头独立计算相关程度。
        q, k, v = [
            z.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
            for z in (q, k, v)
        ]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.dropout(self.proj(out))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["width"]
        self.norm1 = nn.LayerNorm(width)
        self.attn = Attention(cfg)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width), nn.Dropout(cfg["dropout"]),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.context = cfg["context"]
        width = cfg["width"]
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(self.context, width)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError("输入必须是 [batch, length]，且 1 <= length <= context。")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)
        x = self.norm(self.blocks(x))

