"""
model.py
========
配球序列模型：輸入最近 6 顆球 + 情境，輸出本球 7 類事件的機率 (outcomes.EVENTS)。

架構：
  token  = 數值特徵投影 + 球種族 embedding + 打者反應 embedding + 位置 embedding
  序列   = Transformer encoder (帶 padding mask)
  摘要   = 本球的 token 表示 + 整段序列的 attention pooling
  情境   = 數值特徵投影 + 類別 embedding (球數、壘包出局、左右對戰、前一打席)
  輸出   = MLP -> 7 個 logits -> softmax

從 7 類機率可以換算任何條件機率 (例如 P(揮棒)、P(強擊 | 打進場))，見 evaluate.py。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from modeling import features as F
from modeling.outcomes import EVENTS


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    ctx_cat_dim: int = 16


class PitchSequenceModel(nn.Module):
    def __init__(self, n_token_num: int, n_ctx_num: int, cfg: ModelConfig, seq_len: int = F.SEQ_LEN):
        super().__init__()
        d = cfg.d_model
        self.token_proj = nn.Linear(n_token_num, d)
        self.family_emb = nn.Embedding(len(F.FAMILY_VOCAB), d, padding_idx=0)
        self.outcome_emb = nn.Embedding(len(F.OUTCOME_VOCAB), d, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, dim_feedforward=4 * d, dropout=cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.attn_pool = nn.Linear(d, 1)

        self.ctx_num_proj = nn.Linear(n_ctx_num, d)
        self.ctx_cat_embs = nn.ModuleList(
            nn.Embedding(len(F.CAT_VOCABS[c]), cfg.ctx_cat_dim, padding_idx=0) for c in F.CTX_CAT)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(d + cfg.ctx_cat_dim * len(F.CTX_CAT), d), nn.GELU(), nn.Dropout(cfg.dropout))

        self.head = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, len(EVENTS)))

    def forward(self, token_num, token_cat, pad, ctx_num, ctx_cat):
        x = (self.token_proj(token_num) + self.family_emb(token_cat[..., 0])
             + self.outcome_emb(token_cat[..., 1]) + self.pos_emb)
        h = self.encoder(x, src_key_padding_mask=pad)
        target = h[:, -1]
        w = self.attn_pool(h).squeeze(-1).masked_fill(pad, float("-inf"))
        pooled = (torch.softmax(w, dim=1).unsqueeze(-1) * h).sum(1)

        cat = torch.cat([emb(ctx_cat[:, i]) for i, emb in enumerate(self.ctx_cat_embs)], dim=-1)
        ctx = self.ctx_mlp(torch.cat([torch.relu(self.ctx_num_proj(ctx_num)), cat], dim=-1))
        return self.head(torch.cat([target, pooled, ctx], dim=-1))
