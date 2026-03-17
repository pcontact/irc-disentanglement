#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_common import FEATURES, load_embeddings


def apply_nonlin(x, nonlin):
    if nonlin == "linear":
        return x
    if nonlin == "tanh":
        return torch.tanh(x)
    if nonlin == "cube":
        return x ** 3
    if nonlin == "logistic":
        return torch.sigmoid(x)
    if nonlin == "relu":
        return F.relu(x)
    if nonlin == "elu":
        return F.elu(x)
    if nonlin == "selu":
        return F.selu(x)
    if nonlin == "softsign":
        return F.softsign(x)
    if nonlin == "swish":
        return x * torch.sigmoid(x)
    raise ValueError("Unknown non-linearity: {}".format(nonlin))


class DisentanglementModel(nn.Module):
    def __init__(self, word_vectors, hidden_dim, num_layers, nonlin, dropout):
        super().__init__()

        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.use_words = False
        self.word_dim = 0
        self.word_emb = None

        input_dim = FEATURES
        if word_vectors:
            _, _, weights = load_embeddings(word_vectors)
            self.word_dim = weights.shape[1]
            self.word_emb = nn.Embedding(weights.shape[0], weights.shape[1])
            self.word_emb.weight.data.copy_(torch.from_numpy(weights))
            self.use_words = True
            input_dim += 4 * self.word_dim

        self.hidden = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.hidden.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim

    def _pool_query(self, query_ids, query_lens):
        q_emb = self.word_emb(query_ids)  # (B, L, D)
        device = q_emb.device
        max_len = q_emb.shape[1]
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        q_mask = (arange < query_lens.unsqueeze(1)).unsqueeze(-1)
        q_mask_f = q_mask.float()

        q_emb_masked = q_emb * q_mask_f
        q_sum = q_emb_masked.sum(dim=1)
        denom = query_lens.unsqueeze(1).clamp(min=1).float()
        q_mean = q_sum / denom

        q_emb_for_max = q_emb.masked_fill(~q_mask, -1e9)
        q_max = q_emb_for_max.max(dim=1).values
        return q_max, q_mean

    def _pool_candidates(self, cand_ids, cand_lens):
        c_emb = self.word_emb(cand_ids)  # (B, C, L, D)
        device = c_emb.device
        max_len = c_emb.shape[2]
        arange = torch.arange(max_len, device=device).view(1, 1, -1)
        c_mask = (arange < cand_lens.unsqueeze(2)).unsqueeze(-1)
        c_mask_f = c_mask.float()

        c_emb_masked = c_emb * c_mask_f
        c_sum = c_emb_masked.sum(dim=2)
        denom = cand_lens.unsqueeze(2).clamp(min=1).float()
        c_mean = c_sum / denom

        c_emb_for_max = c_emb.masked_fill(~c_mask, -1e9)
        c_max = c_emb_for_max.max(dim=2).values
        zero_mask = cand_lens.unsqueeze(2) == 0
        c_max = torch.where(zero_mask, torch.zeros_like(c_max), c_max)
        return c_max, c_mean

    def forward(self, features, query_ids, cand_ids, query_lens, cand_lens):
        # features: (B, C, F)
        # query_ids: (B, Lq)
        # cand_ids: (B, C, Lc)
        if self.use_words:
            q_max, q_mean = self._pool_query(query_ids, query_lens)
            c_max, c_mean = self._pool_candidates(cand_ids, cand_lens)
            q_max = q_max.unsqueeze(1).expand(-1, features.shape[1], -1)
            q_mean = q_mean.unsqueeze(1).expand(-1, features.shape[1], -1)
            inputs = torch.cat([features, q_max, q_mean, c_max, c_mean], dim=-1)
        else:
            inputs = features

        if self.dropout is not None:
            inputs = self.dropout(inputs)

        h = inputs
        for layer in self.hidden:
            h = layer(h)
            h = apply_nonlin(h, self.nonlin)

        scores = h.sum(dim=-1)
        return scores

