"""Gated attention MIL, completing the supplied notebook architecture."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class Classifier1FC(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim):
        super().__init__()
        self.fc = nn.Linear(n_channels, m_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class AttentionGated(nn.Module):
    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        return self.attention_weights_w(a_v * a_u).transpose(1, 2)


class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, return_attention=False, mask=None):
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError('Expected nonempty x with shape [B,N,D].')
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('mask must be boolean [B,N].')
        if not mask.any(dim=1).all():
            raise ValueError('Each bag must contain at least one valid patch.')
        # Clear padded input before projection as well as masking its attention.
        med_feat = self.dimreduction(x.masked_fill(~mask.unsqueeze(-1), 0))
        a_logits = self.attention(med_feat).masked_fill(~mask.unsqueeze(1), -torch.inf)
        attention = F.softmax(a_logits, dim=-1)
        bag_feat = torch.bmm(attention, med_feat).squeeze(1)
        logits = self.classifier(bag_feat)
        return (logits, attention) if return_attention else logits
