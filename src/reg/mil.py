ORGANS = ["breast","colon","stomach","prostate","bladder","lung","cervix"]
#!/usr/bin/env python3
"""Phase 2: multi-head ABMIL classifier on UNI2 features for REG2026 Metric A.

Shared gated-attention aggregator over a slide's UNI2 patch features [N,1536] ->
slide embedding -> 85 organ-masked classification heads (one per predictable
question). Per-sample loss is summed only over heads applicable to that slide.
Class imbalance handled with inverse-frequency class weights per head.
"""
import os, json, time, math, argparse, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BASE = "/workspace/sota_run"
FEAT_DIR = f"{BASE}/features/uni2-h_20x_256"


# ----------------------------- data -----------------------------
class GatedAttn(nn.Module):
    """ABMIL gated attention pooling."""
    def __init__(self, dim, hid=512):
        super().__init__()
        self.V = nn.Linear(dim, hid); self.U = nn.Linear(dim, hid)
        self.w = nn.Linear(hid, 1)

    def forward(self, x):                      # x: [N, dim]
        a = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        a = torch.softmax(self.w(a), dim=0)    # [N,1]
        return (a * x).sum(0)                  # [dim]


class TransMILPool(nn.Module):
    """Lightweight TransMIL-style: a couple of self-attention layers over patches
    (subsampled) + cls-token readout. O(N^2) so we cap N at forward time."""
    def __init__(self, dim, heads=8, layers=2, cap=2048):
        super().__init__()
        self.cap = cap
        self.cls = nn.Parameter(torch.randn(1, dim) * 0.02)
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.1, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(enc, layers)

    def forward(self, x):                      # [N, dim]
        if x.shape[0] > self.cap:
            idx = torch.randperm(x.shape[0], device=x.device)[:self.cap]
            x = x[idx]
        seq = torch.cat([self.cls, x], 0).unsqueeze(0)   # [1, N+1, dim]
        out = self.tr(seq)[0, 0]                          # cls token
        return out


class MambaMILPool(nn.Module):
    """MambaMIL-style sequence aggregator. mamba-ssm needs nvcc (absent here), so
    we use a pure-PyTorch bidirectional GRU scan over patches as a no-compile
    stand-in for the selective state-space scan, then attention-pool the outputs."""
    def __init__(self, dim, cap=2048):
        super().__init__()
        self.cap = cap
        self.rnn = nn.GRU(dim, dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = GatedAttn(dim, dim)

    def forward(self, x):                      # [N, dim]
        if x.shape[0] > self.cap:
            x = x[torch.randperm(x.shape[0], device=x.device)[:self.cap]]
        h, _ = self.rnn(x.unsqueeze(0))        # [1, N, dim]
        return self.attn(h[0])


class CLAMPool(nn.Module):
    """ABMIL attention + instance-level attention scores exposed for an auxiliary
    top-k instance loss (CLAM-style clustering regularization)."""
    def __init__(self, dim, hid=512):
        super().__init__()
        self.V = nn.Linear(dim, hid); self.U = nn.Linear(dim, hid)
        self.w = nn.Linear(hid, 1)

    def forward(self, x):
        a = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        s = self.w(a)                          # [N,1] raw scores
        att = torch.softmax(s, dim=0)
        return (att * x).sum(0), s.squeeze(1)  # emb, instance scores


class FiLM(nn.Module):
    """Organ-conditioning: organ embedding -> per-channel scale+shift on slide emb."""
    def __init__(self, n_organ, dim):
        super().__init__()
        self.emb = nn.Embedding(n_organ, dim)
        self.gamma = nn.Linear(dim, dim); self.beta = nn.Linear(dim, dim)

    def forward(self, h, organ_idx):
        e = self.emb(organ_idx)
        return h * (1 + self.gamma(e)) + self.beta(e)


class MultiHeadMIL(nn.Module):
    def __init__(self, dim, head_classes, hid=512, agg="abmil", n_organ=7, use_film=True, agg_cap=2048):
        super().__init__()
        self.agg_name = agg
        self.use_film = use_film
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.ReLU(), nn.Dropout(0.25))
        if agg == "abmil":
            self.agg = GatedAttn(hid, hid)
        elif agg == "transmil":
            self.agg = TransMILPool(hid, cap=agg_cap)
        elif agg == "clam":
            self.agg = CLAMPool(hid, hid)
        elif agg == "mambamil":
            self.agg = MambaMILPool(hid, cap=agg_cap)
        else:
            raise ValueError(agg)
        self.film = FiLM(n_organ, hid) if use_film else None
        self.heads = nn.ModuleList([nn.Linear(hid, c) for c in head_classes])
        # CLAM instance classifier: binary (salient vs non-salient patch)
        self.inst_clf = nn.Linear(hid, 2) if agg == "clam" else None

    def forward(self, feats, organ_idx=None):
        embs, inst = [], []
        for x in feats:
            h = self.proj(x)
            if self.agg_name == "clam":
                e, s = self.agg(h)
                inst.append((h, s))            # keep patch feats + attention scores
            else:
                e = self.agg(h)
            embs.append(e)
        emb = torch.stack(embs)                # [B, hid]
        if self.film is not None and organ_idx is not None:
            emb = self.film(emb, organ_idx)
        logits = [head(emb) for head in self.heads]
        return logits, inst

    def clam_instance_loss(self, inst, k=8):
        """CLAM-style instance clustering: top-k attention patches should be class
        'salient' (1), bottom-k 'non-salient' (0). Trains attention to localize."""
        if self.inst_clf is None or not inst:
            return torch.zeros((), device=next(self.parameters()).device)
        loss = 0.0; cnt = 0
        for h, s in inst:                       # h:[N,hid], s:[N]
            n = h.shape[0]
            kk = min(k, n // 2)
            if kk < 1:
                continue
            order = s.argsort(descending=True)
            top, bot = order[:kk], order[-kk:]
            xi = torch.cat([h[top], h[bot]], 0)
            yi = torch.cat([torch.ones(kk, dtype=torch.long), torch.zeros(kk, dtype=torch.long)]).to(h.device)
            loss = loss + F.cross_entropy(self.inst_clf(xi), yi)
            cnt += 1
        return loss / max(1, cnt)

