"""
models.py: AAM-TRANS, its matched vanilla Transformer control, the
single-component ablations, and two non-attention baselines.

Every attention backbone is built by one class, TabTransformer, from four
switches. The vanilla Transformer and AAM-TRANS differ in exactly these four
switches and in nothing else (same depth, width, heads, FFN, norm placement,
dropout, CLS pooling and classifier):

  switch       vanilla Transformer        AAM-TRANS
  tokenizer    'patch'  (T = 8 groups,    'feature' (T = d tokens,
                shared linear embedding)   one embedding per feature)
  pe           'sinusoidal' (fixed)       'learned'
  gate         False                      True  (input-conditioned per-head)
  aux          False                      True  (reconstruction head)

Each ablation turns exactly one switch back to its vanilla setting.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, N_PATCH = 96, 4, 3, 384, 0.1, 8


# ── Tokenisers ──────────────────────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    """Token j = x_j * w_j + b_j: one learned embedding per input feature."""

    def __init__(self, d: int, dm: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d, dm) / math.sqrt(dm))
        self.b = nn.Parameter(torch.zeros(d, dm))
        self.n_tokens = d

    def forward(self, x):                       # [B, d] -> [B, d, dm]
        return x.unsqueeze(-1) * self.w + self.b


class PatchTokenizer(nn.Module):
    """Split x into N_PATCH contiguous groups (zero-padded) and embed every
    group with one shared linear map: the standard way a vanilla Transformer
    is applied to a flat feature vector."""

    def __init__(self, d: int, dm: int, n_patch: int = N_PATCH):
        super().__init__()
        self.g = math.ceil(d / n_patch)
        self.pad = self.g * n_patch - d
        self.proj = nn.Linear(self.g, dm)
        self.n_tokens = n_patch

    def forward(self, x):                       # [B, d] -> [B, P, dm]
        if self.pad:
            x = F.pad(x, (0, self.pad))
        return self.proj(x.view(x.size(0), -1, self.g))


def sinusoidal(n: int, dm: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dm, 2).float() * (-math.log(10000.0) / dm))
    pe = torch.zeros(n, dm)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


# ── Attention with optional adaptive per-head gate ──────────────────────────
class GatedMHA(nn.Module):
    """Multi-head self-attention with an optional gate.

    gate='record' (v1): every head output A_h is scaled by
        gamma_h(h) = sigmoid(MLP(mean_t h_t))_h, one weight per record and head.
    gate='token'  (v2): every value vector V_{h,t} is scaled by
        gamma_{h,t} = sigmoid(w_h . h_t + b_h), one weight per head and token
        (i.e. per input feature), so a head can stop reading from a feature.
    """

    def __init__(self, dm: int, H: int, dropout: float, gate):
        super().__init__()
        self.H, self.dk = H, dm // H
        self.qkv = nn.Linear(dm, 3 * dm)
        self.out = nn.Linear(dm, dm)
        self.drop = nn.Dropout(dropout)
        self.gate = "record" if gate is True else (gate or None)
        if self.gate == "record":
            self.gate_mlp = nn.Sequential(nn.Linear(dm, dm // 4), nn.ReLU(),
                                          nn.Linear(dm // 4, H))
        elif self.gate == "token":
            self.gate_proj = nn.Linear(dm, H)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)   # start open (~0.88)
        self.last_gate = None                   # [B, H] summary, for inspection
        self.live_gate = None                   # differentiable, for gate losses

    def forward(self, h):
        B, T, dm = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.H, self.dk).permute(2, 0, 3, 1, 4)
        if self.gate == "token":
            g = torch.sigmoid(self.gate_proj(h)).transpose(1, 2)   # [B, H, T]
            self.live_gate = g
            self.last_gate = g.mean(-1).detach()
            v = v * g.unsqueeze(-1)
        a = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dk), dim=-1)
        o = self.drop(a) @ v                    # [B, H, T, dk]
        if self.gate == "record":
            g = torch.sigmoid(self.gate_mlp(h.mean(dim=1)))   # [B, H]
            self.live_gate = g
            self.last_gate = g.detach()
            o = o * g[:, :, None, None]
        return self.out(o.transpose(1, 2).reshape(B, T, dm))


class Block(nn.Module):
    """Post-norm encoder block (Eq. block in the paper)."""

    def __init__(self, dm, H, dff, dropout, gate):
        super().__init__()
        self.attn = GatedMHA(dm, H, dropout, gate)
        self.ffn = nn.Sequential(nn.Linear(dm, dff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(dff, dm))
        self.n1, self.n2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        h = self.n1(h + self.drop(self.attn(h)))
        return self.n2(h + self.drop(self.ffn(h)))


class TabTransformer(nn.Module):
    def __init__(self, d: int, n_classes: int, tokenizer="feature",
                 pe="learned", gate=True, aux=True, dm=D_MODEL, H=N_HEADS,
                 L=N_LAYERS, dff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.tok = (FeatureTokenizer(d, dm) if tokenizer == "feature"
                    else PatchTokenizer(d, dm))
        T = self.tok.n_tokens + 1               # + [CLS]
        self.cls = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        if pe == "learned":
            self.pe = nn.Parameter(torch.randn(1, T, dm) * 0.02)
        else:
            self.register_buffer("pe", sinusoidal(T, dm))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dm, H, dff, dropout, gate)
                                     for _ in range(L)])
        self.head = nn.Linear(dm, n_classes)
        self.aux = nn.Linear(dm, d) if aux else None   # training-only head

    def forward(self, x, return_aux: bool = False):
        z = self.tok(x)
        z = torch.cat([self.cls.expand(z.size(0), -1, -1), z], dim=1)
        h = self.drop(z + self.pe)
        for b in self.blocks:
            h = b(h)
        c = h[:, 0]
        logits = self.head(c)
        if return_aux:
            return logits, (self.aux(c) if self.aux is not None else None)
        return logits

    def gates(self):
        """Per-block [B, H] gate values from the last forward pass."""
        return [b.attn.last_gate for b in self.blocks if b.attn.gate]

    def live_gates(self):
        """Differentiable gate tensors of the last forward pass."""
        return [b.attn.live_gate for b in self.blocks if b.attn.gate]


# ── Non-attention baselines ─────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d, n_classes, width=256, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.ReLU(),
            nn.Linear(width // 2, n_classes))

    def forward(self, x, return_aux=False):
        y = self.net(x)
        return (y, None) if return_aux else y


class LSTMNet(nn.Module):
    """LSTM that reads the feature vector as a length-d sequence of scalars."""

    def __init__(self, d, n_classes, hidden=128, layers=2, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                            dropout=dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x, return_aux=False):
        # cuDNN RNNs cannot backpropagate in eval mode, which white-box
        # attacks need; use the native kernel outside training.
        with torch.backends.cudnn.flags(enabled=self.training):
            _, (hn, _) = self.lstm(x.unsqueeze(-1))
        y = self.head(hn[-1])
        return (y, None) if return_aux else y


AAM = dict(tokenizer="feature", pe="learned", gate=True, aux=True)
VANILLA = dict(tokenizer="patch", pe="sinusoidal", gate=False, aux=False)

BACKBONES = {
    "transformer":  VANILLA,
    "aam_trans":    AAM,
    "aam_noGate":   {**AAM, "gate": False},
    "aam_noTok":    {**AAM, "tokenizer": "patch"},
    "aam_noPE":     {**AAM, "pe": "sinusoidal"},
    "aam_noAux":    {**AAM, "aux": False},
    "vanilla_gate": {**VANILLA, "gate": True},     # gate on the plain backbone
    # Gate v2 candidates (PROTOCOL_GATE_V2.md): per-token value gate.
    "aam_g1":       {**AAM, "gate": "token"},
    "aam_g2":       {**AAM, "gate": "token"},
    "aam_g3":       {**AAM, "gate": "token"},
}

# Extra training-loss terms of the gate v2 candidates: (kind, weight).
GATE_REG = {"aam_g2": ("sparsity", 0.01), "aam_g3": ("consistency", 1.0)}


def build(name: str, d: int, n_classes: int) -> nn.Module:
    if name == "mlp":
        return MLP(d, n_classes)
    if name == "lstm":
        return LSTMNet(d, n_classes)
    return TabTransformer(d, n_classes, **BACKBONES[name])


def n_params(m: nn.Module, inference: bool = True) -> int:
    """Trainable parameters; inference=True excludes the training-only aux head."""
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if inference and getattr(m, "aux", None) is not None:
        n -= sum(p.numel() for p in m.aux.parameters())
    return n
