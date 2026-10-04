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


# ── alpha-entmax (Peters et al., 2019; Correia et al., 2019) ────────────────
class EntmaxBisect(torch.autograd.Function):
    """alpha-entmax over the last dimension by bisection, with the closed-form
    backward of Peters et al. (2019) for the input and for alpha."""

    @staticmethod
    def forward(ctx, X, alpha, n_iter=25):
        am1 = alpha - 1                                    # [..., 1]
        Xs = X * am1
        mx = Xs.max(dim=-1, keepdim=True).values
        lo = mx - 1.0
        hi = mx - (1.0 / X.size(-1)) ** am1
        f_lo = (torch.clamp(Xs - lo, min=0) ** (1 / am1)).sum(-1, keepdim=True) - 1
        dm = hi - lo
        for _ in range(n_iter):
            dm = dm / 2
            mid = lo + dm
            p = torch.clamp(Xs - mid, min=0) ** (1 / am1)
            f = p.sum(-1, keepdim=True) - 1
            lo = torch.where((f * f_lo) >= 0, mid, lo)
        p = p / p.sum(-1, keepdim=True)
        ctx.save_for_backward(p, alpha)
        return p

    @staticmethod
    def backward(ctx, dY):
        Y, alpha = ctx.saved_tensors
        pos = Y > 0
        gppr = torch.where(pos, Y.clamp_min(1e-30) ** (2 - alpha), torch.zeros_like(Y))
        dX = dY * gppr
        q = dX.sum(-1, keepdim=True) / gppr.sum(-1, keepdim=True)
        dX = dX - q * gppr
        S = torch.where(pos, Y * torch.log(Y.clamp_min(1e-30)), torch.zeros_like(Y))
        ent = S.sum(-1, keepdim=True)
        Ysk = gppr / gppr.sum(-1, keepdim=True)
        da = dY * (Y - Ysk) / (alpha - 1) ** 2 - dY * (S - Ysk * ent) / (alpha - 1)
        return dX, da.sum(-1, keepdim=True), None


def entmax(X, alpha):
    """alpha-entmax of X over the last dimension; alpha has shape X[..., :1]."""
    return EntmaxBisect.apply(X, alpha)


# ── Attention with optional adaptive per-head gate ──────────────────────────
class GatedMHA(nn.Module):
    """Multi-head self-attention with an optional gate.

    gate='record' (v1): every head output A_h is scaled by
        gamma_h(h) = sigmoid(MLP(mean_t h_t))_h, one weight per record and head.
    gate='token'  (v2): every value vector V_{h,t} is scaled by
        gamma_{h,t} = sigmoid(w_h . h_t + b_h), one weight per head and token
        (i.e. per input feature), so a head can stop reading from a feature.
    """

    def __init__(self, dm: int, H: int, dropout: float, gate, attn="softmax"):
        super().__init__()
        self.H, self.dk = H, dm // H
        self.attn = attn
        self.qkv = nn.Linear(dm, 3 * dm)
        if attn == "entmax":
            # v4a: alpha_h = 1.01 + 0.98 sigmoid(a_h), 1.5 at initialisation
            self.alpha_raw = nn.Parameter(torch.zeros(1, H, 1, 1))
        self.out = nn.Linear(dm, dm)
        self.drop = nn.Dropout(dropout)
        self.gate = "record" if gate is True else (gate or None)
        if self.gate == "record":
            self.gate_mlp = nn.Sequential(nn.Linear(dm, dm // 4), nn.ReLU(),
                                          nn.Linear(dm // 4, H))
        elif self.gate in ("token", "mask"):
            self.gate_proj = nn.Linear(dm, H)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, 2.0)   # start open (~0.88)
        elif self.gate == "temp":
            self.temp_mlp = nn.Sequential(nn.Linear(dm, dm // 4), nn.ReLU(),
                                          nn.Linear(dm // 4, H))
            nn.init.zeros_(self.temp_mlp[2].weight)
            # 0.5 + softplus(b) = 1 at initialisation
            nn.init.constant_(self.temp_mlp[2].bias, math.log(math.exp(0.5) - 1))
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
        if self.attn == "l2":
            # v4c: L2-distance attention with tied query/key (Kim et al., 2021)
            q2 = (q * q).sum(-1, keepdim=True)
            scores = -(q2 - 2 * q @ q.transpose(-2, -1) + q2.transpose(-2, -1)) \
                / math.sqrt(self.dk)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)   # [B, H, T, T]
        if self.gate == "mask":
            # v3b: adaptive key masking, added to the logits before softmax
            lg = F.logsigmoid(self.gate_proj(h)).transpose(1, 2)      # [B, H, T]
            self.live_gate = lg.exp()
            self.last_gate = lg.exp().mean(-1).detach()
            scores = scores + lg.unsqueeze(2)
        elif self.gate == "temp":
            # v3c: adaptive attention sharpness per head and record
            tau = 0.5 + F.softplus(self.temp_mlp(h.mean(dim=1)))    # [B, H]
            self.live_gate = tau
            self.last_gate = tau.detach()
            scores = scores / tau[:, :, None, None]
        if self.attn == "entmax":
            alpha = (1.01 + 0.98 * torch.sigmoid(self.alpha_raw)).expand(B, -1, T, 1)
            a = entmax(scores.float(), alpha.float()).to(scores.dtype)
        else:
            a = F.softmax(scores, dim=-1)
        o = self.drop(a) @ v                    # [B, H, T, dk]
        if self.gate == "record":
            g = torch.sigmoid(self.gate_mlp(h.mean(dim=1)))   # [B, H]
            self.live_gate = g
            self.last_gate = g.detach()
            o = o * g[:, :, None, None]
        return self.out(o.transpose(1, 2).reshape(B, T, dm))


class Block(nn.Module):
    """Post-norm encoder block (Eq. block in the paper)."""

    def __init__(self, dm, H, dff, dropout, gate, attn="softmax", act="relu"):
        super().__init__()
        self.attn = GatedMHA(dm, H, dropout, gate, attn)
        self.ffn = nn.Sequential(nn.Linear(dm, dff),
                                 nn.GELU() if act == "gelu" else nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(dff, dm))
        self.n1, self.n2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        self.drop = nn.Dropout(dropout)

    def forward(self, h):
        h = self.n1(h + self.drop(self.attn(h)))
        return self.n2(h + self.drop(self.ffn(h)))


class TabTransformer(nn.Module):
    def __init__(self, d: int, n_classes: int, tokenizer="feature",
                 pe="learned", gate=True, aux=True, dm=D_MODEL, H=N_HEADS,
                 L=N_LAYERS, dff=D_FF, dropout=DROPOUT, token_gate=False,
                 attn="softmax", act="relu", purify=False):
        super().__init__()
        self.tok = (FeatureTokenizer(d, dm) if tokenizer == "feature"
                    else PatchTokenizer(d, dm))
        # v3a: perturbation-aware token gate driven by a leave-one-out
        # linear predictor of every feature from the others.
        self.token_gate = token_gate
        self.purify = purify                    # v4b: replace, not scale
        self.use_loo = token_gate or purify
        if self.use_loo:
            self.loo = nn.Linear(d, d)
            nn.init.zeros_(self.loo.weight)
            self.register_buffer("offdiag", 1.0 - torch.eye(d))
            self.tg_a = nn.Parameter(torch.full((d,), 2.0))
            self.tg_b = nn.Parameter(torch.zeros(d))
            self.last_token_gate = None
        T = self.tok.n_tokens + 1               # + [CLS]
        self.cls = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        if pe == "learned":
            self.pe = nn.Parameter(torch.randn(1, T, dm) * 0.02)
        elif pe == "none":                      # FT-Transformer: bias only
            self.register_buffer("pe", torch.zeros(1, T, dm))
        else:
            self.register_buffer("pe", sinusoidal(T, dm))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(dm, H, dff, dropout, gate, attn, act)
                                     for _ in range(L)])
        self.head = nn.Linear(dm, n_classes)
        self.aux = nn.Linear(dm, d) if aux else None   # training-only head

    def loo_predict(self, x):
        """Leave-one-out linear estimate of every feature from the others."""
        return F.linear(x, self.loo.weight * self.offdiag, self.loo.bias)

    def purity(self, x):
        """v4b gate gamma(x) in (0, 1)^d: 1 keeps x_j, 0 replaces it by x_hat_j."""
        xh = self.loo_predict(x)
        return torch.sigmoid(self.tg_a - F.softplus(self.tg_b) * (x - xh).abs()), xh

    def forward(self, x, return_aux: bool = False):
        if self.purify:
            g, xh = self.purity(x)
            self.last_token_gate = g.detach()
            x = g * x + (1 - g) * xh
        z = self.tok(x)
        if self.token_gate:
            r = (x - self.loo_predict(x)).abs()                       # [B, d]
            g = torch.sigmoid(self.tg_a - F.softplus(self.tg_b) * r)  # [B, d]
            self.last_token_gate = g.detach()
            z = z * g.unsqueeze(-1)
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
        """Per-block [B, H] gate values from the last forward pass; for the
        v3a token gate, a single [B, d] entry with the per-feature gate."""
        if self.use_loo:
            return [self.last_token_gate]
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
    # Adaptive mechanism v3 candidates (PROTOCOL_GATE_V3.md).
    "aam_v3a":      {**AAM, "gate": False, "token_gate": True},
    "aam_v3b":      {**AAM, "gate": "mask"},
    "aam_v3c":      {**AAM, "gate": "temp"},
    # v4 (PROTOCOL_V4.md): FT-Transformer-style control and four candidates.
    "ft":           {"tokenizer": "feature", "pe": "none", "gate": False, "aux": False},
    "aam_v4a":      {**AAM, "gate": False, "attn": "entmax"},
    "aam_v4b":      {**AAM, "gate": False, "purify": True},
    "aam_v4c":      {**AAM, "gate": False, "attn": "l2"},
    "aam_v4d":      {**AAM, "gate": False, "act": "gelu"},
}

# Extra training-loss terms of the gate v2 candidates: (kind, weight).
GATE_REG = {"aam_g2": ("sparsity", 0.01), "aam_g3": ("consistency", 1.0)}


def build(name: str, d: int, n_classes: int) -> nn.Module:
    if name == "mlp":
        return MLP(d, n_classes)
    if name == "lstm":
        return LSTMNet(d, n_classes)
    if "+" in name:                             # v4 combination, e.g. aam_v4a+d
        base, extra = name.split("+")
        cfg = {**BACKBONES[base], **{k: v for k, v in BACKBONES["aam_v4" + extra].items()
                                     if k in ("attn", "act", "purify")}}
        return TabTransformer(d, n_classes, **cfg)
    return TabTransformer(d, n_classes, **BACKBONES[name])


def n_params(m: nn.Module, inference: bool = True) -> int:
    """Trainable parameters; inference=True excludes the training-only aux head."""
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if inference and getattr(m, "aux", None) is not None:
        n -= sum(p.numel() for p in m.aux.parameters())
    return n
