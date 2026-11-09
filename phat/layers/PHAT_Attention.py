"""Positive-Negative X-shape Attention (PNA) of PHAT.

Re-implementation of Sec. 3.2 (Eq. 4-12), App. B (bucket without periodicity) and
App. C.2 (stick-breaking view) of

    PHAT: Modeling Period Heterogeneity for Multivariate Time Series Forecasting,
    ICLR 2026, arXiv:2602.00654.

Interface (as used by ``PHATModel`` / ``Transformer_Block``)::

    PHAT_Attention(d_model, head, attn_dropout, layer_index)
    forward: Z (B, Pb, Nb, d_model) -> (B, Pb, Nb, d_model)

``Pb`` is the period-offset axis and ``Nb`` the period-aligned axis of the bucket.
``Nb == 1`` is the bucket without periodicity (App. B): the offset axis then uses the
absolute distance |i - j| instead of the periodic distance of Eq. 8, and the
period-aligned attention degenerates to the identity.

Implementation notes (choices where the paper text leaves room):

* ``W_Q, W_K, W_V`` are ``d_model -> d_model``. Per head (size ``hd = d_model / head``),
  ``Q = [Q1; Q2]`` and ``K = [K1; K2]`` are split in two halves of ``hd / 2``
  (positive / negative branch), ``V`` has ``hd`` channels. ``mu = (hd / 2) ** -0.5``.
* ``Lambda = sigmoid(Z W_g)`` is one scalar per head, position and bucket column.
* The modulation sets use App. C.2: ``Delta = {s: d(m,s) < d(m,n)} + {n}`` and
  ``Nabla = {s: d(m,s) > d(m,n)} + {n}`` (Sec. 3.2 writes ``{m}``; App. C.2 writes ``{n}``,
  which is the one for which the stick-breaking identity of Eq. 22-23 holds).
* ``PNA = A x_1 (A_tilde x_2 V)``: values are first mixed along the period-aligned axis by
  ``A_tilde`` (Eq. 10), then along the period-offset axis by ``A`` (Eq. 7).
* ``head = gamma * tanh(alpha * (PNA + Lambda * Z_h)) + beta`` (Eq. 12, Dynamic Tanh),
  followed by the output projection ``W_O`` (Eq. 11).
* Attention dropout is applied to the attention coefficient matrices.

The full per-head tensors have size ``B x head x Pb x Pb x Nb`` (= ``Pb`` times the padded
series length per sample and head).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PHAT_Attention(nn.Module):
    def __init__(self, d_model, head, attn_dropout=0.1, layer_index=0):
        super().__init__()
        assert d_model % head == 0, "d_model must be divisible by head"
        self.head_num = head
        self.head_dim = d_model // head
        assert self.head_dim % 2 == 0, "d_model / head must be even (positive / negative halves)"
        self.qk_dim = self.head_dim // 2
        self.layer_index = layer_index

        self.to_qkv = nn.Linear(d_model, 3 * d_model)  # Eq. 4: W_q, W_k, W_v
        self.to_gate = nn.Linear(d_model, head)  # Eq. 4: Lambda = sigmoid(Z W_g)
        self.out = nn.Linear(d_model, d_model)  # Eq. 11: W_O

        # learnable coefficient of the period-aligned attention (Eq. 10)
        self.mu_aligned = nn.Parameter(torch.full((head, 1, 1, 1), self.qk_dim ** -0.5))
        # Dynamic Tanh of Eq. 12 (alpha: per head; gamma, beta: per channel)
        self.alpha = nn.Parameter(torch.full((head, 1, 1, 1), 0.5))
        self.gamma = nn.Parameter(torch.ones(head, 1, 1, self.head_dim))
        self.beta = nn.Parameter(torch.zeros(head, 1, 1, self.head_dim))

        self.dropout = nn.Dropout(attn_dropout)
        self.mu = self.qk_dim ** -0.5
        self._layout_cache = {}

    # ------------------------------------------------------------------ distances
    def _layout(self, length, absolute, device):
        """Distance matrix and the two index maps used to sum over distance layers.

        Returns ``dist`` (L, L) with d(m, n), ``idx`` (2, L, Dn): the columns at distance
        ``d`` to the right / left of ``m``, and ``valid`` (2, L, Dn): whether that column is a
        distinct, existing one (``Dn`` = number of distinct distances).
        """
        key = (length, absolute, str(device))
        if key in self._layout_cache:
            return self._layout_cache[key]
        pos = torch.arange(length, device=device)
        if absolute:  # App. B, Eq. 16: |j - i|
            dist = (pos[None, :] - pos[:, None]).abs()
            d_axis = torch.arange(length, device=device)
            right, left = pos[:, None] + d_axis[None, :], pos[:, None] - d_axis[None, :]
            valid = torch.stack([right < length, (left >= 0) & (d_axis[None, :] > 0)])
            idx = torch.stack([right.clamp(max=length - 1), left.clamp(min=0)])
        else:  # Eq. 8: min((i - j) mod L, (j - i) mod L)
            diff = pos[None, :] - pos[:, None]
            dist = torch.minimum(diff % length, (-diff) % length)
            d_axis = torch.arange(length // 2 + 1, device=device)
            right, left = (pos[:, None] + d_axis[None, :]) % length, (pos[:, None] - d_axis[None, :]) % length
            valid = torch.stack([torch.ones_like(right, dtype=torch.bool), left != right])
            idx = torch.stack([right, left])
        out = (dist, idx, valid)
        self._layout_cache[key] = out
        return out

    @staticmethod
    def _modulate(logits, dist, idx, valid, closer):
        """logits - sum of softplus(logits) over Delta (closer) or Nabla (farther), Eq. 7 / App. C.2.

        logits: (B, H, L, L, Nb) indexed [m, s, n-column].
        """
        sp = F.softplus(logits)
        B, H, L, _, Nb = sp.shape
        gather_idx = idx[:, None, None, :, :, None].expand(2, B, H, L, idx.shape[-1], Nb)
        layer = sum(
            sp.gather(3, gather_idx[i]) * valid[i][None, None, :, :, None]
            for i in range(2)
        )  # (B, H, L, Dn, Nb): mass of every distance layer around each m
        csum = layer.cumsum(dim=3)
        if closer:
            per_layer = csum - layer  # strictly closer layers
        else:
            per_layer = csum[:, :, :, -1:] - csum  # strictly farther layers
        sets = per_layer.gather(3, dist[None, None, :, :, None].expand(B, H, L, L, Nb))
        return logits - (sets + sp)  # '+ sp' is the {n} element of both sets

    # -------------------------------------------------------------------- forward
    def forward(self, x):
        B, P, N, _ = x.shape  # period-offset axis P, period-aligned axis N
        H, hd, qd = self.head_num, self.head_dim, self.qk_dim

        qkv = self.to_qkv(x).view(B, P, N, 3, H, hd)
        q, k, v = (t.permute(0, 3, 1, 2, 4) for t in qkv.unbind(dim=3))  # (B, H, P, N, hd)
        q1, q2 = q[..., :qd], q[..., qd:]
        k1, k2 = k[..., :qd], k[..., qd:]
        lam = torch.sigmoid(self.to_gate(x)).permute(0, 3, 1, 2)  # (B, H, P, N)

        # period-offset attention A in R^{P x P x N}, Eq. 6-7
        zeta = torch.einsum('bhmnd,bhsnd->bhmsn', q1, k1) * self.mu
        eta = torch.einsum('bhmnd,bhsnd->bhmsn', q2, k2) * self.mu
        dist, idx, valid = self._layout(P, absolute=(N == 1), device=x.device)
        zeta = self._modulate(zeta, dist, idx, valid, closer=True)
        eta = self._modulate(eta, dist, idx, valid, closer=False)
        attn = torch.softmax(zeta, dim=3) - lam[:, :, :, None, :] * torch.softmax(eta, dim=3)
        attn = self.dropout(attn)

        # period-aligned attention A_tilde in R^{P x N x N}, Eq. 10 (shares Q1, K1, V)
        if N == 1:  # App. B: softmax over a single entry is the identity
            aligned_v = v
        else:
            aligned = torch.einsum('bhmnd,bhmkd->bhmnk', q1, k1) * self.mu_aligned
            aligned = self.dropout(torch.softmax(aligned, dim=-1))
            aligned_v = torch.einsum('bhmnk,bhmkd->bhmnd', aligned, v)

        pna = torch.einsum('bhmsn,bhsnd->bhmnd', attn, aligned_v)  # Eq. 5

        z_h = x.view(B, P, N, H, hd).permute(0, 3, 1, 2, 4)  # per-head slice of Z
        head = self.gamma * torch.tanh(self.alpha * (pna + lam.unsqueeze(-1) * z_h)) + self.beta  # Eq. 12
        return self.out(head.permute(0, 2, 3, 1, 4).reshape(B, P, N, H * hd))  # Eq. 11
