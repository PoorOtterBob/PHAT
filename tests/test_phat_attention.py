"""Checks of PHAT_Attention against a brute-force evaluation of the paper's equations.

Run:  python -m pytest tests/test_phat_attention.py
(only needs torch; the module is loaded by file path, no TFB installation required)
"""
import importlib.util
import pathlib

import pytest
import torch

_PATH = pathlib.Path(__file__).resolve().parents[1] / "phat" / "layers" / "PHAT_Attention.py"
_spec = importlib.util.spec_from_file_location("phat_attention", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PHAT_Attention = _mod.PHAT_Attention


def _dist(P, absolute):
    d = torch.zeros(P, P, dtype=torch.long)
    for i in range(P):
        for j in range(P):
            d[i, j] = abs(i - j) if absolute else min((i - j) % P, (j - i) % P)
    return d


def _brute_modulate(logits, dist, closer):
    """logits (L, L): Eq. 7 / App. C.2 with explicit set loops."""
    L = logits.shape[0]
    sp = torch.nn.functional.softplus(logits)
    out = torch.empty_like(logits)
    for m in range(L):
        for n in range(L):
            s = [t for t in range(L) if (dist[m, t] < dist[m, n] if closer else dist[m, t] > dist[m, n])]
            out[m, n] = logits[m, n] - sp[m, n] - sum(sp[m, t] for t in s)
    return out


@pytest.mark.parametrize("P", [1, 2, 7, 8])
@pytest.mark.parametrize("absolute", [False, True])
def test_modulation_matches_bruteforce(P, absolute):
    torch.manual_seed(0)
    att = PHAT_Attention(16, 2)
    logits = torch.randn(1, 1, P, P, 1)
    dist, idx, valid = att._layout(P, absolute, logits.device)
    assert torch.equal(dist, _dist(P, absolute))
    for closer in (True, False):
        got = att._modulate(logits, dist, idx, valid, closer)[0, 0, :, :, 0]
        want = _brute_modulate(logits[0, 0, :, :, 0], dist, closer)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_stick_breaking_identity():
    """App. C.2, Eq. 22: exp(zeta_tilde[m, n]) = sigmoid(z[m,n]) * prod_{closer s}(1 - sigmoid(z[m,s]))."""
    torch.manual_seed(1)
    P = 9
    att = PHAT_Attention(16, 2)
    z = torch.randn(1, 1, P, P, 1)
    dist, idx, valid = att._layout(P, False, z.device)
    mod = att._modulate(z, dist, idx, valid, True)[0, 0, :, :, 0].exp()
    sig = torch.sigmoid(z[0, 0, :, :, 0])
    for m in range(P):
        for n in range(P):
            want = sig[m, n]
            for s in range(P):
                if dist[m, s] < dist[m, n]:
                    want = want * (1 - sig[m, s])
            torch.testing.assert_close(mod[m, n], want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", [(2, 6, 5, 32), (2, 24, 3, 32), (1, 11, 1, 32)])
def test_shape_backward_and_row_sums(shape):
    torch.manual_seed(2)
    att = PHAT_Attention(32, 4, attn_dropout=0.0).eval()
    x = torch.randn(*shape, requires_grad=True)
    y = att(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    unused = {"mu_aligned"} if shape[2] == 1 else set()  # identity aligned attention (App. B)
    assert all(p.grad is not None for n, p in att.named_parameters() if n not in unused)


def test_offset_attention_rows_sum_to_one_minus_lambda():
    """Eq. 9: sum_j A[i, j, n] = 1 - Lambda[i, n]."""
    torch.manual_seed(3)
    att = PHAT_Attention(16, 2, attn_dropout=0.0).eval()
    x = torch.randn(2, 6, 4, 16)
    captured = {}
    orig = att.dropout.forward
    att.dropout.forward = lambda t, _o=orig: captured.setdefault("A", t) if "A" not in captured else _o(t)
    att(x)
    lam = torch.sigmoid(att.to_gate(x)).permute(0, 3, 1, 2)
    torch.testing.assert_close(captured["A"].sum(dim=3), 1 - lam, atol=1e-5, rtol=1e-5)


def test_period_aligned_axis_is_identity_without_periodicity():
    """App. B: with Nb == 1 the aligned attention is the identity, so changing the mu has no effect."""
    torch.manual_seed(4)
    att = PHAT_Attention(16, 2, attn_dropout=0.0).eval()
    x = torch.randn(2, 10, 1, 16)
    y0 = att(x)
    att.mu_aligned.data.mul_(5.0)
    torch.testing.assert_close(att(x), y0)
