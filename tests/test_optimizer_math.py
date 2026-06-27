"""Deterministic math tests for the three optimizer modes (AdamW, EMA, PACE).

Each test drives the optimizer with a fixed sequence of gradients on CPU and compares
its internal state and parameters, step by step, against an independent reference
implementation written in float64 (``_reference_step``). This verifies the update math
directly rather than against another copy of itself:

- test_adamw_math: with lambda_pullback=0 and use_ema_eval=False, the Adam moments
  (m, v) and the parameter update match the reference and torch.optim.AdamW.
- test_ema_math: with use_ema_eval=True, the EMA accumulator matches the reference, the
  training weights still follow AdamW, and eval()/train() swap to the EMA and back.
- test_pace_math: with lambda_pullback>0, the per-coordinate pullback gain lambda and
  the resulting parameter update match the reference, including the <=1 clamp.

CPU-only and fully deterministic; no GPU, network, or bundled data.
"""

import numpy as np
import torch

from pace import PACE

# Fixed problem: 4 parameters, a fixed initial value, and a fixed gradient per step.
P0 = np.array([0.10, -0.20, 0.30, -0.40], dtype=np.float64)
GRADS = [
    np.array([0.50, -0.30, 0.20, -0.70], dtype=np.float64),
    np.array([0.40, 0.10, -0.60, 0.30], dtype=np.float64),
    np.array([-0.20, 0.50, 0.30, -0.10], dtype=np.float64),
]

LR, B1, B2, EPS, WD = 0.01, 0.9, 0.999, 1e-8, 0.01
KAPPA, GAMMA, RHO = 0.5, 1.0, 0.0


def _reference(c, use_ema, clamp=True, uf=1, kappa=KAPPA):
    """Independent float64 reference. Returns per-step state dicts."""
    p = P0.copy()
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    ema = p.copy() if (use_ema or c > 0) else None
    last_ema = 0
    history = []
    for t in range(1, len(GRADS) + 1):
        g = GRADS[t - 1]
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        m_hat = m / (1 - B1 ** t)
        v_hat_sqrt = np.sqrt(v / (1 - B2 ** t)) + EPS
        p_before = p.copy()
        p = p - LR * WD * p                  # decoupled weight decay
        p = p - LR * m_hat / v_hat_sqrt      # Adam step
        lam = None
        if c > 0 and ema is not None:
            decay = max((1 + GAMMA * max(0, t - RHO)) ** (-kappa), 1e-4)
            lam = LR * c * decay / v_hat_sqrt
            if clamp:
                lam = np.minimum(lam, 1.0)
            p = p + lam * (ema - p_before)
        if ema is not None and (t - last_ema) % uf == 0:
            beta = max((1 + GAMMA * max(0, t - RHO)) ** (-kappa), 1e-4)
            ema = (1 - beta) * ema + beta * p
            last_ema = t
        history.append({"p": p.copy(), "m": m.copy(), "v": v.copy(),
                        "ema": None if ema is None else ema.copy(), "lam": lam})
    return history


def _make_opt(**kw):
    p = torch.tensor(P0, dtype=torch.float32, requires_grad=True)
    opt = PACE([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD,
               ema_gamma=GAMMA, ema_rho=RHO, **kw)
    return p, opt


def _step(p, opt, t):
    p.grad = torch.tensor(GRADS[t], dtype=torch.float32)
    opt.step()


ATOL = 1e-5


def test_adamw_math():
    """c=0, no EMA: Adam moments and the update match the reference and torch.AdamW."""
    ref = _reference(c=0.0, use_ema=False)
    p, opt = _make_opt(lambda_pullback=0.0, use_ema_eval=False)
    for t in range(len(GRADS)):
        _step(p, opt, t)
        st = opt.state[p]
        assert np.allclose(st["m"].numpy(), ref[t]["m"], atol=ATOL), f"m mismatch at step {t+1}"
        assert np.allclose(st["v"].numpy(), ref[t]["v"], atol=ATOL), f"v mismatch at step {t+1}"
        assert np.allclose(p.detach().numpy(), ref[t]["p"], atol=ATOL), f"p mismatch at step {t+1}"

    # Cross-check against the library AdamW.
    pt = torch.tensor(P0, dtype=torch.float32, requires_grad=True)
    aw = torch.optim.AdamW([pt], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=WD)
    for t in range(len(GRADS)):
        pt.grad = torch.tensor(GRADS[t], dtype=torch.float32)
        aw.step()
    assert np.allclose(p.detach().numpy(), pt.detach().numpy(), atol=1e-6)


def test_ema_math():
    """c=0, EMA eval: the EMA accumulator matches; training follows AdamW; eval swaps."""
    ref = _reference(c=0.0, use_ema=True)
    p, opt = _make_opt(lambda_pullback=0.0, use_ema_eval=True, ema_kappa=KAPPA)
    for t in range(len(GRADS)):
        _step(p, opt, t)
        st = opt.state[p]
        assert np.allclose(p.detach().numpy(), ref[t]["p"], atol=ATOL), f"train p mismatch at {t+1}"
        assert np.allclose(st["ema"].numpy(), ref[t]["ema"], atol=ATOL), f"ema mismatch at {t+1}"

    train_w = p.detach().clone()
    opt.eval()
    assert np.allclose(p.detach().numpy(), ref[-1]["ema"], atol=ATOL), "eval() did not expose the EMA"
    opt.train()
    assert torch.equal(p.detach(), train_w), "train() did not restore the training weights"


def test_pace_math():
    """c>0: the pullback gain lambda and the update match the reference, clamp included."""
    C = 2.0
    ref = _reference(c=C, use_ema=True)
    p, opt = _make_opt(lambda_pullback=C, use_ema_eval=True, ema_kappa=KAPPA,
                       clamp_pullback=True)
    for t in range(len(GRADS)):
        _step(p, opt, t)
        st = opt.state[p]
        assert np.allclose(p.detach().numpy(), ref[t]["p"], atol=ATOL), f"pace p mismatch at {t+1}"
        assert np.allclose(st["ema"].numpy(), ref[t]["ema"], atol=ATOL), f"ema mismatch at {t+1}"

    # The pullback must be active (not degenerate) and differ from plain AdamW.
    _, ref_lam = ref[0]["p"], ref[0]["lam"]
    assert ref_lam is not None and np.all(ref_lam > 0) and np.all(ref_lam <= 1.0)
    adamw = _reference(c=0.0, use_ema=False)
    assert not np.allclose(ref[-1]["p"], adamw[-1]["p"], atol=ATOL), "pullback had no effect"


def test_pace_clamp_engages():
    """A very large c drives the gain to its clamp value of 1 on every coordinate."""
    ref = _reference(c=1e6, use_ema=True, clamp=True)
    assert np.allclose(ref[0]["lam"], 1.0), "clamp did not cap the gain at 1"
    p, opt = _make_opt(lambda_pullback=1e6, use_ema_eval=True, ema_kappa=KAPPA,
                       clamp_pullback=True)
    for t in range(len(GRADS)):
        _step(p, opt, t)
        assert np.allclose(p.detach().numpy(), ref[t]["p"], atol=ATOL), f"clamped p mismatch at {t+1}"
