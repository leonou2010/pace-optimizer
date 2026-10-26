"""Offline checks of pace.PACE on random tensors (CPU, a few seconds)."""

import copy

import torch

from pace import PACE


def _problem(seed=0):
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(64, 32))
    x = torch.randn(256, 32)
    y = torch.randn(256, 64)
    return w, x, y


def _step(w, x, y, opt):
    opt.zero_grad()
    loss = ((x @ w.T - y) ** 2).mean()
    loss.backward()
    opt.step()
    return loss.item()


def test_pace_with_c0_is_adamw():
    w1, x, y = _problem()
    w2 = copy.deepcopy(w1)
    kw = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    ref = torch.optim.AdamW([w1], **kw)
    opt = PACE([w2], lambda_pullback=0.0, use_ema_eval=False, **kw)
    for _ in range(20):
        _step(w1, x, y, ref)
        _step(w2, x, y, opt)
    assert torch.allclose(w1, w2, atol=1e-6)


def test_pullback_moves_toward_ema_and_eval_swaps():
    w, x, y = _problem()
    opt = PACE([w], lr=1e-2, lambda_pullback=0.3, ema_kappa=0.5, use_ema_eval=True,
               log_stats=True)
    for _ in range(20):
        _step(w, x, y, opt)
    stats = opt.get_step_stats()
    assert 0 < stats["lambda_mean"] <= 1.0 and 0 <= stats["lambda_clipped_frac"] <= 1
    theta = w.detach().clone()
    opt.eval()
    ema = w.detach().clone()
    assert not torch.allclose(theta, ema)
    assert torch.allclose(ema, opt.state[w]["ema"])
    opt.train()
    assert torch.equal(w.detach(), theta)
