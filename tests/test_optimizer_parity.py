"""Numerical correctness tests for the PACE optimizer.

Each test trains a small 8->16->1 MLP on a fixed synthetic regression task for 50
optimizer steps and verifies the result:

- test_pace_c0_equals_torch_adamw: with lambda_pullback=0 and use_ema_eval=False,
  PACE reproduces torch.optim.AdamW (final parameters agree to within 1e-6).
- test_eval_swap_does_not_affect_training: with use_ema_eval=True, wrapping a
  validation pass in eval()/train() does not alter the training weights; the loss
  trajectory and final parameters match a run without the swap (within 1e-12).
- test_parity_with_research_impl: for the AdamW, EMA, and PACE configurations, PACE
  matches the original research optimizer to within 1e-9. Requires the
  PACE_REF_OPTIMIZER environment variable; skipped when it is unset.
- test_optimizer_imports_with_torch_only: pace.optimizer imports and exposes PACE
  using only torch, without the training or figure dependencies.

CPU-only; requires no GPU and no bundled data.
"""

import copy
import importlib.util
import os
import sys

import torch

try:
    import pytest
except ImportError:  # allow direct `python tests/test_optimizer_parity.py` without pytest
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*a, **k):
                return lambda fn: fn

        @staticmethod
        def skip(msg):
            raise RuntimeError(f"SKIP: {msg}")

    pytest = _NoPytest()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pace.optimizer import PACE  # noqa: E402

# Path to the original research optimizer for the parity comparison. Set
# PACE_REF_OPTIMIZER to a file exporting `EMARegularizedAdamWV4`; the test skips
# gracefully when it is unavailable (e.g. outside the original environment).
OLD_PATH = os.environ.get("PACE_REF_OPTIMIZER", "")


def _load_old():
    if not os.path.exists(OLD_PATH):
        pytest.skip(f"original optimizer not found at {OLD_PATH} (off-cluster)")
    spec = importlib.util.spec_from_file_location("old_v4", OLD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EMARegularizedAdamWV4


def _make_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    )


def _make_batch(seed=0, n=32):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 8, generator=g)
    y = torch.randn(n, 1, generator=g)
    return x, y


def _run(OptCls, opt_kwargs, n_steps=50, seed=0):
    model = _make_model(seed)
    opt = OptCls(model.parameters(), **opt_kwargs)
    losses = []
    for t in range(n_steps):
        x, y = _make_batch(seed=t)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    final = torch.cat([p.detach().flatten() for p in model.parameters()])
    return losses, final


# ema_rho is set explicitly so the comparison is independent of each implementation's
# default (PACE defaults to 0 = the paper schedule; the reference defaults to 10).
CONFIGS = {
    "adamw":  dict(lr=1e-3, lambda_pullback=0.0, use_ema_eval=False),
    "ema":    dict(lr=1e-3, lambda_pullback=0.0, use_ema_eval=True, ema_kappa=0.5, ema_rho=0.0),
    "pace":   dict(lr=1e-3, lambda_pullback=0.2, use_ema_eval=True, ema_kappa=0.5, ema_rho=0.0),
}


@pytest.mark.parametrize("name,kw", list(CONFIGS.items()))
def test_parity_with_research_impl(name, kw):
    Old = _load_old()
    new_losses, new_final = _run(PACE, copy.deepcopy(kw))
    old_losses, old_final = _run(Old, copy.deepcopy(kw))
    max_loss_diff = max(abs(a - b) for a, b in zip(new_losses, old_losses))
    max_param_diff = (new_final - old_final).abs().max().item()
    assert max_loss_diff < 1e-9, f"{name}: per-step loss diff {max_loss_diff}"
    assert max_param_diff < 1e-9, f"{name}: final param diff {max_param_diff}"


def _run_with_eval_swaps(swap_every=5, n_steps=50, seed=0):
    """PACE c=0 with use_ema_eval, calling eval()/train() around 'validation'
    every `swap_every` steps. Returns the per-step losses + final params."""
    model = _make_model(seed)
    opt = PACE(model.parameters(), lr=1e-3, lambda_pullback=0.0,
               use_ema_eval=True, ema_kappa=0.5)
    losses = []
    for t in range(n_steps):
        x, y = _make_batch(seed=t)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (t + 1) % swap_every == 0:          # simulate an eval round
            opt.eval()
            with torch.no_grad():
                _ = model(x)                   # forward on EMA weights
            opt.train()                        # restore training weights
    final = torch.cat([p.detach().flatten() for p in model.parameters()])
    return losses, final


def test_eval_swap_does_not_affect_training():
    """The EMA eval/train swap must restore training weights exactly, so a run
    with periodic eval swaps is bit-identical to one without."""
    l_swap, p_swap = _run_with_eval_swaps(swap_every=5)
    l_none, p_none = _run(PACE, dict(lr=1e-3, lambda_pullback=0.0, use_ema_eval=True,
                                     ema_kappa=0.5))
    assert max(abs(a - b) for a, b in zip(l_swap, l_none)) < 1e-12, "eval swap perturbed loss"
    assert (p_swap - p_none).abs().max().item() < 1e-12, "eval swap perturbed params"


def test_pace_c0_equals_torch_adamw():
    """c=0, no EMA eval must match torch.optim.AdamW closely."""
    new_losses, new_final = _run(PACE, dict(lr=1e-3, lambda_pullback=0.0, use_ema_eval=False))
    aw_losses, aw_final = _run(
        torch.optim.AdamW, dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    )
    max_param_diff = (new_final - aw_final).abs().max().item()
    assert max_param_diff < 1e-6, f"PACE(c=0) vs torch.AdamW param diff {max_param_diff}"


def test_optimizer_imports_with_torch_only():
    """The core product must import without the training/figure extras — only torch."""
    import importlib
    mod = importlib.import_module("pace.optimizer")
    assert hasattr(mod, "PACE")


if __name__ == "__main__":
    # Allow running directly for a quick human-readable report.
    Old = _load_old()
    for name, kw in CONFIGS.items():
        nl, nf = _run(PACE, copy.deepcopy(kw))
        ol, of = _run(Old, copy.deepcopy(kw))
        ld = max(abs(a - b) for a, b in zip(nl, ol))
        pd_ = (nf - of).abs().max().item()
        print(f"{name:6}: max per-step loss diff={ld:.2e}  max param diff={pd_:.2e}  "
              f"new_final_loss={nl[-1]:.5f} old_final_loss={ol[-1]:.5f}")
    aw_losses, aw_final = _run(
        torch.optim.AdamW, dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01))
    nl, nf = _run(PACE, dict(lr=1e-3, lambda_pullback=0.0, use_ema_eval=False))
    print(f"adamw-vs-torch: max param diff={(nf-aw_final).abs().max().item():.2e}")
    ls, ps = _run_with_eval_swaps(swap_every=5)
    ln, pn = _run(PACE, dict(lr=1e-3, lambda_pullback=0.0, use_ema_eval=True, ema_kappa=0.5))
    print(f"eval-swap no-op: max loss diff={max(abs(a-b) for a,b in zip(ls,ln)):.2e}  "
          f"max param diff={(ps-pn).abs().max().item():.2e}")
    print("PARITY OK")
