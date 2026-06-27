"""PACE — Pullback-Augmented, Constant-LR-friendly Estimation.

PACE is AdamW with an additive, per-coordinate pullback toward an exponential
moving average (EMA) of the iterates. The pullback gain is scaled by the learning
rate and inversely by the Adam second-moment estimate, so its strength tracks the
size of the Adam step automatically (no pullback when ``lr -> 0``, e.g. during
warmup).

Update (per coordinate ``i`` at step ``t``)::

    # Phase A — standard AdamW
    m       = b1*m + (1-b1)*g
    v       = b2*v + (1-b2)*g^2
    theta'  = theta - lr*wd*theta - lr * m_hat / (sqrt(v_hat) + eps)

    # Phase B — pullback toward the EMA (reference is the PRE-Adam theta)
    decay     = (1 + gamma * max(0, t - rho))^(-kappa)         # floored at 1e-4
    lambda_i  = clamp( lr * c * decay / (sqrt(v_hat_i) + eps), max=1 )
    theta_t1  = theta' + lambda_i * (ema - theta)

    # Phase C — EMA update (every ``ema_update_freq`` steps)
    beta_t    = (1 + gamma * max(0, t - rho))^(-kappa)         # floored at 1e-4
    ema       = (1 - beta_t)*ema + beta_t*theta_t1

Special cases:
  * ``lambda_pullback == 0`` and ``use_ema_eval == False`` -> the EMA buffer is
    never allocated and the update reduces to AdamW (matching
    ``torch.optim.AdamW`` up to floating-point ordering).
  * ``lambda_pullback == 0`` and ``use_ema_eval == True``  -> training follows
    plain AdamW; evaluation swaps in the EMA weights (the "EMA" baseline).
  * ``lambda_pullback > 0``                                 -> full PACE.

All optimizer state is kept in fp32 (moments, EMA, and the saved training
weights) to avoid bf16 accumulation error, matching ``torch.optim.AdamW``.
"""

import math
from typing import Tuple, Union, Optional, Iterable, Dict, Callable, Any
from typing_extensions import TypeAlias
import torch
import torch.optim

try:
    from torch.optim.optimizer import ParamsT
except ImportError:
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class PACE(torch.optim.Optimizer):
    """AdamW with an additive, lr-scaled, per-coordinate pullback toward an EMA.

    Args:
        params: iterable of parameters or parameter groups.
        lr: learning rate.
        betas: Adam coefficients ``(beta1, beta2)``.
        eps: term added to the denominator for numerical stability.
        weight_decay: decoupled (AdamW) weight decay.
        lambda_pullback: pullback strength ``c``. ``0`` disables the pullback
            entirely (the optimizer then reduces to AdamW, or to the EMA-eval
            baseline if ``use_ema_eval`` is set).
        clamp_pullback: clamp each per-coordinate gain to ``<= 1`` (interpolate
            toward the EMA rather than extrapolate past it).
        beta_ema: fixed EMA decay, used only when ``ema_kappa is None``.
        use_ema_eval: if True, ``eval()`` swaps the parameters to the EMA weights
            and ``train()`` swaps them back.
        ema_kappa: exponent ``kappa`` of the decaying EMA schedule. ``None`` uses
            the fixed ``beta_ema`` instead.
        ema_rho: offset ``rho`` before the schedule begins to decay. Default ``0``
            gives the paper's ``(1 + gamma*t)^(-kappa)`` schedule.
        ema_gamma: schedule rate ``gamma``.
        ema_update_freq: update the EMA every this many steps.
        log_stats: if True, compute per-step telemetry (lambda/clip-fraction and
            update norms) exposed via ``get_step_stats``. Off by default because each
            stat calls ``.item()`` (a GPU sync) every step; the optimizer math is
            identical either way. Enable it for analysis/figure reproduction.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        lambda_pullback: float = 0.0,
        clamp_pullback: bool = True,
        beta_ema: float = 0.01,
        use_ema_eval: bool = False,
        ema_kappa: Optional[float] = None,
        ema_rho: float = 0.0,
        ema_gamma: float = 1.0,
        ema_update_freq: int = 1,
        log_stats: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if lambda_pullback < 0.0:
            raise ValueError(f"Invalid lambda_pullback: {lambda_pullback}")
        if ema_kappa is None and not 0.0 < beta_ema <= 1.0:
            raise ValueError(f"Invalid beta_ema: {beta_ema}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            lambda_pullback=lambda_pullback,
            clamp_pullback=clamp_pullback,
            beta_ema=beta_ema,
            use_ema_eval=use_ema_eval,
            ema_kappa=ema_kappa,
            ema_rho=ema_rho,
            ema_gamma=ema_gamma,
            ema_update_freq=ema_update_freq,
            log_stats=log_stats,
            train_mode=True,
        )
        super().__init__(params, defaults)
        # Per-step stats for logging
        self._last_raw_lambda_mean = None   # decay / (sqrt(v_hat) + eps), before c and lr
        self._last_raw_lambda_max = None
        self._last_raw_lambda_min = None
        self._last_lambda_mean = None       # lr * c * raw_lambda, after clamp
        self._last_lambda_max = None
        self._last_lambda_min = None
        self._last_lambda_clipped_frac = None
        self._last_pullback_norm = None
        self._last_adam_step_norm = None
        self._last_update_norm = None
        # effective pullback strength = lr * c * raw_lambda_mean (lr-comparable)
        self._last_effective_lambda_mean = None
        self._last_effective_lambda_max = None
        self._last_effective_lambda_min = None

    @torch.no_grad()
    def eval(self):
        """Swap parameters to EMA weights for evaluation."""
        for group in self.param_groups:
            if not group['use_ema_eval']:
                continue
            if group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'ema' in state:
                        state['theta'].copy_(p.data)
                        p.data.copy_(state['ema'])
                group['train_mode'] = False

    @torch.no_grad()
    def train(self):
        """Swap parameters back to training weights."""
        for group in self.param_groups:
            if not group['use_ema_eval']:
                continue
            if not group['train_mode']:
                for p in group['params']:
                    state = self.state[p]
                    if 'theta' in state:
                        p.data.copy_(state['theta'])
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            lambda_pullback = group['lambda_pullback']
            clamp_pullback = group['clamp_pullback']
            beta_ema = group['beta_ema']
            use_ema_eval = group['use_ema_eval']
            ema_kappa = group['ema_kappa']
            ema_rho = group['ema_rho']
            ema_gamma = group['ema_gamma']
            log_stats = group['log_stats']

            # Per-step logging accumulators — only used when log_stats is on. They
            # call .item() each step (a GPU sync), so gating them keeps the core
            # optimizer step sync-free in production; the algorithm runs regardless.
            if log_stats:
                raw_lam_sum = 0.0   # decay / (sqrt(v_hat) + eps), before c and lr
                raw_lam_max = -float('inf')
                raw_lam_min = float('inf')
                lam_sum = 0.0       # after lr * c * raw and clamp
                lam_max = -float('inf')
                lam_min = float('inf')
                clipped_count = 0
                total_count = 0
                adam_step_norms_sq = 0.0
                pullback_norms_sq = 0.0
                update_norms_sq = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state on first step. All states in fp32 to avoid
                # bf16 precision loss in moment accumulation and EMA tracking
                # (PyTorch AdamW does the same).
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data, dtype=torch.float32)
                    state['v'] = torch.zeros_like(p.data, dtype=torch.float32)
                    if use_ema_eval or lambda_pullback > 0:
                        state['ema'] = p.data.clone().float()
                        state['theta'] = p.data.clone().float()
                        state['last_ema_step'] = 0

                state['step'] += 1
                t = state['step']

                m = state['m']
                v = state['v']

                # Phase A: standard Adam moment updates (in fp32)
                grad_fp32 = grad.float()
                m.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                m_hat = m / bias_correction1
                v_hat_sqrt = (v / bias_correction2).sqrt_().add_(eps)

                # Save p before update; the pullback references the pre-Adam theta.
                p_before = p.data.float()

                # Weight decay (decoupled, AdamW-style)
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                # Adam step (m_hat, v_hat_sqrt are fp32; addcdiv_ handles mixed dtypes)
                p.data.addcdiv_(m_hat, v_hat_sqrt, value=-lr)

                if log_stats:
                    adam_step_norms_sq += (p.data.float() - p_before).norm().square().item()

                # Phase B: lr-scaled, per-coordinate pullback toward the EMA.
                # lam_i = lr * c * decay / (sqrt(v_hat_i) + eps), clamped to <= 1.
                # The reference point is the pre-Adam theta (p_before), so the full
                # Adam step from Phase A is always preserved (purely additive).
                if lambda_pullback > 0 and 'ema' in state:
                    # EMA decay schedule (same schedule as Phase C)
                    if ema_kappa is not None:
                        t_eff = max(0, t - ema_rho)
                        decay = max((1.0 + ema_gamma * t_eff) ** (-ema_kappa), 1e-4)
                    else:
                        decay = 1.0  # no decay when using fixed beta_ema

                    # raw_lam = decay / (sqrt(v_hat) + eps)  — before c and lr.
                    # Reuse v_hat_sqrt tensor (not needed after this, safe to mutate).
                    lam = v_hat_sqrt.reciprocal_().mul_(decay)

                    if log_stats:
                        raw_lam_sum += lam.sum().item()
                        raw_lam_max = max(raw_lam_max, lam.max().item())
                        raw_lam_min = min(raw_lam_min, lam.min().item())
                        total_count += lam.numel()

                    # multiply by lr * c
                    lam.mul_(lr * lambda_pullback)
                    if clamp_pullback:
                        if log_stats:
                            clipped_count += (lam >= 1.0).sum().item()
                        lam.clamp_(max=1.0)

                    if log_stats:
                        lam_sum += lam.sum().item()
                        lam_max = max(lam_max, lam.max().item())
                        lam_min = min(lam_min, lam.min().item())

                    # pullback = lam * (ema - p_before)
                    diff = state['ema'] - p_before  # 1 new tensor (p_before needed later)
                    diff.mul_(lam)                   # in-place, now = pullback displacement
                    p.data.add_(diff)
                    if log_stats:
                        pullback_norms_sq += diff.norm().square().item()

                if log_stats:
                    update_norms_sq += (p.data.float() - p_before).norm().square().item()

                # Phase C: EMA update (gated by ema_update_freq)
                if 'ema' in state:
                    ema_update_freq = group['ema_update_freq']
                    if (t - state['last_ema_step']) % ema_update_freq == 0:
                        if ema_kappa is not None:
                            t_eff = max(0, t - ema_rho)
                            beta_ema_t = max((1.0 + ema_gamma * t_eff) ** (-ema_kappa), 1e-4)
                        else:
                            beta_ema_t = beta_ema
                        state['ema'].mul_(1.0 - beta_ema_t).add_(p.data.float(), alpha=beta_ema_t)
                        state['last_ema_step'] = t
                    # Keep theta in sync every step for train/eval swaps
                    state['theta'].copy_(p.data)

            # Aggregate per-step stats (only when logging)
            if log_stats:
                if total_count > 0:
                    self._last_raw_lambda_mean = raw_lam_sum / total_count
                    self._last_raw_lambda_max = raw_lam_max
                    self._last_raw_lambda_min = raw_lam_min
                    self._last_lambda_mean = lam_sum / total_count
                    self._last_lambda_max = lam_max
                    self._last_lambda_min = lam_min
                    self._last_lambda_clipped_frac = clipped_count / total_count
                    # effective = lr * c * raw (before clamp)
                    self._last_effective_lambda_mean = lr * lambda_pullback * (raw_lam_sum / total_count)
                    self._last_effective_lambda_max = lr * lambda_pullback * raw_lam_max
                    self._last_effective_lambda_min = lr * lambda_pullback * raw_lam_min
                self._last_adam_step_norm = math.sqrt(adam_step_norms_sq)
                self._last_pullback_norm = math.sqrt(pullback_norms_sq)
                self._last_update_norm = math.sqrt(update_norms_sq)

        return loss

    def get_step_stats(self) -> dict:
        """Return optimizer step stats for logging."""
        stats = {}
        if self._last_raw_lambda_mean is not None:
            stats['raw_lambda_mean'] = self._last_raw_lambda_mean
            stats['raw_lambda_max'] = self._last_raw_lambda_max
            stats['raw_lambda_min'] = self._last_raw_lambda_min
        if self._last_lambda_mean is not None:
            stats['lambda_mean'] = self._last_lambda_mean
            stats['lambda_max'] = self._last_lambda_max
            stats['lambda_min'] = self._last_lambda_min
            stats['lambda_clipped_frac'] = self._last_lambda_clipped_frac
        if self._last_effective_lambda_mean is not None:
            stats['effective_lambda_mean'] = self._last_effective_lambda_mean
            stats['effective_lambda_max'] = self._last_effective_lambda_max
            stats['effective_lambda_min'] = self._last_effective_lambda_min
        if self._last_adam_step_norm is not None:
            stats['adam_step_norm'] = self._last_adam_step_norm
        if self._last_pullback_norm is not None:
            stats['pullback_norm'] = self._last_pullback_norm
        if self._last_update_norm is not None:
            stats['update_norm'] = self._last_update_norm
        return stats

    def get_ema_beta_t(self) -> Optional[float]:
        """Return the current EMA interpolation rate for logging."""
        for group in self.param_groups:
            if not group['use_ema_eval'] and group['lambda_pullback'] == 0:
                return None
            ema_kappa = group['ema_kappa']
            if ema_kappa is not None:
                for p in group['params']:
                    state = self.state[p]
                    if 'step' in state:
                        t = state['step']
                        t_eff = max(0, t - group['ema_rho'])
                        return max((1.0 + group['ema_gamma'] * t_eff) ** (-ema_kappa), 1e-4)
            else:
                return group['beta_ema']
        return None


__all__ = ["PACE"]
