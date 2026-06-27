# PACE

📄 Paper: **[Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models](https://arxiv.org/abs/2606.25086)** (Au & Block, 2026 · arXiv:2606.25086)

**PACE** is a drop-in optimizer (a `torch.optim.Optimizer`) — AdamW plus a small,
per-coordinate pullback that nudges the weights toward an exponential moving average
(EMA) of recent weights, with a clipped strength.

In the paper it beats AdamW and EMA-evaluated AdamW on 1–2B fine-tuning and on
GPT-2 / FineWeb pretraining, across a range of learning rates and schedules.

Each optimizer step `t` (per coordinate `i`), as implemented:

```
# 1. AdamW step (decoupled weight decay + Adam), using the second moment v_hat
theta'      = theta_t - AdamW_step(theta_t)

# 2. pullback toward the EMA (applied every step; reference is the PRE-Adam theta_t)
decay_t     = (1 + gamma * max(0, t - rho))^(-kappa)      # gamma=1, rho=0 by default -> (1+t)^(-kappa), floored at 1e-4
lambda_{t,i}= clamp( lr * c * decay_t / (sqrt(v_hat_i) + eps),  max = 1 )
theta_{t+1} = theta' + lambda_{t,i} * (ema_{t-1} - theta_t)

# 3. EMA update (every `ema_update_freq` steps)
ema_t       = (1 - beta_t) * ema_{t-1} + beta_t * theta_{t+1},   beta_t = decay_t
```

With `use_ema_eval=True` the model you evaluate/return is the EMA (`ema_t`). Note the
pullback (step 2) is applied on every step; only the EMA update (step 3) honors
`ema_update_freq`.

## Use the optimizer

The optimizer is the whole product and depends only on PyTorch:

```bash
pip install pace-optimizer          # or: uv add pace-optimizer
```

```python
from pace import PACE

opt = PACE(model.parameters(), lr=1e-3,
           lambda_pullback=0.15,    # c — pullback strength
           ema_kappa=0.2,           # kappa — EMA decay exponent
           use_ema_eval=True)       # evaluate (and return) the EMA weights

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step(); opt.zero_grad()

opt.eval()    # swap to EMA weights for validation
validate(model)
opt.train()   # swap back to the training weights
```

Two knobs recover the baselines:
- `use_ema_eval` chooses what you **return/evaluate** — the EMA weights (`True`) or the
  final iterate (`False`).
- `lambda_pullback` (c) chooses whether **training** pulls toward the EMA (`c>0`) or not (`c=0`).

So: `c=0, use_ema_eval=True` → the **EMA baseline** (AdamW training, EMA returned);
`c=0, use_ema_eval=False` → **plain AdamW** (matches `torch.optim.AdamW` up to
floating-point ordering, ~1e-6 in the parity test); `c>0` → **PACE**. PACE generalizes
both baselines.

### All parameters

`PACE(params, ...)` — constructor arguments (paper symbol in parentheses):

| argument | symbol | default | meaning |
|----------|--------|---------|---------|
| `params` | – | – | iterable of parameters or param groups (standard `torch.optim` API) |
| `lr` | η | `1e-4` | AdamW learning rate |
| `betas` | β₁,β₂ | `(0.9, 0.999)` | AdamW moment decay rates |
| `eps` | ε | `1e-8` | AdamW denominator epsilon (also the floor in the pullback gain) |
| `weight_decay` | – | `0.01` | decoupled (AdamW) weight decay |
| `lambda_pullback` | c | `0.0` | pullback strength. `0` = no pullback (AdamW/EMA baseline); `>0` = PACE |
| `clamp_pullback` | – | `True` | clamp each per-coordinate gain to `≤ 1` (interpolate toward the EMA, never past it) |
| `beta_ema` | – | `0.01` | fixed EMA decay, used **only** when `ema_kappa is None` |
| `use_ema_eval` | – | `False` | if `True`, `eval()` returns the EMA weights and `train()` restores the iterate |
| `ema_kappa` | κ | `None` | exponent of the decaying EMA schedule `(1+γ·max(0,t−ρ))^(−κ)`. `None` ⇒ use the fixed `beta_ema` |
| `ema_rho` | ρ | `0.0` | steps to hold before the schedule decays. `0` = the paper's `(1+γ·t)^(−κ)` |
| `ema_gamma` | γ | `1.0` | schedule rate |
| `ema_update_freq` | uf | `1` | update the EMA every `uf` steps (the pullback itself is applied every step) |
| `log_stats` | – | `False` | compute per-step telemetry (λ stats, clip fraction, update norms) via `get_step_stats()`; off by default because each stat is a `.item()` GPU sync. The optimizer math is identical either way |

Per step `t`, per coordinate `i`: `λ = clamp(η·c·decay/(√v̂ᵢ+ε), max=1)` with
`decay = max((1+γ·max(0,t−ρ))^(−κ), 1e-4)`; the weights move
`θ ← (θ − AdamW_step) + λ·(ema_{t−1} − θ_t)`, then the EMA updates every `uf` steps with
`β = decay`. (The paper's Algorithm 1 box places the pullback inside the `t mod uf`
gate; the code applies it every step — identical for `uf=1`, which all paper runs use.)

## Reproduce the paper

The repo ships **no result data and no plots** — only the optimizer, the training
pipeline, and a **static map** from each paper figure to the exact training configs.
Reproducing a figure means training its runs and reading the result JSONs.

```bash
uv sync --extra experiments        # training + reproduction deps
uv run pace-figure fig=list        # the figure catalog
uv run pace-figure fig=fig1        # print the exact pace-train commands for Fig 1
```

Each figure config (`src/pace_figures/conf/fig/<id>.yaml`) lists its curves; every
curve is a complete `pace-train` spec. For example Fig 1's nine runs (baselines use
their best decay schedule, PACE uses a constant LR):

| model | method | lr | c | kappa | schedule |
|-------|--------|----|---|-------|----------|
| SmolLM2-1.7B | AdamW / EMA / PACE | 1e-3 | – / 0 / 0.15 | – / 0.5 / 0.2 | cos / cos / const |
| Qwen3-1.7B | AdamW / EMA / PACE | 5e-4 | – / 0 / 0.25 | – / 0.5 / 0.5 | cos / cos / const |
| Gemma3-1B | AdamW / EMA / PACE | 1e-3 | – / 0 / 0.15 | – / 0.5 / 0.5 | cos / cos / const |

Expected PACE `final_val_loss`: 0.6911 / 0.7103 / 0.8576. Train them with
`uv run pace-figure fig=fig1 run=true data=<prepared dataset>` or by running the
printed commands.

**Scope.** The shipped trainer is a fine-tuning (SFT) trainer, so the fine-tuning
figures (Fig 1, 3, 6, 7, 10, 11, 14–17 and the clip-fraction figures) reproduce
end-to-end. The GPT-2 / FineWeb **pretraining** figures (Fig 5, 9, 18, 19, 20) are
catalogued in the map for completeness, but were produced with a separate pretraining
pipeline that is not included here; `pace-figure` prints their configs but the SFT
trainer cannot run a raw-text corpus.

## Train

Training is configured with [Hydra](https://hydra.cc); config groups live in
`src/pace/conf/`: `model/`, `dataset/`, `optimizer/` (`adamw` | `ema` | `pace`),
`schedule/` (`const` | `cos` | `wsd`; `cos` decays to zero). Prepare a dataset once
with `pace.data.prepare_sft`, then:

```bash
pace-train model=smollm2_1.7b optimizer=pace schedule=const \
    optimizer.lr=1e-3 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
    dataset.path=data/smoltalk
```

A run writes `{config, summary, eval_logs, step_logs}` as JSON under `runs/`;
`summary.final_val_loss` is the plotted metric (`final_train_loss` is recorded too).
The metric convention follows the paper: `val_loss`/`ema_val_loss` are the **returned
(EMA)** weights, `vanilla_val_loss` is the final iterate. W&B logging is off by default,
and the optimizer's per-step telemetry (gradient/pullback norms, clip fraction) is gated
behind `optimizer.log_stats` — leave it off for the fastest path. Training runs a
single pass and does not checkpoint/resume; size `training.max_steps` accordingly.

### Data recipe (exact, reproducible)

`pace.data.prepare_sft` rebuilds the paper's SFT data byte-for-byte. Defaults:
`dataset="HuggingFaceTB/smol-smoltalk"`, `max_seq_len=1280`, `subset_ratio=0.5`,
`seed=42`, using the dataset's native `train` / `test` splits (a fixed seeded
`random.sample` subset of each). With a SmolLM2 tokenizer this yields the same
**113,832 train / 11,986 val** tokenized examples used for the figures. Tokenization is
hand-rolled ChatML — each turn is `<|im_start|>{role}\n{content}<|im_end|>\n` with the
header and body tokenized separately and `add_special_tokens=False` (so **no BOS** is
prepended and **no system turn** is injected); labels are `-100` everywhere except the
assistant turn's **body** (the assistant header is masked). `max_train`/`max_val`
instead stream a small deterministic first-N slice for a quick self-contained test.

## Tests

```bash
uv run pytest -q                   # fast: optimizer parity + config map + result fixtures
```

- `tests/test_optimizer_parity.py` — `c=0 == torch.AdamW` (up to fp ordering), the EMA
  eval-swap never perturbs training, and the core optimizer imports with torch only (plus
  bit-identical parity vs a reference if `PACE_REF_OPTIMIZER` is set). Runs anywhere.
- `tests/test_cli_map.py` — the figure→config reproduction map is complete and every
  curve expands to a well-formed `pace-train` command.
- `tests/test_result_fixtures.py` — checks the two bundled PACE result fixtures
  (`tests/fixtures/results/`, one SmolLM2-135M and one SmolLM2-1.7B run) parse, are
  genuinely PACE runs, and match their recorded **train and val** loss. Offline, no GPU.
- `tests/test_train_smoke.py` — **self-contained training test**: downloads a model +
  smol-smoltalk from HuggingFace and runs PACE end-to-end (no bundled data), then checks
  the final train/val loss against a committed reference within `PACE_SMOKE_TOL` (loose by
  default, since same-seed runs drift across hardware). Length is env-configurable:
  ```bash
  uv run pytest tests/test_train_smoke.py                       # SmolLM2-135M, 20 steps
  PACE_TEST_STEPS=200 uv run pytest tests/test_train_smoke.py   # train longer
  PACE_TEST_TRAIN=all PACE_TEST_STEPS=1778 PACE_TEST_BIG=1 \
      uv run pytest tests/test_train_smoke.py                   # full-scale, incl. 1.7B
  ```

Anything runnable as a console command also runs as a module:
`python -m pace.train …` and `python -m pace_figures.reproduce fig=…`.

## Citation

Paper: [Training for the Model You Return: Improving Optimization for
Iterate-Averaged Language Models](https://arxiv.org/abs/2606.25086) (arXiv:2606.25086).

```bibtex
@article{au2026pace,
  title         = {Training for the Model You Return: Improving Optimization for
                   Iterate-Averaged Language Models},
  author        = {Au, Kwok Chun and Block, Adam},
  journal       = {arXiv preprint arXiv:2606.25086},
  year          = {2026},
  eprint        = {2606.25086},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2606.25086}
}
```

## License

MIT — see [LICENSE](LICENSE).
