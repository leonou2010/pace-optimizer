# PACE

📄 Paper: **[Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models](https://arxiv.org/abs/2606.25086)** (Au & Block, 2026 · arXiv:2606.25086)

**PACE** is a small wrapper around AdamW. During training it nudges the weights
toward an exponential moving average (EMA) of recent weights — per-coordinate, with
a clipped strength.

In the paper it beats AdamW and EMA-evaluated AdamW on 1–2B fine-tuning and on
GPT-2 / FineWeb pretraining, across a range of learning rates and schedules.

```
theta_{t+1} = (theta_t - AdamW_step) + lambda_{t,i} * (ema_{t-1} - theta_t)
lambda_{t,i} = clamp( lr * c * decay_t / (sqrt(v_hat_i) + eps),  max = 1 )
```

## Use the optimizer

The optimizer is the whole product and depends only on PyTorch:

```bash
pip install pace-optimizer          # or: uv add pace-optimizer
```

```python
from pace import PACE

opt = PACE(model.parameters(), lr=1e-3,
           lambda_pullback=0.15,    # c — pullback strength (0 ⇒ plain AdamW)
           ema_kappa=0.2,           # kappa — EMA decay exponent
           use_ema_eval=True)       # evaluate with the EMA weights

for batch in loader:
    loss = model(batch).loss
    loss.backward()
    opt.step(); opt.zero_grad()

opt.eval()    # swap to EMA weights for validation
validate(model)
opt.train()   # swap back to the training weights
```

Setting `lambda_pullback=0` reduces PACE to AdamW — matching `torch.optim.AdamW` up
to floating-point ordering (~1e-6 in the parity test); adding `use_ema_eval=True` then
recovers the EMA-evaluated baseline. So PACE generalizes both.

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

## Train

Training is configured with [Hydra](https://hydra.cc); config groups live in
`src/pace/conf/`: `model/`, `dataset/`, `optimizer/` (`adamw` | `ema` | `pace`),
`schedule/` (`const` | `cos` | `wsd` | `cosine_to_zero`). Prepare a dataset once
with `pace.data.prepare_sft`, then:

```bash
pace-train model=smollm2_1.7b optimizer=pace schedule=const \
    optimizer.lr=1e-3 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
    dataset.path=data/smoltalk
```

A run writes `{config, summary, eval_logs}` as JSON under `runs/`;
`summary.final_val_loss` is the plotted metric. W&B logging is off by default.

## Tests

```bash
uv run pytest -q                   # fast: optimizer parity + config map + standalone gate
```

- `tests/test_optimizer_parity.py` — `c=0 == torch.AdamW` (up to fp ordering) and that
  the EMA eval-swap never perturbs training (plus bit-identical parity vs a reference
  if `PACE_REF_OPTIMIZER` is set). Torch-only, runs anywhere.
- `tests/test_cli_map.py`, `tests/test_standalone.py` — the reproduction map is complete
  and the repo is lightweight/self-contained.
- `tests/test_train_smoke.py` — **self-contained training test**: downloads a model +
  smol-smoltalk from HuggingFace and runs PACE end-to-end (no bundled data). You choose
  how long via env vars:
  ```bash
  uv run pytest tests/test_train_smoke.py                       # SmolLM2-135M, 20 steps
  PACE_TEST_STEPS=200 uv run pytest tests/test_train_smoke.py   # train longer
  PACE_TEST_TRAIN=all PACE_TEST_STEPS=1778 PACE_TEST_BIG=1 \
      uv run pytest tests/test_train_smoke.py                   # full-scale, incl. 1.7B
  ```

Anything runnable as a console command also runs as a module:
`python -m pace.train …` and `python -m pace_figures.reproduce fig=…`.

## Relation to prior work

PACE's iterate-averaging component builds on the line of work on EMA / averaged
optimizers for language models, in particular **BEMA** (Block & Zhang, 2025 —
[paper](https://arxiv.org/abs/2508.00180), [code](https://github.com/abblock/bema))
and the broader literature on iterate averaging (Polyak–Ruppert). PACE differs in
that the averaging is coupled to the optimizer step through a learning-rate-scaled,
per-coordinate pullback.

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
