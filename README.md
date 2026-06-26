# PACE

**PACE** is a drop-in AdamW variant for language-model training. It adds a small,
per-coordinate *pullback* toward an exponential moving average (EMA) of the
iterates; the pullback gain is scaled by the learning rate and inversely by the
Adam second-moment estimate. The effect is that PACE trains well at a **constant
learning rate** and matches or beats carefully decayed AdamW/EMA baselines — no
learning-rate schedule to tune.

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

Setting `lambda_pullback=0` makes the update **bit-identical to `torch.optim.AdamW`**;
adding `use_ema_eval=True` gives the EMA-evaluation baseline. PACE is a strict
superset of both.

## Reproduce the paper

The repo ships **no result data and no plots** — only the optimizer, a training
pipeline, and a **static map** from each paper figure to the exact training
configs. See [REPRODUCE.md](REPRODUCE.md). In short:

```bash
uv sync --extra experiments        # training + reproduction deps
pace-figure fig=list               # the figure catalog
pace-figure fig=fig1               # print the exact pace-train commands for Fig 1
```

Each figure config (`figures/conf/fig/<id>.yaml`) lists its runs; every run is a
normal `pace-train` invocation, so reproducing a figure is just training those
configs and reading the result JSONs.

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
uv run pytest -q                   # config map + standalone gate + optimizer parity
```

`tests/test_optimizer_parity.py` checks `c=0 == torch.AdamW` and that the EMA
eval-swap never perturbs training (and, if `PACE_REF_OPTIMIZER` points at a
reference implementation, bit-identical parity). `tests/test_cli_map.py` and
`tests/test_standalone.py` check the reproduction map and that the repo is
lightweight and self-contained. A few-step GPU training check is in
`tests/smoke_train.py`.

## Relation to prior work

PACE's iterate-averaging component builds on the line of work on EMA / averaged
optimizers for language models, in particular BEMA (Block & Zhang, 2025) and the
broader literature on iterate averaging (Polyak–Ruppert). PACE differs in that the
averaging is coupled to the optimizer step through a learning-rate-scaled,
per-coordinate pullback, which is what enables strong constant-learning-rate
training.

## Citation

```bibtex
@article{pace2026,
  title  = {Training for the Model You Return: Improving Optimization for
            Iterate-Averaged Language Models},
  author = {Au, Kwok Chun and Block, Adam},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
