# PACE

**PACE** is an AdamW variant for language-model training that adds a small,
per-coordinate *pullback* toward an exponential moving average (EMA) of the
iterates. The pullback gain is scaled by the learning rate and inversely by the
Adam second-moment estimate, which lets PACE train at a **constant learning
rate** and still match or beat carefully decayed AdamW/EMA baselines — no
learning-rate schedule to tune.

This repository contains a clean, self-contained implementation of the optimizer,
a Hydra-configured training pipeline, and a one-command reproduction of every
figure in the paper.

```
theta_{t+1} = (theta_t - AdamW_step) + lambda_{t,i} * (ema_{t-1} - theta_t)
lambda_{t,i} = clamp( lr * c * decay_t / (sqrt(v_hat_i) + eps),  max = 1 )
```

Setting the pullback strength `c = 0` recovers plain AdamW exactly; adding
EMA evaluation recovers the EMA baseline. PACE is therefore a strict superset of
both.

## Install (uv)

This project uses [uv](https://docs.astral.sh/uv/). Install uv, then:

```bash
uv sync --extra experiments     # optimizer + training + figure deps, from uv.lock
```

For just the optimizer (to drop `PACE` into your own training code):

```bash
uv add pace-optimizer           # or: pip install -r requirements.txt
```

```python
from pace import PACE
opt = PACE(model.parameters(), lr=1e-3, lambda_pullback=0.15, ema_kappa=0.2,
           use_ema_eval=True)        # c=0 -> plain AdamW
# ... opt.step() as usual; call opt.eval()/opt.train() around validation to use EMA weights.
```

## Reproduce the paper figures

Every figure is one command. Curated result data ships in `data/results/`, so no
retraining is needed:

```bash
uv run pace-figure fig=fig1                     # headline finetuning figure
uv run pace-figure fig=fig1 fig.models=[Qwen3-1.7B]   # override inputs
uv run pace-figure fig=fig5_pretraining         # GPT-2 pretraining
```

Output PDFs/PNGs are written to `figures/rendered/`. Each figure has a config at
`figures/conf/fig/<id>.yaml` (its inputs: models, schedules, sweep axes, metric)
and a builder at `figures/_figs/<id>.py`.

| `fig=` | Figure | Models |
|--------|--------|--------|
| `fig1` | Best-per-method validation loss | SmolLM2-1.7B, Qwen3-1.7B, Gemma3-1B |
| `fig_clip_frac_body` | Clip-fraction of the pullback gain | SmolLM2-1.7B |
| `fig3_135m_grid` | Update-freq × pullback grid | SmolLM2-135M |
| `fig5_pretraining` | Pretraining (const / c-sweep / WSD) | GPT-2-124M |
| `fig_appendix_c_kappa_all_models` | κ × c response | all three |
| `fig_appendix_train_curves` | Training-loss comparison | SmolLM2-135M / Qwen3 |
| `fig_appendix_135m_full_grid` | Full (c, κ, uf) grid | SmolLM2-135M |
| `fig_appendix_tulu_qwen` | Qwen3 on Tulu-3 | Qwen3-1.7B |
| `fig_appendix_full_train_curves` | Fig 1 + last-25% zoom | all three |
| `fig_appendix_clip_frac` / `_vs_c` | Coordinate saturation | SmolLM2-1.7B |
| `fig_appendix_wsd_val_curves` / `_wsd_finetune` | Under decay schedules | SmolLM2-135M |
| `fig_appendix_fig1_all_schedules` | 3 models × 3 schedules | all three |
| `fig_appendix_multiseed_robust` | Seed robustness | all three, seeds 42/1337/2024 |
| `fig_appendix_pretrain_kappa_sweep` / `_csweep` / `_csweep_kappa` / `_river` | Pretraining ablations | GPT-2-124M |

The large GPT-2 pretraining sweep data is fetched on demand:

```bash
bash data/download_extended.sh
```

## Train

Training is configured with [Hydra](https://hydra.cc). Config groups live in
`src/pace/conf/`: `model/`, `dataset/`, `optimizer/` (`adamw` | `ema` | `pace`),
and `schedule/` (`const` | `cos` | `wsd` | `cosine_to_zero`).

First build a tokenized, prompt-masked dataset (once):

```python
from pace.data import prepare_sft
prepare_sft("data/smoltalk", tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
            dataset_name="HuggingFaceTB/smol-smoltalk", max_seq_len=1280)
```

Then train. PACE at constant LR (the headline setting):

```bash
uv run pace-train model=smollm2_1.7b optimizer=pace schedule=const \
    optimizer.lr=1e-3 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
    dataset.path=data/smoltalk
```

Baselines (decayed AdamW / EMA):

```bash
uv run pace-train model=smollm2_1.7b optimizer=adamw schedule=cos dataset.path=data/smoltalk
uv run pace-train model=smollm2_1.7b optimizer=ema   schedule=wsd dataset.path=data/smoltalk
```

Key knobs: `optimizer.lambda_pullback` (=c, pullback strength), `optimizer.ema_kappa`
(=κ, EMA decay exponent), `optimizer.ema_update_freq` (=uf), `seed`,
`training.batch_size`, `training.gradient_accumulation_steps`,
`training.eval_interval`. Results are written as a JSON
(`{config, summary, eval_logs}`) under `runs/`; `summary.final_val_loss` is the
plotted metric. W&B logging is off by default (`wandb.enabled=true` to turn on).

## Tests

```bash
uv run python tests/test_optimizer_parity.py   # PACE == reference impl (bit-identical)
uv run python tests/smoke_train.py             # few-step GPU training sanity (needs a GPU)
```

## Relation to prior work

PACE's iterate-averaging component builds on the line of work on EMA / averaged
optimizers for language models, in particular BEMA (Block & Zhang, 2025) and the
broader literature on iterate averaging (Polyak–Ruppert). PACE differs in that
the averaging is coupled to the optimizer step through a learning-rate-scaled,
per-coordinate pullback, which is what enables strong constant-learning-rate
training. See the paper for a full discussion and comparisons.

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
