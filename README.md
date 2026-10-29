# PACE

PyTorch implementation of PACE, the optimizer from

> **Training for the Model You Return: Improving Optimization for Iterate-Averaged
> Language Models.** Kwok Chun Au and Adam Block, 2026.
> [arXiv:2606.25086](https://arxiv.org/abs/2606.25086)

PACE is AdamW plus a per-coordinate pullback of the iterate toward an exponential moving
average (EMA) of past iterates. The EMA is the model you return. In the paper this
improves over AdamW and EMA-evaluated AdamW on 1-2B-parameter fine-tuning and GPT-2
pretraining, across learning rates and schedules.

## Install

```bash
pip install -e .                   # the optimizer (torch only)
pip install -r requirements.txt    # also training / reproduction scripts
```

## Usage

```python
from pace import PACE

opt = PACE(model.parameters(), lr=1e-3,
           lambda_pullback=0.15,   # c, pullback strength
           ema_kappa=0.2,          # kappa, EMA decay exponent
           use_ema_eval=True)      # eval()/train() swap the EMA weights in and out

for batch in loader:
    loss = model(**batch).loss
    loss.backward()
    opt.step()
    opt.zero_grad()

opt.eval()     # parameters are now the EMA
validate(model)
opt.train()    # back to the training iterate
```

Update rule, per coordinate, at step t:

```
theta'   = theta_t - AdamW_step(theta_t)
decay_t  = (1 + t)^(-kappa)
lambda_t = min(1, lr * c * decay_t / (sqrt(v_hat) + eps))
theta_t+1 = theta' + lambda_t * (ema_t-1 - theta_t)
ema_t    = (1 - decay_t) * ema_t-1 + decay_t * theta_t+1
```

`lambda_pullback=0` gives AdamW (`use_ema_eval=False`) or the EMA baseline
(`use_ema_eval=True`). Other arguments: `betas`, `eps`, `weight_decay` as in AdamW;
`ema_update_freq` to update the EMA every k steps; `ema_rho`, `ema_gamma` for the general
schedule `(1 + gamma * max(0, t - rho))^(-kappa)`; `beta_ema` for a fixed EMA rate when
`ema_kappa=None`; `log_stats=True` to record pullback statistics via `get_step_stats()`.

## Training

```bash
python prepare_data.py --out data/smoltalk --tokenizer HuggingFaceTB/SmolLM2-1.7B

python train.py model=smollm2_1.7b optimizer=pace schedule=const \
    optimizer.lr=1e-3 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
    dataset.path=data/smoltalk
```

Configs are Hydra groups under `configs/`: `model` (smollm2_135m, smollm2_1.7b,
qwen3_1.7b, gemma3_1b), `dataset` (smoltalk, tulu3), `optimizer` (adamw, ema, pace),
`schedule` (const, cos, wsd). Each run writes `runs/<run_name>.json` with the config,
`summary.final_val_loss` (the number reported in the paper), and the eval and step logs.
Validation loss is measured on the EMA weights for EMA and PACE, and on the final
iterate for AdamW.

## Reproducing the paper

`figures/<id>.yaml` lists fine-tuning run specifications, using the camera-ready
figure numbers. Figure 1 uses WSD for all methods and seeds 42, 1337, and 2024;
`fig_best_per_method_single` is the single-seed comparison in Figure 16.
Model configs set the paper's batch sizes and step limits. Figure 7 overrides the
135M limit to 3,557 steps. Data must be prepared separately with each model's
tokenizer. These commands train runs; they do not plot the paper figures.

```bash
python reproduce.py list
python reproduce.py fig1                               # print the commands
python reproduce.py fig1 --run --data data/smoltalk    # run them
```

`results/` holds two single-seed reference runs: SmolLM2-1.7B PACE and one
SmolLM2-135M PACE cell. They are not the three-seed averages in Figure 1. The GPT-2
pretraining experiments used a separate JAX codebase and are not included here.
Schedule-Free and token-budget branch
experiments are also not included; Figure 7 specifies only AdamW, EMA, and PACE.
Figure 6 lists displayed pullback-strength bins; exact source values and the
original result logs are not bundled.

## Tests

```bash
pytest tests/
```

## Citation

```bibtex
@article{au2026pace,
  title   = {Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models},
  author  = {Au, Kwok Chun and Block, Adam},
  journal = {arXiv preprint arXiv:2606.25086},
  year    = {2026}
}
```

MIT license.
