# Reproducing the paper

The repository ships **no result data and no plots** — only the optimizer, a
training pipeline, and this **static map** from each paper figure to the exact
training configurations behind it. To reproduce a figure you train its runs and
plot the resulting JSONs however you like.

Each run writes `{config, summary, eval_logs}` to a JSON; `summary.final_val_loss`
is the headline number and `eval_logs` is the validation trajectory.

## Workflow

1. Prepare the dataset once (tokenized + prompt-masked):
   ```python
   from pace.data import prepare_sft
   prepare_sft("data/smoltalk", tokenizer_name="HuggingFaceTB/SmolLM2-1.7B",
               dataset_name="HuggingFaceTB/smol-smoltalk", max_seq_len=1280)
   ```
2. Print the exact `pace-train` commands for a figure:
   ```bash
   pace-figure fig=fig1
   ```
3. Run them (or run `pace-figure fig=fig1 run=true data=data/smoltalk`).

The machine-readable specs live in `figures/conf/fig/<id>.yaml`; `pace-figure
fig=list` prints the catalog.

## Fig 1 — best-per-method finetuning (worked example)

Three models, three methods. Baselines use their best decay schedule; PACE uses a
constant learning rate. The nine runs:

| model | method | lr | c | kappa | schedule |
|-------|--------|----|---|-------|----------|
| SmolLM2-1.7B | AdamW | 1e-3 | – | – | cos |
| SmolLM2-1.7B | EMA | 1e-3 | 0 | 0.5 | cos |
| SmolLM2-1.7B | PACE | 1e-3 | 0.15 | 0.2 | const |
| Qwen3-1.7B | AdamW | 5e-4 | – | – | cos |
| Qwen3-1.7B | EMA | 5e-4 | 0 | 0.5 | cos |
| Qwen3-1.7B | PACE | 5e-4 | 0.25 | 0.5 | const |
| Gemma3-1B | AdamW | 1e-3 | – | – | cos |
| Gemma3-1B | EMA | 1e-3 | 0 | 0.5 | cos |
| Gemma3-1B | PACE | 1e-3 | 0.15 | 0.5 | const |

e.g. the SmolLM2-1.7B PACE run:
```bash
pace-train model=smollm2_1.7b dataset=smoltalk optimizer=pace schedule=const seed=42 \
    optimizer.lr=0.001 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
    dataset.path=data/smoltalk
```
Expected `final_val_loss`: PACE 0.6911 / 0.7103 / 0.8576 (SmolLM2 / Qwen3 / Gemma3);
AdamW 0.7002 / 0.7488 / 0.9249; EMA 0.7021 / 0.7528 / 0.9279.

## Catalog (all paper figures)

`pace-figure fig=list` prints this; full machine-readable per-curve specs are added
under `figures/conf/fig/` (Fig 1 is the reference example). The figure families:

| paper | id | family |
|-------|----|--------|
| Fig 1 | `fig1` | SmolLM2-1.7B / Qwen3-1.7B / Gemma3-1B finetuning, best-per-method |
| Fig 2,12,13 | `fig_clip_frac_body`, `fig_appendix_clip_frac*` | SmolLM2-1.7B pullback-gain saturation |
| Fig 3,8 | `fig3_135m_grid`, `fig_appendix_135m_full_grid` | SmolLM2-135M (c, kappa, uf) grid |
| Fig 5,9,18,19,20 | `fig5_pretraining`, `fig_appendix_pretrain_*` | GPT-2-124M FineWeb pretraining |
| Fig 6 | `fig_appendix_c_kappa_all_models` | kappa x c, three models |
| Fig 7 | `fig_appendix_train_curves` | training-loss comparison |
| Fig 10 | `fig_appendix_tulu_qwen` | Qwen3-1.7B on Tulu-3 |
| Fig 11 | `fig_appendix_full_train_curves` | Fig 1 with last-25% zoom |
| Fig 14,15 | `fig_appendix_wsd_*` | SmolLM2-135M under decay schedules |
| Fig 16 | `fig_appendix_fig1_all_schedules` | three models x three schedules |
| Fig 17 | `fig_appendix_multiseed_robust` | seed robustness (42/1337/2024) |

All runs use effective batch size 256, 50-step warmup, and the hyperparameters in
each figure's config. See the paper for the exact sweep ranges of the grid/heatmap
figures.
