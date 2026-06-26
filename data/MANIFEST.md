# Bundled figure data

Each `results/<id>/` directory holds the minimal curated result files needed to
reproduce figure `<id>` via `uv run pace-figure fig=<id>`. These are a tiny
curated subset of the full research campaign (~60k runs); only the runs actually
plotted in the paper are shipped here.

Result JSONs follow the schema `{config, summary, eval_logs}` where
`summary.final_val_loss` is the plotted metric and `eval_logs` holds the
validation trajectory. Method is read from the config: PACE (`lambda_pullback>0`),
EMA (`lambda_pullback=0`, `use_ema_eval=true`), AdamW (otherwise). Some figures
ship a small precomputed cache (`_cache_clip_frac.json`) or summary CSV instead
of raw per-run JSONs.

| dir | figure | contents |
|-----|--------|----------|
| `fig1/` | Fig 1 | 9 per-run JSONs (3 models × {AdamW, EMA, PACE}) + `selection.json` |
| `multiseed/` | Figs 16, 17 | 3-seed const/wsd runs + manifest |
| `fig3_135m_grid/`, `fig_appendix_135m_full_grid/` | Figs 3, 8 | exp13 chunked results |
| `clip_frac/` | Figs 2, 12, 13 | `_cache_clip_frac.json` |
| `c_kappa_all_models/` | Fig 6 | per-run JSONs, 3 models |
| `wsd_finetune/` | Figs 14, 15 | WSD/const fine-tune JSONs |
| `fig5_pretraining/` | Fig 5 | GPT-2 raw curves + sweep |
| `fig_appendix_tulu_qwen/` | Fig 10 | Qwen3 Tulu-3 JSONs |
| `fig_appendix_pretrain_*/` | Figs 9, 18, 19, 20 | GPT-2 summary CSVs + curves |

Total in-repo: a few MB. The dense GPT-2 pretraining sweep (~180 MB) is fetched
separately via `data/download_extended.sh`.

These figures compare PACE, AdamW, and EMA only. Other baselines from the paper
are not reproduced in this repository; see the paper for those comparisons.
