"""Fine-tune a causal LM with PACE, AdamW, or EMA-evaluated AdamW.

    python train.py model=smollm2_1.7b optimizer=pace schedule=const \
        optimizer.lr=1e-3 optimizer.lambda_pullback=0.15 optimizer.ema_kappa=0.2 \
        dataset.path=data/smoltalk

Configs live in configs/. Each run writes one JSON (config, summary, eval and step
logs) to <out_dir>/<run_name>.json.
"""

import json
import math
import os
import random
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

from pace import PACE
from prepare_data import collate_fn, load_prepared

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_lambda(name, total_steps, warmup_steps=0, decay_frac=0.2):
    """Multiplier for LambdaLR. const / cos (to zero) / wsd (linear decay over the
    last decay_frac of training). All start with a linear warmup."""

    def f(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if name == "const":
            return 1.0
        if name == "cos":
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if name == "wsd":
            decay_steps = max(1, int(round(decay_frac * total_steps)))
            decay_start = total_steps - decay_steps
            if step < decay_start:
                return 1.0
            return max(0.0, 1.0 - (step - decay_start) / decay_steps)
        raise ValueError(f"unknown schedule {name}")

    return f


def run_name(cfg):
    o = cfg.optimizer
    eff_bs = cfg.training.batch_size * cfg.training.gradient_accumulation_steps
    parts = [o.name, f"lr{o.lr}", f"bs{eff_bs}"]
    if o.use_ema_eval and o.get("ema_kappa") is not None:
        parts.append(f"k{o.ema_kappa}")
    if o.lambda_pullback > 0:
        parts.append(f"lp{o.lambda_pullback}")
    parts += [f"sched-{cfg.schedule.name}", f"s{cfg.seed}"]
    return "_".join(parts)


def build_optimizer(model, o):
    if o.name == "adamw" and not o.use_ema_eval and o.lambda_pullback == 0:
        return torch.optim.AdamW(model.parameters(), lr=o.lr, betas=tuple(o.betas),
                                 eps=o.eps, weight_decay=o.weight_decay)
    opt = PACE(model.parameters(), lr=o.lr, betas=tuple(o.betas), eps=o.eps,
               weight_decay=o.weight_decay, lambda_pullback=o.lambda_pullback,
               clamp_pullback=o.get("clamp_pullback", True), beta_ema=o.beta_ema,
               use_ema_eval=o.use_ema_eval, ema_kappa=o.get("ema_kappa"),
               ema_rho=o.get("ema_rho", 0.0), ema_gamma=o.get("ema_gamma", 1.0),
               ema_update_freq=o.get("ema_update_freq", 1),
               log_stats=o.get("log_stats", True))
    opt.train()
    return opt


@torch.no_grad()
def eval_loss(model, loader, max_samples, ctx):
    total, n_tok, n = 0.0, 0, 0
    for batch in loader:
        if max_samples is not None and n >= max_samples:
            break
        batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with ctx:
            out = model(**batch)
        n_lab = (batch["labels"] != -100).sum().item()
        total += out.loss.item() * n_lab
        n_tok += n_lab
        n += batch["input_ids"].size(0)
    return total / max(n_tok, 1)


@torch.no_grad()
def evaluate(model, optimizer, loader, max_samples, ctx, use_ema_eval):
    """Loss of the final iterate, and of the EMA weights when use_ema_eval is set.
    val_loss is the loss of the weights the method returns."""
    model.eval()
    vanilla = eval_loss(model, loader, max_samples, ctx)
    ema = None
    if use_ema_eval:
        optimizer.eval()
        ema = eval_loss(model, loader, max_samples, ctx)
        optimizer.train()
    model.train()
    return {"val_loss": ema if use_ema_eval else vanilla,
            "vanilla_val_loss": vanilla, "ema_val_loss": ema}


def opt_stats(optimizer):
    if hasattr(optimizer, "get_step_stats"):
        return {k: float(v) for k, v in optimizer.get_step_stats().items()}
    return {}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    name = run_name(cfg)
    print(name)
    print(OmegaConf.to_yaml(cfg))

    wandb = None
    if cfg.wandb.enabled:
        import wandb
        wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    train_ds, val_ds = load_prepared(cfg.dataset.path, cfg.dataset.get("max_samples"))
    print(f"train {len(train_ds)} | val {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True,
                              generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=torch.bfloat16).to(DEVICE)
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    use_ema_eval = bool(cfg.optimizer.use_ema_eval)
    optimizer = build_optimizer(model, cfg.optimizer)

    ctx = torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE)
    tr = cfg.training
    grad_accum = tr.gradient_accumulation_steps
    total_steps = (len(train_loader) // grad_accum) * tr.epochs
    if tr.get("max_steps"):
        total_steps = min(total_steps, tr.max_steps)
    scheduler = LambdaLR(optimizer, lr_lambda(cfg.schedule.name, total_steps,
                                              cfg.schedule.get("warmup_steps", 0),
                                              cfg.schedule.get("decay_frac", 0.2)))
    print(f"total_steps {total_steps} | schedule {cfg.schedule.name}")

    eval_logs, step_logs = [], []
    best_val = float("inf")
    step = 0
    t0 = time.time()
    model.train()

    def do_eval(epoch, train_loss, tag=""):
        nonlocal best_val
        res = evaluate(model, optimizer, val_loader, tr.get("eval_samples"), ctx, use_ema_eval)
        best_val = min(best_val, res["val_loss"])
        entry = {"step": step, "epoch": epoch, "train_loss": train_loss, **res,
                 "best_val_loss": best_val, "elapsed_sec": time.time() - t0, "tag": tag}
        entry.update(opt_stats(optimizer))
        eval_logs.append(entry)
        if wandb:
            wandb.log({"val/loss": res["val_loss"], "val/vanilla": res["vanilla_val_loss"]}, step=step)
        print(f"step {step}/{total_steps} | train {train_loss:.4f} | val {res['val_loss']:.4f}"
              + (f" | vanilla {res['vanilla_val_loss']:.4f}" if res["ema_val_loss"] is not None else ""),
              flush=True)
        return res

    done = False
    for epoch in range(tr.epochs):
        if done:
            break
        micro, accum = 0, 0.0
        for batch in train_loader:
            micro += 1
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with ctx:
                loss = model(**batch).loss / grad_accum
            loss.backward()
            accum += loss.item()
            if not math.isfinite(accum):
                print(f"non-finite loss at micro step {micro}", flush=True)
                sys.exit(1)
            if micro % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), tr.grad_clip if tr.grad_clip > 0 else float("inf"))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                train_loss, accum = accum, 0.0
                lr = optimizer.param_groups[0]["lr"]
                step_logs.append({"step": step, "train_loss": train_loss,
                                  "grad_norm": grad_norm.item(), "lr": lr,
                                  "elapsed_sec": time.time() - t0})
                if wandb:
                    wandb.log({"train/loss": train_loss, "train/grad_norm": grad_norm.item(),
                               "train/lr": lr,
                               **{f"train/{k}": v for k, v in opt_stats(optimizer).items()}}, step=step)
                if step % tr.eval_interval == 0 or step == 1:
                    do_eval(epoch + 1, train_loss)
                if tr.get("max_steps") and step >= tr.max_steps:
                    done = True
                    break

    final_train = step_logs[-1]["train_loss"] if step_logs else 0.0
    res = do_eval(tr.epochs, final_train, tag="final")
    total_sec = time.time() - t0
    summary = {"best_val_loss": best_val, "final_train_loss": final_train,
               "final_val_loss": res["val_loss"], "total_sec": total_sec,
               "total_steps": step}

    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"config": OmegaConf.to_container(cfg, resolve=True), "summary": summary,
                   "eval_logs": eval_logs, "step_logs": step_logs}, f, indent=2)
    print(f"final_val_loss {res['val_loss']:.4f} | best_val_loss {best_val:.4f} | {path}")
    if wandb:
        wandb.finish(quiet=True)
    return best_val


if __name__ == "__main__":
    main()
