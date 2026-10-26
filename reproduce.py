"""Print (or run) the train.py commands behind a paper figure.

    python reproduce.py list
    python reproduce.py fig1
    python reproduce.py fig1 --run --data data/smoltalk [--max-steps N] [--limit N]

figures/<id>.yaml lists the runs of each figure, either explicitly (`curves`) or as a
grid (`sweep`/`sweeps`). Results go to runs/<id>/.
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

FIGURES = Path(__file__).resolve().parent / "figures"


def expand_sweep(sw):
    base = sw.get("base", {})
    grid = {k: list(v) for k, v in (sw.get("grid") or {}).items()}
    runs = []
    for combo in itertools.product(*grid.values()) if grid else [()]:
        r = {"model": sw["model"], "method": base.get("method", "PACE"),
             "schedule": base.get("schedule", "const"), "seed": base.get("seed", 42),
             "optimizer": dict(base.get("optimizer", {}))}
        for k, v in zip(grid, combo):
            if k.startswith("optimizer."):
                r["optimizer"][k.split(".", 1)[1]] = v
            elif k in ("schedule", "seed", "model", "method"):
                r[k] = v
            else:
                r["optimizer"][k] = v
        runs.append(r)
    return runs


def expand_figure(fig):
    runs = list(fig.get("curves") or [])
    for sw in ([fig["sweep"]] if fig.get("sweep") else []) + list(fig.get("sweeps") or []):
        runs += expand_sweep(sw)
    return runs


def command(fig, run, extra):
    o = run["optimizer"]
    cmd = [sys.executable, "train.py", f"model={run['model']}",
           f"dataset={fig.get('dataset', 'smoltalk')}", f"optimizer={o['name']}",
           f"schedule={run['schedule']}", f"seed={run['seed']}", f"optimizer.lr={o['lr']}",
           f"optimizer.lambda_pullback={o.get('lambda_pullback', 0.0)}"]
    if o.get("ema_kappa") is not None:
        cmd.append(f"optimizer.ema_kappa={o['ema_kappa']}")
    if o.get("ema_update_freq") is not None:
        cmd.append(f"optimizer.ema_update_freq={o['ema_update_freq']}")
    return cmd + extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fig", help="figure id (a file in figures/), or 'list'")
    p.add_argument("--run", action="store_true", help="run the commands instead of printing them")
    p.add_argument("--data", help="prepared dataset dir (required with --run)")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--limit", type=int, help="only run the first N")
    a = p.parse_args()

    if a.fig == "list":
        for f in sorted(FIGURES.glob("*.yaml")):
            fig = yaml.safe_load(open(f))
            print(f"{f.stem:40} {fig.get('paper', ''):8} {fig.get('desc', '')}")
        return

    fig = yaml.safe_load(open(FIGURES / f"{a.fig}.yaml"))
    runs = expand_figure(fig)
    extra = [f"out_dir=runs/{a.fig}"]
    if a.data:
        extra.append(f"dataset.path={a.data}")
    if a.max_steps:
        extra.append(f"training.max_steps={a.max_steps}")
    if a.run and not a.data:
        p.error("--run needs --data")

    print(f"# {a.fig} ({fig.get('paper', '')}): {fig.get('desc', '')}, {len(runs)} runs")
    for i, r in enumerate(runs):
        cmd = command(fig, r, extra)
        print(" ".join(cmd[1:]).replace("train.py", "python train.py", 1))
        if a.run and (a.limit is None or i < a.limit):
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
