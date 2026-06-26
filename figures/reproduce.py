"""Reproduce a paper figure from bundled result data.

Usage (after ``uv sync``)::

    uv run pace-figure fig=fig1
    uv run pace-figure fig=fig1 fig.models=[SmolLM2-1.7B]

Or directly::

    python figures/reproduce.py fig=fig1

Each figure ``<id>`` has a Hydra config at ``conf/fig/<id>.yaml`` describing its
inputs, and a builder module ``_figs/<id>.py`` exposing ``build(cfg) -> dict``.
The builder reads the curated JSONs under ``data/results/`` (override the data
location with the ``PACE_DATA`` env var) and writes the PDF/PNG under
``figures/rendered`` (override with ``PACE_FIG_OUT``).
"""

import importlib
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Make `plot_utils` and the `_figs` package importable whether this file is run
# directly or via the installed `pace-figure` console script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


@hydra.main(version_base=None, config_path="conf", config_name="reproduce")
def main(cfg: DictConfig):
    fig_id = cfg.fig.id
    builder = importlib.import_module(f"_figs.{fig_id}")
    print(f"[pace-figure] reproducing {fig_id}")
    result = builder.build(cfg.fig)
    print("\n--- result ---")
    print(OmegaConf.to_yaml({"config": OmegaConf.to_container(cfg.fig, resolve=True),
                             "output": result}))
    return result


if __name__ == "__main__":
    main()
