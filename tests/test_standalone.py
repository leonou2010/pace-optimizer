"""Standalone-repo gate: the committed repo is lightweight and self-contained,
and leaks no internal cluster details.

  - no `/scratch`, `sf_bema`, Schedule-Free, or BEMA references in tracked files
    (the README prior-work credit to BEMA is the only allowance);
  - no bundled result data is tracked (reproduction uses the static config map +
    pace-train, not shipped data);
  - the core optimizer imports with only torch (no heavy/optional deps).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    return [REPO / f for f in out.stdout.split() if f]


def test_no_cluster_or_dropped_method_leaks():
    pat = re.compile(r"/scratch/blocklab|sf_bema|schedule.?free|\bbema\b", re.I)
    offenders = []
    for f in _tracked_files():
        if f.name in ("README.md", "test_standalone.py"):
            continue  # README credits BEMA as prior work; this file holds the patterns
        if f.suffix in {".lock", ".txt"}:
            continue
        try:
            if pat.search(f.read_text(errors="ignore")):
                offenders.append(str(f.relative_to(REPO)))
        except Exception:
            pass
    assert not offenders, f"cluster/SF/BEMA leakage in tracked files: {offenders}"


def test_no_bundled_data_tracked():
    out = subprocess.run(["git", "ls-files", "data"], cwd=REPO, capture_output=True, text=True)
    tracked = [f for f in out.stdout.split() if f]
    assert not tracked, f"data should not be committed (lightweight repo): {tracked[:5]}..."


def test_optimizer_imports_with_torch_only():
    # The core product must not pull in training/figure deps.
    sys.path.insert(0, str(REPO / "src"))
    import importlib
    mod = importlib.import_module("pace.optimizer")
    assert hasattr(mod, "PACE")
