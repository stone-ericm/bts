"""Probe 3d: Run probe_3c with zero-arg _load_mdp() and analyze full parser/selection/health matrix."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bts.strategy as strategy_mod  # noqa: E402

_real_load = strategy_mod._load_mdp
strategy_mod._load_mdp = lambda *a: _real_load()

from probe_3c_sequences import run_all  # noqa: E402
from probe_3_parser_ledger_sequences import analyze_all  # noqa: E402

if __name__ == "__main__":
    m_res = run_all()
    analyze_all(m_res)
