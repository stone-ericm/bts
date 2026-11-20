"""Item 3 probe (v2): byte-identical verification of same-day/no-today/future runs and all live/CLI/morning paths (BASE vs HEAD) on identical paths."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_3_live_and_controls_parity import BASE_SRC, HEAD_SRC, HERE, collect_snapshot, deny_network, seed_fixtures


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        logging.disable(logging.CRITICAL)
        fdir = Path(sys.argv[2])
        names = json.loads(sys.argv[3])
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            snap = collect_snapshot(fdir, names)
        print(json.dumps(snap, sort_keys=True))
        return

    signal.alarm(50)
    with TemporaryDirectory(dir=HERE) as tmp:
        shared_dir = Path(tmp) / "shared_fixtures"

        def run_side(src_path: Path) -> str:
            if shared_dir.exists():
                shutil.rmtree(shared_dir)
            names = seed_fixtures(shared_dir)
            env = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(src_path)}
            return subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "worker", str(shared_dir), json.dumps(names)],
                env=env, capture_output=True, text=True, timeout=30, check=True,
            ).stdout

        base_json = run_side(BASE_SRC)
        head_json = run_side(HEAD_SRC)
        base_sha = hashlib.sha256(base_json.encode("utf-8")).hexdigest()
        head_sha = hashlib.sha256(head_json.encode("utf-8")).hexdigest()
        assert base_json == head_json, "BASE vs HEAD mismatch!"
        assert base_sha == head_sha

        import bts.contest_state as cs
        import bts.saver_state as ss
        import bts.season_contract as sc
        for mod_name, fn_name in [
            ("contest_state", "load_decision_streak_state"),
            ("contest_state", "load_contest_streak_state"),
            ("saver_state", "load_saver_state"),
            ("season_contract", "switch_state"),
            ("season_contract", "live_automatic_refusal"),
        ]:
            mod = {"contest_state": cs, "saver_state": ss, "season_contract": sc}[mod_name]
            assert "historical" not in inspect.signature(getattr(mod, fn_name)).parameters
        print(f"[PASS] Item 3: BASE vs HEAD 100% byte-identical including body_sha256 (sha256={head_sha[:16]})")
    signal.alarm(0)


if __name__ == "__main__":
    main()
