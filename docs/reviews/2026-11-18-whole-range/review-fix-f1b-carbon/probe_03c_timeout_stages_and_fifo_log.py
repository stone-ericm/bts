"""Probe 03c: Clean standalone runner for timeout-stage preservation, stated loss case, and consecutive FIFO hangs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

OUT_DIR = Path(__file__).resolve().parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

from probe_03_class_hunt_and_standing import (
    _kill_pids,
    check_stated_first_observation_timeout_loss_and_trickle,
    hunt_timeout_after_recorded_observation,
    REPO,
    FLOCK_PATH,
    LOCK_DEFAULT,
    TIMEOUT_PATH,
    PITCH,
    DAY,
)


def check_two_consecutive_fifo_hangs(pids: set[int]) -> dict:
    import os
    import re
    import shlex
    import subprocess
    import tempfile
    sys.path.insert(0, str(REPO))
    from probe_02_k1_to_k5_and_wrapper import generate_enter_cron_line
    from tests.test_wr_f1_wrong_game import WRONG, World

    p03 = OUT_DIR / "probe_03_class_hunt_and_standing.py"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        line, _, _ = generate_enter_cron_line(root / "inst")
        shell_tmpl = line.split(maxsplit=5)[5]
        flock_cmd = [sys.executable, str(p03), "--flock"]
        base_env = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}", "TZ": "America/New_York"}

        def make_cmd(lock_p: Path, inner: list[str], dur: str = "0.7") -> str:
            s = (
                shell_tmpl
                .replace(FLOCK_PATH, shlex.join(flock_cmd), 1)
                .replace(LOCK_DEFAULT, shlex.quote(str(lock_p)), 1)
                .replace(f"{TIMEOUT_PATH} --verbose -k 30 840", f"{TIMEOUT_PATH} --verbose -k 0.2 {dur}", 1)
            )
            return re.sub(r"\S+/uv run bts check-pick-entered.*?(?=>>)", lambda _: shlex.join(inner) + " ", s, count=1)

        w_fifo = World(root / "fifo_world")
        w_fifo.commit(pitch=PITCH)
        fifo_path = w_fifo.units / f"{DAY}T210000Z.json.gz"
        os.mkfifo(fifo_path)
        lock1 = root / "fifo.lock"

        for slot_name in ("s1", "s2"):
            p = subprocess.Popen(
                ["/bin/sh", "-c", make_cmd(lock1, [sys.executable, str(p03), "--child", "hang_fifo_or_sleep", str(w_fifo.root), slot_name], dur="0.7")],
                env=base_env,
            )
            pids.add(p.pid)
            rc = p.wait(timeout=6.0)
            if (w_fifo.root / f"slot_{slot_name}_pid").exists():
                pids.add(int((w_fifo.root / f"slot_{slot_name}_pid").read_text()))
            assert rc == 124
            assert not (w_fifo.root / f"slot_{slot_name}_done").exists()

        fifo_path.unlink()
        w_fifo.capture([WRONG])
        p3 = subprocess.Popen(
            ["/bin/sh", "-c", make_cmd(lock1, [sys.executable, str(p03), "--child", "hang_fifo_or_sleep", str(w_fifo.root), "s3"], dur="5.0")],
            env=base_env,
        )
        pids.add(p3.pid)
        assert p3.wait(timeout=8.0) == 1
        assert (w_fifo.root / "slot_s3_done").read_text() == "1"
        assert w_fifo.read_marker()["game_unresolved"] is True

    return {"two_consecutive_hung_runs_recover_on_slot_3": True}


def run_all() -> dict:
    pids: set[int] = set()
    try:
        summary = {
            "timeout_after_prior_observation": hunt_timeout_after_recorded_observation(pids),
            "first_observation_loss_and_trickle": check_stated_first_observation_timeout_loss_and_trickle(pids),
            "consecutive_fifo_hangs": check_two_consecutive_fifo_hangs(pids),
        }
    finally:
        _kill_pids(pids)
    (OUT_DIR / "log_03c_timeout_stages_and_fifo.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_03c():
    res = run_all()
    assert len(res["timeout_after_prior_observation"]["stages_preserving_prior_observation"]) == 7


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
