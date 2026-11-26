"""Route the lineup fixture through the real live loader and a distinct error."""

from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from bts import contest_state, orchestrator, scheduler

import probe_dedup


def read_contest(config, *_args, **_kwargs):
    root = Path(config["orchestrator"]["picks_dir"])
    manual = root / "account_state/contest_streak.manual.json"
    manual.write_text("{")
    try:
        contest_state.load_decision_streak_state(root, require_contest_state=True)
    finally:
        manual.unlink()
    raise AssertionError("expected a live refusal from the malformed manual observation")


if __name__ == "__main__":
    with patch.object(orchestrator, "run_and_pick", read_contest), patch.object(
            scheduler, "heartbeat_watchdog", lambda *_a, **_kw: nullcontext()):
        probe_dedup.run()
