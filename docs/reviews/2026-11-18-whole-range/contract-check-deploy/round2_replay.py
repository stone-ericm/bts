"""Run the fault model and give the archived slice fixtures the current repo root."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import round2_probe as probe


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cron"]:
        sys.path.insert(0, str(probe.REPO / "docs/reviews/2026-11-18-whole-range/slice-2-codex"))
        import offline_fixtures

        offline_fixtures.REPO = probe.REPO
        probe.cron_child(sys.argv[2], sys.argv[3:])
        sys.exit(0)

    for query in ("is-active", "is-enabled"):
        line = f"systemctl --user {query} bts-scheduler"
        assert probe.SHELL.count(line) == 1
        probe.SHELL = probe.SHELL.replace(line, line + " >/dev/null")
    with TemporaryDirectory(prefix="round2-replay-", dir=probe.HERE) as temp:
        work = Path(temp)
        model = probe.systemd_cases(work)
        probe.__file__ = __file__
        print(json.dumps({"systemd_model": model, "cron_writer": probe.replay_cron_writer(work)}, indent=2))
