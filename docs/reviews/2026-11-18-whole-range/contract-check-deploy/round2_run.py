"""Run the draft-2 model with verification query output kept out of the log."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import round2_probe as probe


if __name__ == "__main__":
    for query in ("is-active", "is-enabled"):
        line = f"systemctl --user {query} bts-scheduler"
        assert probe.SHELL.count(line) == 1
        probe.SHELL = probe.SHELL.replace(line, line + " >/dev/null")
    with TemporaryDirectory(prefix="round2-replay-", dir=probe.HERE) as temp:
        work = Path(temp)
        print(json.dumps({"systemd_model": probe.systemd_cases(work),
                          "cron_writer": probe.replay_cron_writer(work)}, indent=2))
