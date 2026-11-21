#!/usr/bin/env python3
"""Item 4 standalone probe v2: passing transcript comparison (legacy absent & present), stamp fsyncs, and commit ordering."""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / ".codex-review/wr-d3"))

import probe_workflow_paths_and_signals_v2 as p2


def main() -> None:
    deploy, _ = p2.orig.extract_scripts(p2.orig.WORKFLOW)
    base_deploy, = p2.orig.extract_scripts(p2.orig.BASE_WORKFLOW)
    assert "f.flush()\n                os.fsync(f.fileno())" in deploy or "os.fsync(f.fileno())" in deploy

    for leg in (False, True):
        scen = "legacy-present" if leg else "active"
        with tempfile.TemporaryDirectory() as td:
            box_b = p2.orig.make_box(Path(td))
            if leg:
                (box_b / "etc/systemd/system").mkdir(parents=True)
                (box_b / "etc/systemd/system/bts-scheduler.service").touch()
                (box_b / "etc/systemd/system/bts-dashboard.service").touch()
            r_base, calls_base = p2.run_script_60(box_b, base_deploy, scenario=scen)

        with tempfile.TemporaryDirectory() as td:
            box_c = p2.orig.make_box(Path(td))
            if leg:
                (box_c / "etc/systemd/system").mkdir(parents=True)
                (box_c / "etc/systemd/system/bts-scheduler.service").touch()
                (box_c / "etc/systemd/system/bts-dashboard.service").touch()
            r_cand, calls_cand = p2.run_script_60(box_c, deploy, scenario=scen)
            fsyncs = (box_c / "fsyncs").read_text().splitlines()
            stamp_val = (box_c / "home/projects/bts/data/.last_deploy_iso").read_text().strip()
            rec_val = (box_c / "run/bts-deploy/attempt").read_text().strip()

        assert r_base.returncode == 0 and r_cand.returncode == 0
        idx_begun = next(i for i, c in enumerate(calls_cand) if c[0] == "mv" and str(box_c / "run/bts-deploy/attempt") in c)
        idx_disable = calls_cand.index(["systemctl", "--user", "disable", "bts-scheduler"])
        idx_reset = calls_cand.index(["git", "reset", "--hard", p2.orig.TARGET])
        idx_sync = calls_cand.index(["uv", "sync", "--locked", "--extra", "model"])
        idx_restart = calls_cand.index(["systemctl", "--user", "restart", "bts-scheduler", "bts-dashboard"])
        idx_wait = calls_cand.index(["timeout", "-k", "2", "33", "sleep", "30"])
        idx_canary = next(i for i, c in enumerate(calls_cand) if c[0] == "bts" and "deploy-canary" in c)
        idx_stage = next(i for i, c in enumerate(calls_cand) if c[0] == "python3" and len(c) > 3 and c[3] == "stage")
        idx_enable = calls_cand.index(["systemctl", "--user", "enable", "bts-scheduler"])
        idx_stamp = next(i for i, c in enumerate(calls_cand) if c[0] == "python3" and len(c) > 3 and c[3] == "stamp")
        idx_commit = [i for i, c in enumerate(calls_cand) if c[0] == "mv" and str(box_c / "run/bts-deploy/attempt") in c][-1]
        idx_pass = next(i for i, c in enumerate(calls_cand) if c[0] == "bash" and any("CANARY PASSED" in a for a in c))

        chain = [idx_begun, idx_disable, idx_reset, idx_sync, idx_restart, idx_wait, idx_canary, idx_stage, idx_enable, idx_stamp, idx_commit, idx_pass]
        assert chain == sorted(chain) and len(set(chain)) == len(chain), chain
        assert fsyncs == ["dir"], fsyncs
        assert " committed - - - -" in rec_val

        opcodes = list(difflib.SequenceMatcher(
            a=[json.dumps(c) for c in calls_base],
            b=[json.dumps(c) for c in calls_cand],
            autojunk=False,
        ).get_opcodes())
        print(
            f"Item 4 (legacy={leg}): base={len(calls_base)} calls -> cand={len(calls_cand)} calls "
            f"({len(opcodes)} diff spans), chain={chain}, dir_fsync={fsyncs}, stamp={stamp_val}"
        )
        print("  cand stdout:", json.dumps(r_cand.stdout.splitlines()))


if __name__ == "__main__":
    main()
