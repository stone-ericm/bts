"""Run probe_callers_v4 with JSON default serializer for SimpleNamespace truthiness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

_orig_dumps = json.dumps
json.dumps = lambda obj, *a, **kw: _orig_dumps(obj, *a, **({"default": bool} | kw))

import probe_callers_v4

if __name__ == "__main__":
    if "--worker" in sys.argv:
        idx_s = sys.argv.index("--src")
        idx_o = sys.argv.index("--out")
        probe_callers_v4.run_worker(Path(sys.argv[idx_s + 1]), Path(sys.argv[idx_o + 1]))
    else:
        import subprocess
        repo = Path(__file__).resolve().parents[2]
        work = Path(__file__).resolve().parent
        base_p = work / "callers_base.json"
        branch_p = work / "callers_branch.json"
        me = Path(__file__).resolve()
        for src, out_p in [(work / "base/src", base_p), (repo / "src", branch_p)]:
            proc = subprocess.run(
                [sys.executable, str(me), "--worker", "--src", str(src), "--out", str(out_p)],
                capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"worker failed ({src}):\n{proc.stdout}\n{proc.stderr}")
        base_c = json.loads(base_p.read_text(encoding="utf-8"))
        branch_c = json.loads(branch_p.read_text(encoding="utf-8"))
        assert base_c["denied_network"] == [] and branch_c["denied_network"] == []
        assert base_c["run_and_pick_uncovered"] == {"accepted": True, "action": "double"}
        assert branch_c["run_and_pick_uncovered"]["accepted"] is False
        assert base_c["orchestrate_initial_uncovered"] == {"returned": True, "pick_saved": True, "dms_count": 0, "posts_count": 1}
        assert branch_c["orchestrate_initial_uncovered"] == {"returned": False, "pick_saved": False, "dms_count": 1, "posts_count": 0}
        for sub in ("post_save", "skip", "locked_catchup"):
            assert branch_c[f"orchestrate_mid_revoke_{sub}"]["raised"] is True
            assert branch_c[f"orchestrate_mid_revoke_{sub}"]["posts"] == 0
        assert base_c["scheduler_run_single_check_uncovered"]["pick_file_exists"] is True
        assert branch_c["scheduler_run_single_check_uncovered"] == {
            "should_post": False,
            "pick_result_none": True,
            "selection_none": True,
            "pick_file_exists": False,
            "decision_exists": False,
            "alert_dms": 1,
        }
        assert base_c["deliver_and_lock_dm"]["returned"] is True and base_c["deliver_and_lock_dm"]["pick_dms"] == 1
        assert branch_c["deliver_and_lock_dm"]["returned"] is False and branch_c["deliver_and_lock_dm"]["pick_dms"] == 0
        assert base_c["deliver_and_lock_public"]["returned"] is True and base_c["deliver_and_lock_public"]["posts"] == 1
        assert branch_c["deliver_and_lock_public"]["returned"] is False and branch_c["deliver_and_lock_public"]["posts"] == 0
        assert branch_c["deliver_and_lock_private"]["returned"] is True
        assert branch_c["deliver_and_lock_delivered"]["returned"] is True
        assert branch_c["deliver_and_lock_crash"]["returned"] is False and branch_c["deliver_and_lock_crash"]["pick_locked"] is True
        assert base_c["fallback_refresh_uncovered"]["raised"] is False
        assert branch_c["fallback_refresh_uncovered"]["raised"] is True
        assert branch_c["non_blocked_grading_skip_day_outcome"]["check_results_exit"] == 0
        assert branch_c["non_blocked_grading_skip_day_outcome"]["streak_after_check_results"] == 1
        assert branch_c["non_blocked_grading_skip_day_outcome"]["poll_status"] == "final"
        assert branch_c["non_blocked_grading_skip_day_outcome"]["streak_after_poll"] == 1
        assert branch_c["non_blocked_grading_skip_day_outcome"]["eod_skip_written"] is True
        assert branch_c["non_blocked_grading_skip_day_outcome"]["day_outcome_exit"] == 0
        print(json.dumps({"base": base_c, "branch": branch_c}, indent=2))
