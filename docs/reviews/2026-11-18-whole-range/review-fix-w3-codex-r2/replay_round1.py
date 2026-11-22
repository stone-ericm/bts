"""Replay both reviewers' original inputs without changing their scripts."""

import importlib.util
import json
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = HERE.parents[2]
CARBON = PARENT / "main-docs/docs/reviews/2026-11-18-whole-range/review-fix-w3-carbon"
sys.path.insert(0, str(PARENT))

import review_world

review_world.fixture.HERE = HERE


def carbon():
    from bts import picks

    for script, cases in {
        "probe_8b_verify_59b72c04.py": ["void_deliver_past_cutoff", "void_defer_crossing_cutoff"],
        "probe_6_real_mdp_controls.py": ["void_deliver_past_cutoff_check", "void_defer_crossing_cutoff_check"],
    }.items():
        spec = importlib.util.spec_from_file_location("carbon", CARBON / script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.REPO = ROOT
        module.REVIEW_DIR = HERE
        module.TARGET_SRC = Path(picks.__file__).resolve().parents[1]
        for case in cases:
            result = module.worker_main(case)
            print("CARBON", case, json.dumps(result), flush=True)
            assert result["refusal_reasons"] == ["game_not_playable"]
            assert result["postponed_alerts"] == 1 and not result["late_alerts"]
            assert not result.get("names_batter")


def run(case):
    if case == "r1-durable":
        import probe_durable as probe

        probe.EXPECT_ALERT = True
        for name in ["locked-unconfirmed", "attempt-invalid-action", "attempt-deliver", "attempt-defer",
                     "delivered", "scoreable", "mismatched"]:
            probe.during_poll(name)
    elif case == "r1-gap":
        import probe_gap_window as probe

        probe.EXPECT_DELIVERY_GUARDS = True
        probe.run(False)
        probe.run(True)
    elif case == "r1-clear":
        import probe_delivery as probe

        probe.EXPECT_CLEAR_ERROR_ALERT = True
        probe.cutoff_clear_error("dm")
        probe.cutoff_clear_error("public")
    elif case == "carbon":
        carbon()
    else:
        sys.argv = [str(PARENT / case), *sys.argv[2:]]
        runpy.run_path(str(PARENT / case), run_name="__main__")


if __name__ == "__main__":
    run(sys.argv[1])
