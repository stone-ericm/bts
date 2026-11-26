"""Complete follow-up runs for round-1 scheduler/inputs probes and Carbon adversarial checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPO = Path(__file__).resolve().parents[2]
R1_DIR = REPO / "docs/reviews/2026-11-18-whole-range/review-fix-f3-codex"
OUT_DIR = Path(__file__).resolve().parent
BASE_SRC = Path("/tmp/wr-f3-r2-scratch").resolve() / "base" / "src"
F3_SRC = REPO / "src"


def child_scheduler_complete(base: bool) -> None:
    sys.path.insert(0, str(R1_DIR))
    import review_support  # type: ignore

    review_support.REPO = REPO
    import probe_scheduler  # type: ignore

    # Fix Codex's first-run expectation noted in review-fix-f3-codex/report.md Note 4:
    # K2d_next_day_replaces_match retains the 09:01 streak-1 archive as well as the D revocation.
    probe_scheduler.CASES["K2d_next_day_replaces_match"] = ("archived", "revocation")
    review_support.identify_package()
    with TemporaryDirectory() as temp:
        for case in probe_scheduler.CASES:
            probe_scheduler.run_case(Path(temp) / case, case, base)


def child_inputs_complete(base: bool) -> None:
    sys.path.insert(0, str(R1_DIR))
    import review_support  # type: ignore

    review_support.REPO = REPO
    import probe_inputs  # type: ignore

    review_support.identify_package()
    with TemporaryDirectory() as temp:
        root = Path(temp)
        probe_inputs.bad_files(root / "files", base)
        probe_inputs.timestamp_edges(root / "timestamps", base)
        # Fix Codex's first-run expectation noted in review-fix-f3-codex/report.md Note 4:
        # ACCEPTED_SCHEMAS is a tuple, so schema_version=[] returns None -> no_decision on both base and F3.
        changes = (
            {"streak": True}, {"streak": False}, {"streak": -1}, {"streak": "30"}, {"streak": []},
            {"streak": 30.0}, {"saver_available": 0}, {"saver_available": "false"}, {"saver_available": {}},
            {"saver_available": None}, {"scoreable": True}, {"date": "2027-04-01"}, {"date": [probe_inputs.DATE]},
            {"date": None}, {"scoreable": 0}, {"action": {}}, {"schema_version": []},
        )
        for index, change in enumerate(changes):
            cfg, picks, account = probe_inputs.setup(root / "skips" / str(index), probe_inputs.SKIP | change)
            for name in (probe_inputs.CURRENT, probe_inputs.ARCHIVE, probe_inputs.SWITCH):
                probe_inputs.put(account / name, probe_inputs.valid(name))
            result = probe_inputs.command(cfg, picks)
            if "action" in change:
                assert result == {"exit": 1, "stdout": "", "stderr": "", "sends": [], "exception": "TypeError"}
            elif change in ({"scoreable": 0}, {"schema_version": []}):
                probe_inputs.assert_out(result, "no_decision")
            else:
                probe_inputs.assert_out(result, "deliberate_skip")
            print(json.dumps({"skip_corruption": change, **result}))
        probe_inputs.enumeration(root / "enumeration", base)
        probe_inputs.standing(root / "standing", base)


def run_child(mode: str, base: bool) -> str:
    src = BASE_SRC if base else F3_SRC
    env = os.environ | {
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{src}:{REPO}:{R1_DIR}:{OUT_DIR}",
    }
    flag = "--base" if base else "--f3"
    res = subprocess.run(
        [sys.executable, __file__, f"--child-{mode}", flag],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=45, check=True,
    )
    first = res.stdout.splitlines()[0]
    assert Path(first.split("=", 1)[1]).resolve().is_relative_to(src.resolve()), first
    return res.stdout


def run_r1_followups() -> None:
    sched_base = run_child("scheduler", base=True)
    sched_f3 = run_child("scheduler", base=False)
    (OUT_DIR / "r1-scheduler-complete-base.log").write_text(sched_base)
    (OUT_DIR / "r1-scheduler-complete-f3.log").write_text(sched_f3)
    assert len(sched_base.splitlines()) == 27 and len(sched_f3.splitlines()) == 27
    print("r1 scheduler complete: 26/26 cases passed on base and F3.")

    in_base = run_child("inputs", base=True)
    in_f3 = run_child("inputs", base=False)
    (OUT_DIR / "r1-inputs-complete-base.log").write_text(in_base)
    (OUT_DIR / "r1-inputs-complete-f3.log").write_text(in_f3)

    edges = [json.loads(ln) for ln in in_f3.splitlines() if ln.startswith('{"edge":')]
    assert len(edges) == 7
    for row in edges[:6]:
        assert row["exit"] == 0 and json.loads(row["stdout"])["state"] == "deliberate_skip", row
    assert edges[6]["exit"] == 1 and json.loads(edges[6]["stdout"])["state"] == "skip_state_review", edges[6]

    skips_base = [ln for ln in in_base.splitlines() if ln.startswith('{"skip_corruption":')]
    skips_f3 = [ln for ln in in_f3.splitlines() if ln.startswith('{"skip_corruption":')]
    assert len(skips_base) == len(skips_f3) == 17 and skips_base == skips_f3
    print("r1 inputs complete: bad_files, 7 E1 timestamp edges, 17 malformed skips (base==F3), enumeration, standing all passed.")


def run_adversarial_complete() -> None:
    sys.path.insert(0, str(OUT_DIR))
    import probe_adversarial as adv  # type: ignore

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        iso_cases = [
            ("lowercase_z", f"{adv.DAY}T14:00:00z"),
            ("compact_iso", "20270402T140000Z"),
            ("space_sep", f"{adv.DAY} 14:00:00-04:00"),
            ("arbitrary_sep_with_Z", f"{adv.DAY}x14:00:00Z"),
            ("plus_as_date_time_sep_naive", f"{adv.DAY}+14:00"),
            ("plus_as_sep_with_Z", f"{adv.DAY}+14:00Z"),
            ("subminute_offset", f"{adv.DAY}T14:00:00.123456+05:45:30.123456"),
            ("date_only_Z", f"{adv.DAY}Z"),
        ]
        for label, stamp in iso_cases:
            sub_d = root / f"iso_d_{label}"
            (sub_d / "picks/account_state").mkdir(parents=True)
            (sub_d / "picks/account_state/contest_streak.manual.json").write_text(
                json.dumps(adv.make_dated(recorded_at=stamp))
            )
            res_d = adv.invoke_cli(sub_d, decision=adv.make_skip())

            sub_s = root / f"iso_s_{label}"
            (sub_s / "picks/account_state").mkdir(parents=True)
            (sub_s / "picks/account_state/season_contract.json").write_text(
                json.dumps(adv.make_switch(revoked_at=stamp))
            )
            res_s = adv.invoke_cli(sub_s, decision=adv.make_skip())

            assert res_d["exit"] in (0, 1) and res_s["exit"] in (0, 1)
            assert res_d["exit"] == res_s["exit"], (label, stamp, res_d, res_s)
            print(f"ISO grammar [{label}] {stamp!r}: exit={res_d['exit']} (dated & switch consistent)")

    adv.check_unhashable_and_hostile_fields()
    adv.check_dst_and_saver_boundaries()
    adv.check_filesystem_symlinks_and_partial_reads()
    adv.check_extra_parity_and_r64()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child-scheduler":
        child_scheduler_complete(sys.argv[2] == "--base")
    elif len(sys.argv) == 3 and sys.argv[1] == "--child-inputs":
        child_inputs_complete(sys.argv[2] == "--base")
    elif len(sys.argv) == 2 and sys.argv[1] == "r1_followups":
        run_r1_followups()
    elif len(sys.argv) == 2 and sys.argv[1] == "adversarial":
        run_adversarial_complete()
    else:
        raise SystemExit(f"unknown args: {sys.argv[1:]}")
