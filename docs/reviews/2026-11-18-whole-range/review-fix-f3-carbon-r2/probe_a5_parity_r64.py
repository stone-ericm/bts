"""Run extra base-vs-F3 parity edge cases and R64 operator-text audit."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_SRC = Path("/tmp/wr-f3-r2-scratch").resolve() / "base" / "src"
F3_SRC = REPO / "src"


def parity_child() -> None:
    import bts
    from tests.test_day_outcome import _decision
    sys.path.insert(0, str(OUT_DIR))
    import probe_adversarial as adv  # type: ignore

    print(f"bts.__file__={bts.__file__}")
    results = []
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases = [
            ("absent_picks_dir", dict(day=adv.DAY, decision=None, create_picks=False)),
            ("bad_date_no_decision", dict(day="not-a-date", decision=None)),
            ("bad_date_skip", dict(day="not-a-date", decision=adv.make_skip("not-a-date"))),
            ("leap_invalid_skip", dict(day="2027-02-29", decision=adv.make_skip("2027-02-29"))),
            ("year1_no_decision", dict(day="0001-01-01", decision=None)),
            ("year9999_no_games", dict(day="9999-12-31", decision=adv.make_skip("9999-12-31"), schedule=())),
            ("scheduler_list_config", dict(day=adv.DAY, decision=adv.make_skip(), raw_config="scheduler = []\n")),
            ("invalid_utf8_decision", dict(day=adv.DAY, decision=b"\xff\xfe")),
            ("single_with_runner_up", dict(
                day=adv.DAY,
                decision=_decision("single", double={"batter_name": "Runner Up"}),
            )),
        ]
        for name, kw in cases:
            sub = root / name
            sub.mkdir(parents=True, exist_ok=True)
            if kw.get("create_picks", True):
                acct = sub / "picks/account_state"
                acct.mkdir(parents=True, exist_ok=True)
                (acct / "contest_streak.manual.json").write_text(json.dumps(adv.make_dated()))
                (acct / "season_contract.json").write_text(json.dumps(adv.make_switch()))
            res = adv.invoke_cli(sub, **kw)
            results.append({"case": name, **res})
    print(json.dumps(results, sort_keys=True))


def main() -> None:
    sys.path.insert(0, str(OUT_DIR))
    import probe_adversarial as adv  # type: ignore

    outs = {}
    for label, src in [("base", BASE_SRC), ("f3", F3_SRC)]:
        env = os.environ | {
            "TZ": "America/New_York",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{src}:{REPO}:{OUT_DIR}",
        }
        proc = subprocess.run(
            [sys.executable, __file__, "--parity-child"],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=30, check=True,
        )
        lines = proc.stdout.splitlines()
        assert Path(lines[0].split("=", 1)[1]).resolve().is_relative_to(src.resolve())
        outs[label] = (lines[0], lines[1])

    print(f"base: {outs['base'][0]}")
    print(f"f3:   {outs['f3'][0]}")
    assert outs["base"][1] == outs["f3"][1], "extra parity cases differed between base and F3!"
    parsed = json.loads(outs["f3"][1])
    print(f"A5 parity: {len(parsed)}/{len(parsed)} tricky cases matched byte for byte.")

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        shapes = {
            "installed_only": {"contest_streak.manual.json": adv.make_dated(streak=0, saver="not_earned")},
            "archive_only": {"contest_streak.manual.revoked-1.json": adv.make_dated(streak=1, saver="used")},
            "revoke_only": {"season_contract.json": adv.make_switch()},
            "installed_and_revoke": {
                "contest_streak.manual.json": adv.make_dated(streak=12, saver="active"),
                "season_contract.json": adv.make_switch(),
            },
            "archive_and_revoke": {
                "contest_streak.manual.revoked-1.json": adv.make_dated(streak=12, saver="used"),
                "season_contract.json": adv.make_switch(),
            },
        }
        forbidden_tokens = ("B8a", "DIR", "set-contest-streak", "revoke --season", "restart", "Nothing needs doing")
        for name, files in shapes.items():
            sub = root / name
            acct = sub / "picks/account_state"
            acct.mkdir(parents=True)
            for fn, doc in files.items():
                (acct / fn).write_text(json.dumps(doc))
            res = adv.invoke_cli(sub, decision=adv.make_skip(streak=30, saver=False))
            assert res["exit"] == 1 and len(res["sends"]) == 1
            dm_text = res["sends"][0][1]
            assert dm_text.startswith(f"BTS day check {adv.DAY}: SKIP STATE REVIEW. ")
            assert dm_text.endswith("Check MLB for unsettled entries and verify the current streak, season best and saver.")
            for tok in forbidden_tokens:
                assert tok not in dm_text, (name, tok, dm_text)
            print(f"R64 [{name}]: {dm_text}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--parity-child":
        parity_child()
    else:
        main()
