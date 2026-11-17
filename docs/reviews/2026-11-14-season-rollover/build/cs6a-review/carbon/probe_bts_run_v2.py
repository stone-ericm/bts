"""Adversarial Probe 2 (v2): Exhaustive `bts run` (rows 6 and 7) Path × Contest-State matrix
comparing 09d9b51f (base) vs cad7c5ec (branch) with real `load_decision_streak_state` on pinned files.
"""
from __future__ import annotations

import datetime as dt
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO / "src"

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)
EOD = datetime.combine(dt.date(2027, 7, 15), dt.time.max, ET)
ACT_ID = "a" * 32

BLOCKED_AUDIT_EVENTS = frozenset({
    "socket.connect",
    "socket.connect_ex",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
    "socket.gethostbyaddr",
    "socket.getnameinfo",
    "socket.sendto",
    "urllib.Request",
})


def install_network_audit_hook() -> None:
    def _hook(event: str, args: tuple) -> None:
        if event in BLOCKED_AUDIT_EVENTS:
            raise RuntimeError(f"Network/DNS forbidden by review audit hook: {event} {args!r}")

    sys.addaudithook(_hook)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.replace(tzinfo=None) if tz is None else NOW.astimezone(tz)


def populate_state(picks_dir: Path, state_kind: str) -> None:
    acc = picks_dir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    (picks_dir / "streak.json").write_text(json.dumps({
        "streak": 4,
        "saver_available": True,
        "season": 2027,
    }))

    if state_kind == "none":
        return
    if state_kind == "last_season_auto_v1":
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": 7,
            "source": "mlb_bts_profile",
            "source_date": "2026-09-28",
            "recorded_at": "2026-09-28T16:00:00Z",
        }))
        return
    if state_kind == "legacy_manual":
        (acc / "contest_streak.manual.json").write_text(json.dumps({
            "active_streak": 7,
            "best_streak": 9,
            "source": "manual_screenshot",
            "source_date": DAY,
            "override_expires_at": "2027-07-16T16:00:00Z",
        }))
        return
    if state_kind == "dated_in_effect":
        (acc / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_manual_v3",
            "effective_date": DAY,
            "season": 2027,
            "active_streak": 7,
            "best_streak": 9,
            "saver_state": "used",
            "source": "manual_cli",
            "recorded_at": NOW.isoformat(),
            "expires_at": EOD.isoformat(),
        }))
        # Corrupt switch beside a dated observation in effect -> never read, accepted cleanly!
        (acc / "season_contract.json").write_text("{corrupt switch")
        return
    if state_kind == "covered_auto_v2":
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": ACT_ID,
            "switched_at": "2027-04-10T15:00:00Z",
            "revoked_at": None,
            "note": "verified",
        }))
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ACT_ID,
            "active_streak": 7,
            "best_streak": 9,
            "source": "mlb_bts_profile",
            "source_date": DAY,
            "recorded_at": "2027-07-15T14:00:00Z",
        }))
        return
    if state_kind == "corrupt_switch":
        (acc / "season_contract.json").write_text("{not valid json")
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ACT_ID,
            "active_streak": 7,
            "best_streak": 9,
            "source": "mlb_bts_profile",
            "source_date": DAY,
            "recorded_at": "2027-07-15T14:00:00Z",
        }))
        return
    if state_kind == "undated_auto_v2":
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": ACT_ID,
            "switched_at": "2027-04-10T15:00:00Z",
            "revoked_at": None,
            "note": "verified",
        }))
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ACT_ID,
            "active_streak": 7,
            "best_streak": 9,
            "source": "mlb_bts_profile",
            "source_date": None,
            "recorded_at": None,
        }))
        return
    if state_kind == "corrupt_manual_json":
        (acc / "contest_streak.manual.json").write_text("{corrupt manual")
        return
    raise ValueError(f"Unknown state_kind: {state_kind}")


PATHS = (
    "skip",
    "locked_unposted",
    "locked_posted",
    "fresh_single",
    "fresh_double",
    "fresh_single_no_post_window",
    "empty_predictions",
    "pipeline_runtime_error",
    "pipeline_generic_error",
    "dry_run",
)

STATES = (
    "none",
    "last_season_auto_v1",
    "legacy_manual",
    "dated_in_effect",
    "covered_auto_v2",
    "corrupt_switch",
    "undated_auto_v2",
    "corrupt_manual_json",
)


def run_worker(which: str, out_file: Path) -> None:
    install_network_audit_hook()
    target_src = BASE_SRC if which == "base" else BRANCH_SRC
    cleaned = [
        p for p in sys.path
        if Path(p if p else ".").resolve() != BRANCH_SRC.resolve()
    ]
    sys.path[:] = [str(target_src)] + cleaned

    import bts
    assert Path(bts.__file__).resolve().is_relative_to(target_src.resolve())

    import pandas as pd
    from click.testing import CliRunner
    from bts import cli, contest_state, picks, strategy
    from bts.model import predict

    def make_df() -> pd.DataFrame:
        return pd.DataFrame([
            {
                "batter_name": "Jacob Wilson",
                "batter_id": 700363,
                "team": "ATH",
                "lineup": 1,
                "pitcher_name": "Jose Suarez",
                "pitcher_id": 660761,
                "p_hit_pa": 0.312,
                "p_game_hit": 0.812,
                "flags": "",
                "game_pk": 778899,
                "game_time": f"{DAY}T23:10:00Z",
            },
            {
                "batter_name": "Bobby Witt Jr.",
                "batter_id": 677951,
                "team": "KC",
                "lineup": 2,
                "pitcher_name": "Tarik Skubal",
                "pitcher_id": 669373,
                "p_hit_pa": 0.295,
                "p_game_hit": 0.790,
                "flags": "",
                "game_pk": 778900,
                "game_time": f"{DAY}T23:10:00Z",
            },
        ])

    def make_pick(name: str, bid: int, team: str, p: float, gpk: int) -> picks.Pick:
        return picks.Pick(
            batter_name=name,
            batter_id=bid,
            team=team,
            lineup_position=1,
            pitcher_name="Jose Suarez",
            pitcher_id=660761,
            p_game_hit=p,
            flags=[],
            projected_lineup=False,
            game_pk=gpk,
            game_time=f"{DAY}T23:10:00Z",
        )

    runner = CliRunner()
    records: dict[str, dict] = {}

    def execute_one(path_kind: str, state_kind: str, revoke_before_second_read: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            picks_dir = root / "picks"
            models_dir = root / "models"
            populate_state(picks_dir, state_kind)

            posts: list[str] = []
            saves: list[dict] = []
            loader_calls: list[dict] = []

            real_loader = contest_state.load_decision_streak_state
            real_save = picks.save_pick

            def wrapped_loader(pdir, **kwargs):
                loader_calls.append({"dir": str(pdir), "kwargs": dict(kwargs)})
                if revoke_before_second_read and len(loader_calls) == 2:
                    sc = Path(pdir) / "account_state" / "season_contract.json"
                    if sc.exists():
                        sc.unlink()
                return real_loader(pdir, **kwargs)

            def wrapped_save(daily_obj, pdir):
                res = real_save(daily_obj, pdir)
                saves.append({
                    "bluesky_posted": daily_obj.bluesky_posted,
                    "bluesky_uri": daily_obj.bluesky_uri,
                    "batter": daily_obj.pick.batter_name,
                    "double": daily_obj.double_down.batter_name if daily_obj.double_down else None,
                })
                return res

            def fake_pipeline(*_a, **_kw):
                if path_kind == "pipeline_runtime_error":
                    raise RuntimeError("No lineup data")
                if path_kind == "pipeline_generic_error":
                    raise ValueError("boom")
                if path_kind == "empty_predictions":
                    return make_df().iloc[:0]
                return make_df()

            def fake_select(_preds, _date, _pdir, **kwargs):
                primary = make_pick("Jacob Wilson", 700363, "ATH", 0.812, 778899)
                second = make_pick("Bobby Witt Jr.", 677951, "KC", 0.790, 778900)
                if path_kind == "skip":
                    return SimpleNamespace(
                        pick_result=None,
                        decision=SimpleNamespace(reason="below threshold", streak=kwargs["streak"]),
                    )
                if path_kind == "locked_unposted":
                    daily = picks.DailyPick(
                        date=DAY, run_time=NOW.isoformat(), pick=primary,
                        double_down=None, runner_up=second, bluesky_posted=False,
                    )
                    return SimpleNamespace(pick_result=strategy.PickResult(daily=daily, locked=True), decision=None)
                if path_kind == "locked_posted":
                    daily = picks.DailyPick(
                        date=DAY, run_time=NOW.isoformat(), pick=primary,
                        double_down=None, runner_up=second, bluesky_posted=True, bluesky_uri="at://prior/1",
                    )
                    return SimpleNamespace(pick_result=strategy.PickResult(daily=daily, locked=True), decision=None)
                if path_kind in ("fresh_single", "fresh_single_no_post_window"):
                    daily = picks.DailyPick(
                        date=DAY, run_time=NOW.isoformat(), pick=primary,
                        double_down=None, runner_up=second, bluesky_posted=False,
                    )
                    return SimpleNamespace(pick_result=strategy.PickResult(daily=daily, locked=False), decision=None)
                if path_kind == "fresh_double":
                    daily = picks.DailyPick(
                        date=DAY, run_time=NOW.isoformat(), pick=primary,
                        double_down=second, runner_up=None, bluesky_posted=False,
                    )
                    return SimpleNamespace(pick_result=strategy.PickResult(daily=daily, locked=False), decision=None)
                raise AssertionError(f"select_pick should not run on {path_kind}")

            should_post = path_kind != "fresh_single_no_post_window"

            args = [
                "run", "--date", DAY,
                "--picks-dir", str(picks_dir),
                "--models-dir", str(models_dir),
                "--data-dir", str(root / "data"),
            ]
            if path_kind == "dry_run":
                args.append("--dry-run")

            with (
                patch("datetime.datetime", FixedDateTime),
                patch.object(contest_state, "datetime", FixedDateTime),
                patch.object(picks, "datetime", FixedDateTime),
                patch.object(strategy, "datetime", FixedDateTime),
                patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40),
                patch.object(predict, "run_pipeline", side_effect=fake_pipeline),
                patch.object(picks, "get_game_statuses_detailed", return_value={}),
                patch.object(strategy, "select_pick", side_effect=fake_select),
                patch.object(contest_state, "load_decision_streak_state", side_effect=wrapped_loader),
                patch.object(picks, "save_pick", side_effect=wrapped_save),
                patch("bts.posting.should_post_now", return_value=should_post),
                patch("bts.posting.post_to_bluesky", side_effect=lambda txt: posts.append(txt) or "at://test/post/99"),
            ):
                res = runner.invoke(cli.cli, args)

            pick_file = picks_dir / f"{DAY}.json"
            evo_file = picks_dir / f"lineup_evolution_{DAY}.jsonl"
            norm = lambda s: s.replace(str(picks_dir), "<PICKS_DIR>").replace(str(models_dir), "<MODELS_DIR>")
            return {
                "exit": res.exit_code,
                "exc_type": type(res.exception).__name__ if res.exception else None,
                "stdout": norm(res.stdout),
                "stderr": norm(res.stderr),
                "posts": posts,
                "saves": saves,
                "loader_calls": [{"kwargs": c["kwargs"]} for c in loader_calls],
                "pick_json": json.loads(norm(pick_file.read_text())) if pick_file.exists() else None,
                "evo_lines": [json.loads(norm(ln)) for ln in evo_file.read_text().splitlines()] if evo_file.exists() else None,
            }

    for p, s in itertools.product(PATHS, STATES):
        records[f"{p}__{s}"] = execute_one(p, s)

    for p in ("locked_unposted", "fresh_single", "fresh_double", "fresh_single_no_post_window"):
        records[f"mid_revoke__{p}__covered_auto_v2"] = execute_one(p, "covered_auto_v2", revoke_before_second_read=True)

    out_file.write_text(json.dumps(records, indent=2))


def verify_results(base: dict[str, dict], branch: dict[str, dict]) -> dict:
    EARLY_EXIT_PATHS = {"dry_run", "empty_predictions", "pipeline_runtime_error", "pipeline_generic_error"}
    ACCEPTED_STATES = {"dated_in_effect", "covered_auto_v2"}
    REFUSED_LIVE_STATES = {"none", "last_season_auto_v1", "legacy_manual", "corrupt_switch", "undated_auto_v2"}

    errors: list[dict] = []
    checked = 0

    for key, b1 in branch.items():
        checked += 1
        b0 = base[key]
        if key.startswith("mid_revoke__"):
            _, p, _ = key.split("__")
            assert b0["exit"] == 0, (key, b0)
            if b1["exit"] != 1 or not b1["stderr"].startswith("bts run: no pick: ") or b1["posts"] != []:
                errors.append({"key": key, "reason": "mid_revoke_wrong_exit_or_post", "branch": b1})
            if p == "locked_unposted":
                if b1["saves"] != [] or b1["pick_json"] is not None or b1["evo_lines"] is not None:
                    errors.append({"key": key, "reason": "locked_unposted_saved_after_refusal", "branch": b1})
                if not b0["stdout"].startswith(b1["stdout"]) or "Pick locked: Jacob Wilson (game started)" not in b1["stdout"]:
                    errors.append({"key": key, "reason": "locked_stdout_mismatch", "base": b0["stdout"], "branch": b1["stdout"]})
            else:
                if len(b1["saves"]) != 1 or b1["saves"][0]["bluesky_posted"] is not False:
                    errors.append({"key": key, "reason": "fresh_read3_save_count", "branch": b1})
                if b1["pick_json"] is None or b1["pick_json"]["bluesky_posted"] is not False:
                    errors.append({"key": key, "reason": "fresh_read3_pick_json", "branch": b1})
                if b1["evo_lines"] is None or len(b1["evo_lines"]) != 1:
                    errors.append({"key": key, "reason": "fresh_read3_evo", "branch": b1})
                if not b0["stdout"].startswith(b1["stdout"]) or "Saved to <PICKS_DIR>/2027-07-15.json" not in b1["stdout"]:
                    errors.append({"key": key, "reason": "fresh_read3_stdout_prefix", "base": b0["stdout"], "branch": b1["stdout"]})
            continue

        p, s = key.split("__")

        if p in EARLY_EXIT_PATHS:
            if b1 != b0 or b1["loader_calls"] != []:
                errors.append({"key": key, "reason": "early_exit_differs_from_base", "base": b0, "branch": b1})
            continue

        if not b1["loader_calls"] or any(c["kwargs"] != {"require_contest_state": True} for c in b1["loader_calls"]):
            errors.append({"key": key, "reason": "missing_require_contest_state_true", "calls": b1["loader_calls"]})

        if s in ACCEPTED_STATES:
            b0_cmp = {k: v for k, v in b0.items() if k != "loader_calls"}
            b1_cmp = {k: v for k, v in b1.items() if k != "loader_calls"}
            if b1_cmp != b0_cmp:
                errors.append({"key": key, "reason": "accepted_state_not_byte_identical", "base": b0_cmp, "branch": b1_cmp})
            continue

        if s == "corrupt_manual_json":
            if b0["exit"] != 1 or b0["exc_type"] != "ContestStateError":
                errors.append({"key": key, "reason": "unexpected_base_corrupt_manual", "base": b0})
            if (
                b1["exit"] != 1
                or b1["exc_type"] != "SystemExit"
                or not b1["stderr"].startswith("bts run: no pick: ")
                or b1["posts"] != []
                or b1["saves"] != []
                or b1["stdout"] != b0["stdout"]
            ):
                errors.append({"key": key, "reason": "corrupt_manual_branch_mismatch", "base": b0, "branch": b1})
            continue

        assert s in REFUSED_LIVE_STATES
        if (
            b1["exit"] != 1
            or b1["exc_type"] != "SystemExit"
            or not b1["stderr"].startswith("bts run: no pick: ")
            or len(b1["stderr"].strip().splitlines()) < 1
            or b1["posts"] != []
            or b1["saves"] != []
            or b1["pick_json"] is not None
            or b1["evo_lines"] is not None
            or len(b1["loader_calls"]) != 1
        ):
            errors.append({"key": key, "reason": "read1_refusal_violation", "branch": b1})
        if not b0["stdout"].startswith(b1["stdout"]) or "BTS PICKS — 2027-07-15" not in b1["stdout"]:
            errors.append({"key": key, "reason": "read1_stdout_prefix_mismatch", "base": b0["stdout"], "branch": b1["stdout"]})

    return {"checked_cells": checked, "errors": errors}


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], Path(sys.argv[3]))
        return

    install_network_audit_hook()
    base_out = REVIEW_DIR / "bts_run_base_v2.json"
    branch_out = REVIEW_DIR / "bts_run_branch_v2.json"
    env = dict(os.environ, TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")

    for which, out_file in (("base", base_out), ("branch", branch_out)):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--worker", which, str(out_file)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Worker {which} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")

    summary = verify_results(json.loads(base_out.read_text()), json.loads(branch_out.read_text()))
    (REVIEW_DIR / "bts_run_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    assert not summary["errors"], f"Errors in bts run probe: {summary['errors']}"


if __name__ == "__main__":
    main()
