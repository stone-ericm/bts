"""Adversarial Probe 3B (v4): Direct branch worker & audit probes + base-vs-branch detail summary."""
from __future__ import annotations

import contextlib
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import tomllib
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BRANCH_SRC = REPO / "src"

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)

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

AUDIT_FS_LOG: list[tuple[str, str]] = []
AUDIT_WATCH_PREFIXES: list[str] = []


def install_audit_hooks() -> None:
    def _hook(event: str, args: tuple) -> None:
        if event in BLOCKED_AUDIT_EVENTS:
            raise RuntimeError(f"Network/DNS forbidden by review audit hook: {event} {args!r}")
        if AUDIT_WATCH_PREFIXES and event in {
            "open", "os.stat", "os.listdir", "os.scandir", "os.mkdir", "os.rename", "os.remove", "os.unlink"
        }:
            target = args[0] if args else None
            if isinstance(target, (str, os.PathLike)):
                s = str(Path(target))
                for pfx in AUDIT_WATCH_PREFIXES:
                    if s == pfx or s.startswith(pfx + "/"):
                        AUDIT_FS_LOG.append((event, s))

    sys.addaudithook(_hook)


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.replace(tzinfo=None) if tz is None else NOW.astimezone(tz)


def enter_all(stack: contextlib.ExitStack, cms: list) -> None:
    for cm in cms:
        stack.enter_context(cm)


def write_uncovered_2026_state(picks_dir: Path) -> None:
    acc = picks_dir / "account_state"
    acc.mkdir(parents=True, exist_ok=True)
    (acc / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-28",
        "recorded_at": "2026-09-28T16:00:00Z",
    }))
    (picks_dir / "streak.json").write_text(json.dumps({"streak": 3, "saver_available": True, "season": 2027}))


def run_direct_probes() -> dict:
    sys.path.insert(0, str(BRANCH_SRC))
    import pandas as pd
    from click.testing import CliRunner
    from bts import cli, contest_state, entry_intent, orchestrator, picks, scheduler

    runner = CliRunner()
    report: dict = {}

    def make_df(gtime: str = f"{DAY}T23:10:00Z") -> pd.DataFrame:
        return pd.DataFrame([{
            "batter_name": "Jacob Wilson",
            "batter_id": 700363,
            "team": "ATH",
            "lineup": 1,
            "pitcher_name": "Jose Suarez",
            "pitcher_id": 660761,
            "p_hit_pa": 0.350,
            "p_game_hit": 0.900,
            "flags": "",
            "game_pk": 778899,
            "game_time": gtime,
        }])

    accepted_outward_tomls = [
        ("dm_req_true", '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'),
        ("pub_exp_true", '[scheduler]\npick_delivery = "public"\nentry_intent = "enter"\n[health_checks]\ncontest_state_expected = true\n'),
        ("dm_req_str_false", '[scheduler]\npick_delivery = "bluesky-dm"\nentry_intent = "enter"\ncontest_state_required = "false"\n'),
        ("pub_exp_int_1", '[scheduler]\npick_delivery = "feed"\n[health_checks]\ncontest_state_expected = 1\n'),
        ("dm_req_nonempty_list", '[scheduler]\npick_delivery = "direct-message"\ncontest_state_required = [1]\n'),
        ("pub_exp_nonempty_tbl", '[scheduler]\nposting_mode = "post"\n[health_checks.contest_state_expected]\nenabled = false\n'),
    ]
    outward_results = {}
    for label, tbody in accepted_outward_tomls:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "picks"
            write_uncovered_2026_state(pdir)
            hb = root / "hb.json"
            full_toml = (
                f'tiers = []\n[orchestrator]\npicks_dir = "{pdir}"\nheartbeat_path = "{hb}"\n'
                '[bluesky]\ndm_recipient = "op.bsky.social"\n' + tbody
            )
            cfile = root / "c.toml"
            cfile.write_text(full_toml)
            cfg = tomllib.loads(full_toml)
            assert entry_intent.live_gate_problem(cfg) is None, label
            assert entry_intent.live_gate_problem(cfg, delivery_override="public") is None, label

            pick_posts, pick_dms, alert_dms = [], [], []
            with contextlib.ExitStack() as stack:
                enter_all(stack, [
                    patch("datetime.datetime", FixedDateTime),
                    patch.object(contest_state, "datetime", FixedDateTime),
                    patch.object(picks, "datetime", FixedDateTime),
                    patch.object(scheduler, "_now_et", lambda: NOW),
                    patch.object(orchestrator, "run_cascade", return_value=(make_df(), "local")),
                    patch.object(scheduler, "check_confirmed_lineups", return_value={778899: {"away", "home"}}),
                    patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                    patch("bts.posting.should_post_now", return_value=True),
                    patch("bts.posting.post_to_bluesky", side_effect=lambda t: pick_posts.append(t) or "at://pub/x"),
                    patch("bts.dm.send_dm", side_effect=lambda r, t: pick_dms.append((r, t)) or "dm-x"),
                    patch("bts.health.alert.send_dm", side_effect=lambda r, t: alert_dms.append((r, t)) or "alert-x"),
                ])
                sc_res = scheduler.run_single_check(DAY, [778899], set(), cfg, 0.03)
                assert sc_res["pick_result"] is None and sc_res["should_post"] is False
                assert pick_posts == [] and pick_dms == []
                assert len(alert_dms) == 1

                cached_daily = picks.DailyPick(
                    date=DAY, run_time=NOW.isoformat(),
                    pick=picks.Pick(
                        batter_name="Jacob Wilson", batter_id=700363, team="ATH",
                        lineup_position=1, pitcher_name="Jose Suarez", pitcher_id=660761,
                        p_game_hit=0.900, flags=[], projected_lineup=True,
                        game_pk=778899, game_time=f"{DAY}T23:10:00Z",
                    ),
                    double_down=None, runner_up=None,
                )
                st_obj = scheduler.SchedulerState(
                    date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                    confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                    pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                with patch.object(scheduler, "_void_committed_games", return_value={}):
                    delivered = scheduler._deliver_and_lock_pick(cached_daily, cfg, pdir, st_obj, DAY, "test")
                assert delivered is False and st_obj.pick_locked is False
                assert pick_posts == [] and pick_dms == []

                orch_dms = []
                with patch("bts.dm.send_dm", side_effect=lambda r, t: orch_dms.append((r, t)) or "dm-o"):
                    r_orch = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfile)])
                assert r_orch.exit_code == 1
                assert pick_posts == []
                assert not (pdir / f"{DAY}.json").exists()
                assert len(orch_dms) == 1 and "No pick made" in orch_dms[0][1]

            outward_results[label] = "blocked_cleanly_zero_picks_delivered"
    report["accepted_outward_on_uncovered_state"] = outward_results

    accepted_private_tomls = [
        ("priv_research", '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'),
        ("local_absent_intent", '[scheduler]\nposting_mode = "local"\n'),
        ("priv_mode_str_false", '[scheduler]\nprivate_mode = "false"\n'),
    ]
    priv_results = {}
    for label, tbody in accepted_private_tomls:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdir = root / "picks"
            write_uncovered_2026_state(pdir)
            hb = root / "hb.json"
            full_toml = (
                f'tiers = []\n[orchestrator]\npicks_dir = "{pdir}"\nheartbeat_path = "{hb}"\n'
                '[bluesky]\ndm_recipient = "op.bsky.social"\n' + tbody +
                '[health_checks]\nenabled = false\n'
            )
            cfile = root / "c.toml"
            cfile.write_text(full_toml)
            # 16:40Z = 12:40 ET: check time 11:55 ET <= NOW (12:00 ET), and NOW < cutoff (12:35 ET)!
            gtime = f"{DAY}T16:40:00Z"
            game = {
                "gamePk": 778899,
                "gameDate": gtime,
                "teams": {"away": {"team": {"name": "ATH"}}, "home": {"team": {"name": "DET"}}},
                "status": {"abstractGameState": "Preview", "detailedState": "Pre-Game"},
            }
            posts, dms = [], []
            with contextlib.ExitStack() as stack:
                enter_all(stack, [
                    patch("datetime.datetime", FixedDateTime),
                    patch.object(contest_state, "datetime", FixedDateTime),
                    patch.object(picks, "datetime", FixedDateTime),
                    patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40),
                    patch.object(scheduler, "_now_et", lambda: NOW),
                    patch.object(scheduler, "datetime", FixedDateTime),
                    patch.object(scheduler, "fetch_schedule", return_value=[game]),
                    patch.object(scheduler, "notify_ready", lambda: None),
                    patch.object(scheduler, "notify_watchdog", lambda: None),
                    patch.object(scheduler, "_watchdog_ping_sleep", lambda _s: None),
                    patch.object(scheduler, "_idle_until_next_wakeup", lambda *_a, **_k: None),
                    patch.object(scheduler, "run_result_polling", lambda *_a, **_k: "hit"),
                    patch.object(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_a, **_k: None),
                    patch.object(scheduler, "check_confirmed_lineups", return_value={778899: {"away", "home"}}),
                    patch.object(scheduler, "_void_committed_games", return_value={}),
                    patch.object(orchestrator, "run_cascade", return_value=(make_df(gtime), "local")),
                    patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                    patch("bts.strategy.get_game_statuses", return_value={778899: "P"}),
                    patch("bts.posting.post_to_bluesky", side_effect=lambda t: posts.append(t) or "at://pub/x"),
                    patch("bts.dm.send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-x"),
                ])
                r_sched = runner.invoke(cli.cli, ["schedule", "--config", str(cfile), "--date", DAY])
            assert r_sched.exit_code == 0, (label, r_sched.stderr, repr(r_sched.exception))
            assert posts == [] and dms == []
            dec = json.loads((pdir / DAY / "decision.json").read_text())
            assert dec["delivery_status"] == "private_locked" and dec["scoreable"] is True
            saved_pick = json.loads((pdir / f"{DAY}.json").read_text())
            assert saved_pick["bluesky_posted"] is False and saved_pick["notification_sent"] is False
            priv_results[label] = "locked_privately_zero_dm_zero_post"
    report["accepted_private_through_real_schedule"] = priv_results

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "picks"
        evil_pdir = root / "evil_picks"
        acc = pdir / "account_state"
        acc.mkdir(parents=True)
        act_id = "f" * 32
        (acc / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": act_id, "switched_at": NOW.isoformat(), "revoked_at": None, "note": None,
        }))
        (acc / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": act_id,
            "active_streak": 5, "best_streak": 5, "source": "mlb_bts_profile",
            "source_date": DAY, "recorded_at": NOW.isoformat(),
        }))
        cfile = root / "mutable.toml"
        cfile.write_text(
            f'tiers = []\n[orchestrator]\npicks_dir = "{pdir}"\n'
            '[bluesky]\ndm_recipient = "op.bsky.social"\n'
            '[scheduler]\ncontest_state_required = true\n'
        )

        real_load = orchestrator.load_config
        loaded_dicts = []
        core_received = []
        real_core = orchestrator.orchestrate

        def mutating_loader(path: Path) -> dict:
            d = real_load(path)
            loaded_dicts.append(d)
            cfile.write_text(
                f'[corrupt toml!\npicks_dir = "{evil_pdir}"\ncontest_state_required = false\n'
            )
            return d

        def spying_core(config_path: Path, date: str, *, config: dict | None = None) -> bool:
            core_received.append(config)
            return real_core(config_path, date, config=config)

        AUDIT_FS_LOG.clear()
        AUDIT_WATCH_PREFIXES[:] = [str(cfile)]
        try:
            with contextlib.ExitStack() as stack:
                enter_all(stack, [
                    patch("datetime.datetime", FixedDateTime),
                    patch.object(contest_state, "datetime", FixedDateTime),
                    patch.object(picks, "datetime", FixedDateTime),
                    patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40),
                    patch.object(orchestrator, "load_config", side_effect=mutating_loader),
                    patch.object(orchestrator, "orchestrate", side_effect=spying_core),
                    patch.object(orchestrator, "run_cascade", return_value=(make_df(), "local")),
                    patch.object(picks, "get_game_statuses_detailed", return_value={778899: {"abstract": "P", "detailed": "Pre-Game"}}),
                    patch("bts.strategy.get_game_statuses", return_value={778899: "P"}),
                    patch("bts.posting.should_post_now", return_value=True),
                    patch("bts.posting.post_to_bluesky", return_value="at://pub/ok"),
                ])
                r_mut = runner.invoke(cli.cli, ["orchestrate", "--date", DAY, "--config", str(cfile)])
        finally:
            AUDIT_WATCH_PREFIXES.clear()

        open_events = [ev for ev in AUDIT_FS_LOG if ev[0] == "open"]
        assert r_mut.exit_code == 0, (r_mut.stderr, r_mut.exception)
        assert len(loaded_dicts) == 1 and len(core_received) == 1
        assert core_received[0] is loaded_dicts[0]
        assert len(open_events) == 2
        assert (pdir / f"{DAY}.json").exists() and not evil_pdir.exists()
        report["orchestrate_single_snapshot_mutation"] = {
            "exit": r_mut.exit_code,
            "same_dict_identity": core_received[0] is loaded_dicts[0],
            "cfile_open_events_count": len(open_events),
        }

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pdir = root / "picks"
        pdir.mkdir()
        (pdir / "sentinel.json").write_text('{"untouched": true}')
        hb = root / "hb.json"
        cfile = root / "refuse.toml"
        cfile.write_text(
            f'[orchestrator]\npicks_dir = "{pdir}"\nheartbeat_path = "{hb}"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n'
        )
        before_tree = {str(p.relative_to(root)): p.read_bytes().hex() for p in root.rglob("*") if p.is_file()}
        AUDIT_FS_LOG.clear()
        AUDIT_WATCH_PREFIXES[:] = [str(pdir), str(hb)]
        today_et_spy = Mock(side_effect=AssertionError("_today_et called on refusal"))
        run_day_spy = Mock(side_effect=AssertionError("run_day called on refusal"))
        try:
            with (
                patch.object(cli, "_today_et", today_et_spy),
                patch.object(scheduler, "run_day", run_day_spy),
            ):
                r_ref = runner.invoke(cli.cli, ["schedule", "--config", str(cfile)])
        finally:
            AUDIT_WATCH_PREFIXES.clear()
        after_tree = {str(p.relative_to(root)): p.read_bytes().hex() for p in root.rglob("*") if p.is_file()}
        assert r_ref.exit_code == 2
        assert r_ref.stderr.startswith("bts schedule: refusing to start: ")
        assert AUDIT_FS_LOG == [], f"Unexpected FS access on refusal: {AUDIT_FS_LOG}"
        assert before_tree == after_tree
        assert today_et_spy.call_count == 0 and run_day_spy.call_count == 0
        report["schedule_refusal_audit"] = {
            "exit": r_ref.exit_code,
            "fs_events_on_picks_or_hb": AUDIT_FS_LOG,
            "tree_identical": before_tree == after_tree,
        }

    base_ex = tomllib.loads((REVIEW_DIR / "base_tree" / "config" / "orchestrator.example.toml").read_text())
    branch_ex = tomllib.loads((REPO / "config" / "orchestrator.example.toml").read_text())
    assert branch_ex["scheduler"]["contest_state_required"] is True
    assert branch_ex["health_checks"]["contest_state_expected"] is True
    assert branch_ex["scheduler"]["entry_intent"] == "enter"
    assert entry_intent.live_gate_problem(branch_ex) is None
    assert entry_intent.check_entry_intent(branch_ex).problems == ()
    expected_from_base = copy.deepcopy(base_ex)
    expected_from_base["scheduler"]["contest_state_required"] = True
    expected_from_base["scheduler"]["entry_intent"] = "enter"
    expected_from_base["health_checks"]["contest_state_expected"] = True
    assert branch_ex == expected_from_base
    report["example_config_check"] = "parsed_ast_identical_except_3_specified_keys"
    return report


def main() -> None:
    install_audit_hooks()
    sys.path.insert(0, str(REVIEW_DIR))
    import probe_workers_cron_sched_orch_v4 as p4
    sub_cmp = p4.compare_clean(
        json.loads((REVIEW_DIR / "details_base_v3.json").read_text()),
        json.loads((REVIEW_DIR / "details_branch_v3.json").read_text()),
    )
    direct_rep = run_direct_probes()
    full_summary = {"base_vs_branch_details": sub_cmp, "branch_direct_probes": direct_rep}
    (REVIEW_DIR / "details_summary.json").write_text(json.dumps(full_summary, indent=2))
    print(json.dumps(full_summary, indent=2))


if __name__ == "__main__":
    main()
