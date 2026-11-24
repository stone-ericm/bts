"""Slice 3 supplementary probe: stale skip candidate across state change & BASESHA vs HEAD diff.

Runnable directly (`nice -n 19 .venv/bin/python .carbon-review/whole-range/test_slice3_base_and_edges.py`)
and adoptable as a pytest module (`pytest ...`).
Rules obeyed:
- No module-level import of bts.model.predict or lightgbm.
- No permanent process-wide state mutations.
- Zero network or socket calls.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd

from bts import cli as cli_mod
from bts import daily_decision, dm, orchestrator, picks, posting, scheduler, strategy

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_TMP = Path(__file__).resolve().parent / "scratch_tmp_edges"


@contextmanager
def _temp_root(tmp_path: Path | None):
    if tmp_path is not None:
        yield Path(tmp_path)
        return
    SCRATCH_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH_TMP) as td:
        yield Path(td)


def _pred_frame(day: str, p1: float = 0.72, projected: bool = False) -> pd.DataFrame:
    return pd.DataFrame([{
        "batter_id": 680757,
        "batter_name": "Steven Kwan",
        "team": "CLE",
        "lineup": 1,
        "pitcher_id": 543037,
        "pitcher_name": "Gerrit Cole",
        "opp": "NYY",
        "p_hit_pa": 0.26,
        "p_game_hit": p1,
        "flags": "PROJECTED" if projected else "",
        "projected_lineup": projected,
        "game_pk": 101,
        "game_time": f"{day}T23:10:00Z",
    }])


def _game_dict(pk: int, day: str) -> dict:
    return {
        "gamePk": pk,
        "gameType": "R",
        "officialDate": day,
        "gameDate": f"{day}T23:10:00Z",
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {
            "away": {"team": {"id": 114, "name": "Guardians"}},
            "home": {"team": {"id": 147, "name": "Yankees"}},
        },
    }


def _make_clock(box: list[dt.datetime]):
    class PinnedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            cur = box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    return PinnedDatetime


def test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms(tmp_path: Path | None = None):
    """5a guards saved picks via chosen_on, but state.final_skip_candidate has no chosen_on check.

    Concrete scenario on 2027-03-24 (1-game Opening Night, 19:10 ET):
    - 09:30 ET: operator enters dated observation with a typo (--streak 30 --best-streak 30 --saver used),
      or a stale automatic streak=30 is active.
    - 16:10 ET (Check 1): p=0.72 < 0.796 bar at streak=30 -> MDP skip; sets state.final_skip_candidate
      (streak=30) and state.skip_summary.
    - 16:40 ET: operator notices and runs CLI `bts set-contest-streak --date 2027-03-24 --streak 2
      --best-streak 2 --saver not_earned` while 17:10 and 18:10 checks still remain (at streak=2, p=0.72
      is an immediate pick, bar=0.0). Also an undelivered 03:00 preview `{date}.json` exists on disk.
    - 17:10 ET & 18:10 ET: prediction cascades transiently fail ((None, None)).
    Result:
    - state.final_skip_candidate (streak=30) is NOT cleared.
    - Step 5 skips deadline wait and fallback refresh (`skip_standing == True`).
    - Step 5b (`_maybe_alert_missed_pick` at 19:00 ET) returns early -> ZERO missed_pick DM.
    - EOD writes `decision.json` with `action='skip', source='mdp', streak=30`.
    - EOD health sees valid dated observation (streak=2) -> ZERO health DMs.
    - Next morning 08:30 ET `bts day-outcome --date 2027-03-24` exits 0 (`deliberate_skip`, `ok=True`) -> ZERO DM.
    Total DMs across the entire 24-hour cycle = 0.
    """
    with _temp_root(tmp_path) as root:
        day = "2027-03-24"
        picks_dir = root / "data" / "picks"
        picks_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = root / "orchestrator.toml"
        cfg = {
            "orchestrator": {
                "picks_dir": str(picks_dir),
                "models_dir": str(REPO_ROOT / "data" / "models"),
                "data_dir": str(root / "data" / "processed"),
                "heartbeat_path": str(root / "data" / ".heartbeat"),
            },
            "scheduler": {
                "pick_delivery": "dm",
                "entry_intent": "enter",
                "contest_state_required": True,
                "lineup_check_offset_min": 60,
                "early_lock_gap": 0.015,
                "fallback_deadline_min": 35,
                "cascade_budget_min": 12,
                "operator_reserve_min": 10,
                "missed_pick_alert_min": 10,
                "shadow_model": False,
                "live_forward_capture_on_lock": False,
            },
            "health_checks": {
                "enabled": False,
                "contest_state_expected": True,
            },
            "bluesky": {"dm_recipient": "operator.bsky.social"},
            "tiers": [{"name": "local", "type": "local"}],
        }
        cfg_path.write_text(
            f'[orchestrator]\npicks_dir = "{picks_dir}"\n'
            '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
            '[health_checks]\ncontest_state_expected = true\n'
            '[bluesky]\ndm_recipient = "operator.bsky.social"\n'
        )

        now_box = [dt.datetime(2027, 3, 24, 9, 30, tzinfo=ET)]
        Clock = _make_clock(now_box)
        sent_dms: list[str] = []
        runner = CliRunner()

        # Plant an undelivered 03:00 preview pick file on disk (as cron 03:00 bts preview writes)
        preview_pick = picks.DailyPick(
            date=day,
            run_time=f"{day}T03:00:00-04:00",
            pick=picks.Pick(
                batter_name="Steven Kwan", batter_id=680757, team="CLE",
                lineup_position=1, pitcher_name="Gerrit Cole", pitcher_id=543037,
                p_game_hit=0.72, flags=["PROJECTED"], projected_lineup=True,
                game_pk=101, game_time=f"{day}T23:10:00Z", pitcher_team="NYY",
            ),
            double_down=None,
            runner_up=None,
            chosen_on={"schema": "bts_pick_chosen_on_v1", "required": False,
                       "kind": "model", "streak": 0, "saver_available": True, "dated": None},
        )
        picks.save_pick(preview_pick, picks_dir)

        old_mdp_cache = strategy._mdp_cache
        try:
            with ExitStack() as stack:
                for target in (
                    "bts.contest_state.datetime",
                    "bts.season_contract.datetime",
                    "bts.health.contest_state.datetime",
                    "bts.health.runner.datetime",
                    "bts.health.alert.datetime",
                ):
                    stack.enter_context(patch(target, Clock))
                det_status = {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
                stack.enter_context(patch.object(scheduler, "_now_et", side_effect=lambda: now_box[0]))
                stack.enter_context(patch.object(dm, "send_dm", side_effect=lambda rec, txt: sent_dms.append((now_box[0].strftime("%H:%M"), txt)) or "dm-ok"))
                stack.enter_context(patch("bts.health.alert.send_dm", side_effect=lambda rec, txt: sent_dms.append((now_box[0].strftime("%H:%M"), txt)) or "dm-ok"))
                stack.enter_context(patch.object(posting, "post_to_bluesky", side_effect=AssertionError("unexpected post")))
                stack.enter_context(patch.object(scheduler, "fetch_schedule", return_value=[_game_dict(101, day)]))
                stack.enter_context(patch.object(picks, "get_game_statuses_detailed", return_value=det_status))
                stack.enter_context(patch.object(strategy, "get_game_statuses_detailed", return_value=det_status))
                stack.enter_context(patch.object(picks, "get_game_statuses", return_value={101: "P"}))
                stack.enter_context(patch.object(strategy, "get_game_statuses", return_value={101: "P"}))
                stack.enter_context(patch.object(scheduler, "check_confirmed_lineups", return_value={101: {"away", "home"}}))
                stack.enter_context(patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *a, **k: nullcontext()))
                stack.enter_context(patch.object(scheduler, "write_heartbeat", lambda *a, **k: None))
                stack.enter_context(patch.object(scheduler, "notify_watchdog", lambda *a, **k: None))
                stack.enter_context(patch.object(scheduler, "_idle_until_next_wakeup", lambda *a, **k: None))

                # 09:30 ET: initial dated observation has streak=30 (typo or pre-correction value)
                with patch("datetime.datetime", Clock):
                    r_init = runner.invoke(cli_mod.cli, [
                        "set-contest-streak", "--date", day, "--streak", "30",
                        "--best-streak", "30", "--saver", "used",
                        "--reason", "morning entry", "--picks-dir", str(picks_dir),
                    ])
                    assert r_init.exit_code == 0, r_init.output

                def _sleep_advance(secs: float):
                    prev = now_box[0]
                    nxt = prev + dt.timedelta(seconds=max(secs, 0))
                    # At 16:40 ET (during sleep from 16:10 -> 17:10), operator enters corrected streak=2!
                    t_fix = dt.datetime(2027, 3, 24, 16, 40, tzinfo=ET)
                    if prev < t_fix <= nxt:
                        now_box[0] = t_fix
                        with patch("datetime.datetime", Clock):
                            r_fix = runner.invoke(cli_mod.cli, [
                                "set-contest-streak", "--date", day, "--streak", "2",
                                "--best-streak", "2", "--saver", "not_earned",
                                "--reason", "corrected typo from 30 to 2", "--picks-dir", str(picks_dir),
                            ])
                            assert r_fix.exit_code == 0, r_fix.output
                    now_box[0] = nxt

                stack.enter_context(patch.object(scheduler, "_watchdog_ping_sleep", side_effect=_sleep_advance))

                check_n = [0]

                def _cascade(_tiers, _date):
                    check_n[0] += 1
                    if check_n[0] == 1:
                        return _pred_frame(day, p1=0.72, projected=False), "local"
                    return None, None  # transient cascade failure on 17:10 and 18:10

                stack.enter_context(patch.object(orchestrator, "run_cascade", side_effect=_cascade))
                stack.enter_context(patch.object(
                    scheduler, "count_new_confirmations",
                    side_effect=lambda pks, cs: cs.update({(101, "away"), (101, "home")}) or 2,
                ))

                # Confirm that at streak=2, p=0.72 WOULD have been an immediate pick if re-evaluated
                probe_sel = strategy.select_pick(
                    _pred_frame(day, p1=0.72, projected=False),
                    day, root / "probe_picks", streak=2, saver_available=True, allow_double=True,
                    game_statuses_detailed=det_status, require_detailed_statuses=True,
                )
                assert probe_sel.action in ("single", "double")

                now_box[0] = dt.datetime(2027, 3, 24, 10, 0, tzinfo=ET)
                scheduler.run_day(day, cfg)

                dec = daily_decision.load_decision(day, picks_dir)
                assert dec is not None
                assert dec["action"] == "skip" and dec["source"] == "mdp" and dec["streak"] == 30

                from bts.health import contest_state as health_cs
                from bts.health import late_delivery as health_ld
                from bts.health import post_failure as health_pf
                eod_cs = health_cs.check(picks_dir, expected=True, now=now_box[0])
                eod_ld = health_ld.check(picks_dir, today=dt.date(2027, 3, 24), now=now_box[0])
                eod_pf = health_pf.check(picks_dir, today=dt.date(2027, 3, 24))
                assert eod_cs == [] and eod_ld == [] and eod_pf == []

                # Next morning 08:30 ET: bts day-outcome runs
                now_box[0] = dt.datetime(2027, 3, 25, 8, 30, tzinfo=ET)
                r_out = runner.invoke(cli_mod.cli, [
                    "day-outcome", "--date", day, "--picks-dir", str(picks_dir),
                    "--config", str(cfg_path), "--expected-username", "eric",
                    "--dm-recipient", "operator.bsky.social",
                ])
                assert r_out.exit_code == 0, r_out.output
                out_payload = json.loads(r_out.output)
                assert out_payload["state"] == "deliberate_skip" and out_payload["ok"] is True
                # Only the 16:10 pre-correction skip notice was sent; ZERO DMs after the 16:40 correction!
                assert len(sent_dms) == 1 and sent_dms[0][0] == "16:10" and "No pick yet" in sent_dms[0][1]
        finally:
            strategy._mdp_cache = old_mdp_cache


def test_basesha_vs_head_step4_to_step5_double_cascade(tmp_path: Path | None = None):
    """Compare BASESHA (619abda7) vs HEAD (bc143131) when Step 4 fallback cascade fails after state change."""
    with _temp_root(tmp_path) as root:
        base_dir = root / "base_619abda7"
        base_dir.mkdir(parents=True, exist_ok=True)
        # Extract src/ from 619abda7 into scratch temp dir (cleaned up automatically when root exits)
        git_proc = subprocess.Popen(
            ["git", "archive", "619abda7", "src"],
            cwd=REPO_ROOT, stdout=subprocess.PIPE,
        )
        tar_proc = subprocess.run(
            ["tar", "-x", "-C", str(base_dir)],
            stdin=git_proc.stdout, check=True,
        )
        assert git_proc.wait() == 0 and tar_proc.returncode == 0

        driver = root / "driver_child.py"
        driver.write_text(
            "from contextlib import nullcontext\n"
            "import datetime as dt, json, sys\n"
            "from pathlib import Path\n"
            "from unittest.mock import patch\n"
            "from zoneinfo import ZoneInfo\n"
            "import pandas as pd\n"
            "from bts import dm, orchestrator, picks, posting, scheduler, strategy\n"
            "ET = ZoneInfo('America/New_York')\n"
            "mode, work = sys.argv[1], Path(sys.argv[2])\n"
            "day = '2026-07-16'\n"
            "picks_dir = work / 'data' / 'picks'\n"
            "picks_dir.mkdir(parents=True, exist_ok=True)\n"
            "acct = picks_dir / 'account_state'\n"
            "acct.mkdir(parents=True, exist_ok=True)\n"
            "cfg = {\n"
            "  'orchestrator': {'picks_dir': str(picks_dir), 'models_dir': str(Path.cwd() / 'data' / 'models'),\n"
            "                   'data_dir': str(work / 'data' / 'processed'), 'heartbeat_path': str(work / 'hb')},\n"
            "  'scheduler': {'pick_delivery': 'dm', 'entry_intent': 'enter', 'contest_state_required': True,\n"
            "                'lineup_check_offset_min': 60, 'early_lock_gap': 0.015, 'fallback_deadline_min': 35,\n"
            "                'cascade_budget_min': 12, 'operator_reserve_min': 10, 'missed_pick_alert_min': 10,\n"
            "                'shadow_model': False, 'live_forward_capture_on_lock': False},\n"
            "  'health_checks': {'enabled': False, 'contest_state_expected': True},\n"
            "  'bluesky': {'dm_recipient': 'op.bsky.social'},\n"
            "  'tiers': [{'name': 'local', 'type': 'local'}],\n"
            "}\n"
            "now_box = [dt.datetime(2026, 7, 16, 10, 0, tzinfo=ET)]\n"
            "class Clk(dt.datetime):\n"
            "    @classmethod\n"
            "    def now(cls, tz=None):\n"
            "        c = now_box[0]\n"
            "        return c.astimezone(tz) if tz else c.replace(tzinfo=None)\n"
            "def set_state(streak):\n"
            "    if mode == 'head':\n"
            "        (acct / 'contest_streak.manual.json').write_text(json.dumps({\n"
            "            'schema_version': 'bts_contest_streak_manual_v3', 'effective_date': day, 'season': 2026,\n"
            "            'expires_at': '2026-07-16T23:59:59.999999-04:00', 'active_streak': streak,\n"
            "            'best_streak': max(streak, 5), 'saver_state': 'not_earned',\n"
            "            'recorded_at': '2026-07-16T14:00:00Z', 'source': 'manual_cli'})) \n"
            "    else:\n"
            "        (acct / 'contest_streak.manual.json').write_text(json.dumps({\n"
            "            'schema_version': 'bts_contest_streak_manual_v1', 'active_streak': streak,\n"
            "            'best_streak': max(streak, 5), 'override_expires_at': '2026-07-17T04:00:00Z',\n"
            "            'recorded_at': '2026-07-16T14:00:00Z', 'source': 'manual_cli'}))\n"
            "set_state(2)\n"
            "df = pd.DataFrame([{'batter_id': 680757, 'batter_name': 'Steven Kwan', 'team': 'CLE', 'lineup': 1,\n"
            "  'pitcher_id': 543037, 'pitcher_name': 'Gerrit Cole', 'opp': 'NYY', 'p_hit_pa': 0.29,\n"
            "  'p_game_hit': 0.84, 'flags': 'PROJECTED', 'projected_lineup': True, 'game_pk': 101,\n"
            "  'game_time': f'{day}T23:10:00Z'}])\n"
            "g = [{'gamePk': 101, 'gameType': 'R', 'officialDate': day, 'gameDate': f'{day}T23:10:00Z',\n"
            "      'status': {'abstractGameCode': 'P', 'detailedState': 'Scheduled'},\n"
            "      'teams': {'away': {'team': {'id': 114, 'name': 'G'}}, 'home': {'team': {'id': 147, 'name': 'Y'}}}}]\n"
            "calls = [0]\n"
            "events = []\n"
            "def cas(_t, _d):\n"
            "    calls[0] += 1\n"
            "    events.append(['cascade_start', calls[0], now_box[0].strftime('%H:%M')])\n"
            "    if calls[0] == 1:\n"
            "        return df, 'local'\n"
            "    if mode == 'head' and calls[0] in (2, 3):\n"
            "        return df, 'local'\n"
            "    if (mode == 'base' and calls[0] == 2) or (mode == 'head' and calls[0] == 4):\n"
            "        # Step 4 fallback cascade at 18:35 ET\n"
            "        set_state(3)  # operator updated streak before fallback finished\n"
            "        now_box[0] += dt.timedelta(minutes=16)\n"
            "        return None, None\n"
            "    # Unbudgeted Step 5 cascade at 18:51 ET (only reached on head after 5a refusal)!\n"
            "    now_box[0] += dt.timedelta(minutes=15)\n"
            "    df2 = df.copy(); df2['projected_lineup'] = False; df2['flags'] = ''\n"
            "    return df2, 'local'\n"
            "det = {101: {'abstract': 'P', 'detailed': 'Scheduled', 'code': 'S'}}\n"
            "dms = []\n"
            "with patch('bts.contest_state.datetime', Clk), \\\n"
            "     patch.object(scheduler, '_now_et', side_effect=lambda: now_box[0]), \\\n"
            "     patch.object(scheduler, '_watchdog_ping_sleep', side_effect=lambda s: now_box.__setitem__(0, now_box[0] + dt.timedelta(seconds=max(s,0)))), \\\n"
            "     patch.object(dm, 'send_dm', side_effect=lambda r, t: dms.append((now_box[0].strftime('%H:%M'), t)) or 'dm-1'), \\\n"
            "     patch('bts.health.alert.send_dm', side_effect=lambda r, t: dms.append((now_box[0].strftime('%H:%M'), t)) or 'dm-1'), \\\n"
            "     patch.object(posting, 'post_to_bluesky', side_effect=AssertionError), \\\n"
            "     patch.object(scheduler, 'fetch_schedule', return_value=g), \\\n"
            "     patch.object(picks, 'get_game_statuses_detailed', return_value=det), \\\n"
            "     patch.object(strategy, 'get_game_statuses_detailed', return_value=det), \\\n"
            "     patch.object(picks, 'get_game_statuses', return_value={101: 'P'}), \\\n"
            "     patch.object(strategy, 'get_game_statuses', return_value={101: 'P'}), \\\n"
            "     patch.object(scheduler, 'check_confirmed_lineups', return_value={101: set()}), \\\n"
            "     patch.object(scheduler, 'heartbeat_watchdog', side_effect=lambda *a, **k: nullcontext()), \\\n"
            "     patch.object(orchestrator, 'run_cascade', side_effect=cas), \\\n"
            "     patch.object(scheduler, 'count_new_confirmations', return_value=0), \\\n"
            "     patch.object(scheduler, 'write_heartbeat', lambda *a, **k: None), \\\n"
            "     patch.object(scheduler, 'notify_watchdog', lambda *a, **k: None), \\\n"
            "     patch.object(scheduler, '_idle_until_next_wakeup', lambda *a, **k: None), \\\n"
            "     patch.object(scheduler, 'run_result_polling', return_value='final'):\n"
            "    if mode == 'head':\n"
            "        with patch('bts.season_contract.datetime', Clk):\n"
            "            scheduler.run_day(day, cfg)\n"
            "    else:\n"
            "        scheduler.run_day(day, cfg)\n"
            "st = json.loads((picks_dir / day / 'scheduler_state.json').read_text())\n"
            "print(json.dumps({'mode': mode, 'events': events, 'pick_locked': st['pick_locked'],\n"
            "                  'delivery_refusals': st.get('delivery_refusals'), 'dms': dms}))\n"
        )

        env_base = dict(os.environ, PYTHONPATH=str(base_dir / "src"))
        env_head = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
        res_base = subprocess.run(
            [sys.executable, str(driver), "base", str(root / "w_base")],
            cwd=REPO_ROOT, env=env_base, capture_output=True, text=True,
        )
        assert res_base.returncode == 0, f"base failed:\n{res_base.stderr}\n{res_base.stdout}"
        res_head = subprocess.run(
            [sys.executable, str(driver), "head", str(root / "w_head")],
            cwd=REPO_ROOT, env=env_head, capture_output=True, text=True,
        )
        assert res_head.returncode == 0, f"head failed:\n{res_head.stderr}\n{res_head.stdout}"
        out_base = json.loads(res_base.stdout.strip().splitlines()[-1])
        out_head = json.loads(res_head.stdout.strip().splitlines()[-1])

        # On BASESHA (619abda7): Step 4 delivered cached_daily on the old streak at 18:51, never ran Step 5
        assert out_base["pick_locked"] is True
        assert len(out_base["events"]) == 2  # 18:10 check + 18:35 Step 4 fallback only

        # On HEAD (bc143131): 5a refused cached_daily at 18:51, then Step 5 launched an unbudgeted
        # second fallback cascade at 18:51 -> finished 19:06 (1 min past 19:05 cutoff) -> refused late,
        # and missed_pick DM did not fire until 19:06 (after cutoff closed)!
        assert out_head["pick_locked"] is False, out_head
        assert ["cascade_start", 4, "18:35"] in out_head["events"], out_head
        assert ["cascade_start", 5, "18:51"] in out_head["events"], out_head
        assert "19:05:00" in out_head["delivery_refusals"][0]["cutoff_et"], out_head
        assert [t for t, _ in out_head["dms"]] == ["18:51", "19:06"], out_head
        assert "LATE DELIVERY REFUSED" in out_head["dms"][1][1], out_head


if __name__ == "__main__":
    try:
        test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms()
        test_basesha_vs_head_step4_to_step5_double_cascade()
        print("BASE_AND_EDGES PROBE OK")
    finally:
        shutil.rmtree(SCRATCH_TMP, ignore_errors=True)
