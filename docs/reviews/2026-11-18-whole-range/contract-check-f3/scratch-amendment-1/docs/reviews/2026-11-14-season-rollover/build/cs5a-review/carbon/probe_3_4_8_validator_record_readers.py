"""Adversarial probe for CS5a items 3, 4, 8 (validator/matcher oracle, record lifecycle, readers)."""

from __future__ import annotations

import copy
import datetime as dt_mod
import hashlib
import io
import json
import re
import sys
import tempfile
from contextlib import ExitStack, nullcontext, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REAL_DATETIME = dt_mod.datetime
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = REAL_DATETIME.fromisoformat("2027-07-15T18:35:00-04:00")


def install_network_guard() -> list[tuple[str, tuple]]:
    file_opens: list[tuple[str, tuple]] = []
    trap_files = [False]

    def hook(event: str, args: tuple) -> None:
        if event in {"socket.getaddrinfo", "socket.getnameinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            raise RuntimeError(f"Forbidden DNS: {event} {args!r}")
        if event == "socket.connect":
            addr = args[1] if len(args) > 1 else None
            host = addr[0] if isinstance(addr, tuple) and addr else None
            if host not in {"127.0.0.1", "::1", "localhost"}:
                raise RuntimeError(f"Forbidden connect: {addr!r}")
        if trap_files[0] and event == "open":
            file_opens.append((event, args))

    sys.addaudithook(hook)
    return trap_files, file_opens


# --- Independent Oracle for Item 3 ---

def oracle_valid_record(v: object) -> bool:
    if type(v) is not dict:
        return False
    if set(v.keys()) != {"schema", "required", "kind", "streak", "saver_available", "dated"}:
        return False
    if type(v["schema"]) is not str or v["schema"] != "bts_pick_chosen_on_v1":
        return False
    if type(v["required"]) is not bool or type(v["saver_available"]) is not bool:
        return False
    if type(v["kind"]) is not str or v["kind"] not in {"dated", "automatic", "legacy_manual", "model"}:
        return False
    if type(v["streak"]) is not int or v["streak"] < 0:
        return False
    if v["kind"] in {"dated", "automatic"} and v["streak"] > 57:
        return False
    if v["kind"] != "dated":
        return v["dated"] is None
    d = v["dated"]
    if type(d) is not dict or set(d.keys()) != {"effective_date", "active_streak", "best_streak", "saver_state"}:
        return False
    eff = d["effective_date"]
    if type(eff) is not str or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", eff):
        return False
    try:
        parsed_d = dt_mod.date.fromisoformat(eff)
    except ValueError:
        return False
    if parsed_d.isoformat() != eff:
        return False
    act, bst, sav = d["active_streak"], d["best_streak"], d["saver_state"]
    if type(act) is not int or type(bst) is not int:
        return False
    if not (0 <= act <= bst <= 57) or act != v["streak"]:
        return False
    if type(sav) is not str or sav not in {"not_earned", "active", "used"}:
        return False
    expected_saver_avail = (sav == "active") or (sav == "not_earned" and act < 10)
    return v["saver_available"] == expected_saver_avail


def oracle_matches(v: object, state) -> bool:
    if not oracle_valid_record(v):
        return False
    if v["required"] is not True:
        return False
    if state.observation_kind == "automatic" and v["kind"] == "automatic":
        return True
    if state.observation_kind == "dated" and v["kind"] == "dated":
        return v["dated"] == state.dated_identity
    return False


def build_record_corpus() -> list[tuple[str, object]]:
    cases: list[tuple[str, object]] = []
    # 1. Valid records of all 4 kinds x required True/False, incl. 0, 57, and 58/999 for model/legacy_manual
    for kind in ("dated", "automatic", "legacy_manual", "model"):
        for req in (True, False):
            streaks = [0, 7, 57] + ([58, 999] if kind in {"legacy_manual", "model"} else [])
            for s in streaks:
                sav_avail = s < 10
                rec = {
                    "schema": "bts_pick_chosen_on_v1",
                    "required": req,
                    "kind": kind,
                    "streak": s,
                    "saver_available": sav_avail,
                    "dated": (
                        {
                            "effective_date": DAY,
                            "active_streak": s,
                            "best_streak": max(s, 9 if s <= 9 else s),
                            "saver_state": "not_earned",
                        } if kind == "dated" else None
                    ),
                }
                cases.append((f"valid_{kind}_req={req}_s={s}", rec))

    # 2. Dated saver consistency across streaks 0, 9, 10, 15, 16, 57 x each saver_state x bool saver_available
    for s in (0, 9, 10, 15, 16, 57):
        for sav_state in ("not_earned", "active", "used"):
            for sav_bool in (True, False):
                cases.append((
                    f"saver_rule_s={s}_{sav_state}_avail={sav_bool}",
                    {
                        "schema": "bts_pick_chosen_on_v1",
                        "required": True,
                        "kind": "dated",
                        "streak": s,
                        "saver_available": sav_bool,
                        "dated": {
                            "effective_date": DAY,
                            "active_streak": s,
                            "best_streak": max(s, 16),
                            "saver_state": sav_state,
                        },
                    },
                ))

    # 3. Dated mismatches (each of the 4 fields differing while valid)
    base_dated = {
        "schema": "bts_pick_chosen_on_v1",
        "required": True,
        "kind": "dated",
        "streak": 7,
        "saver_available": False,
        "dated": {"effective_date": DAY, "active_streak": 7, "best_streak": 9, "saver_state": "used"},
    }
    for label, mut in (
        ("diff_date", {"dated": {**base_dated["dated"], "effective_date": "2027-07-14"}}),
        ("diff_active", {"streak": 8, "dated": {**base_dated["dated"], "active_streak": 8}}),
        ("diff_best", {"dated": {**base_dated["dated"], "best_streak": 10}}),
        ("diff_saver", {"saver_available": True, "dated": {**base_dated["dated"], "saver_state": "active"}}),
    ):
        c = copy.deepcopy(base_dated)
        c.update(mut)
        cases.append((label, c))

    # 4. Comprehensive invalid corpus
    for non_dict in (None, True, False, 0, 1, 3.14, "str", [], [base_dated]):
        cases.append((f"outer_{non_dict!r}", non_dict))

    for k in list(base_dated.keys()):
        c = copy.deepcopy(base_dated)
        c.pop(k)
        cases.append((f"missing_outer_{k}", c))
    c_extra = copy.deepcopy(base_dated)
    c_extra["extra"] = 1
    cases.append(("extra_outer", c_extra))

    for dk in list(base_dated["dated"].keys()):
        c = copy.deepcopy(base_dated)
        c["dated"].pop(dk)
        cases.append((f"missing_dated_{dk}", c))
    c_dextra = copy.deepcopy(base_dated)
    c_dextra["dated"]["extra"] = 1
    cases.append(("extra_dated", c_dextra))

    bad_mutations = [
        (("schema",), "bts_pick_chosen_on_v2"),
        (("schema",), None),
        (("schema",), []),
        (("schema",), {}),
        (("required",), 1),
        (("required",), 0),
        (("required",), "true"),
        (("required",), []),
        (("saver_available",), 1),
        (("saver_available",), 0),
        (("saver_available",), None),
        (("kind",), "other"),
        (("kind",), []),
        (("kind",), {}),
        (("kind",), 123),
        (("streak",), True),
        (("streak",), False),
        (("streak",), 7.0),
        (("streak",), "7"),
        (("streak",), -1),
        (("streak",), 58),
        (("streak",), []),
        (("streak",), 6),  # disagrees with active_streak=7
        (("dated",), None),
        (("dated",), []),
        (("dated",), "2027-07-15"),
        (("dated", "effective_date"), "2027-7-15"),
        (("dated", "effective_date"), "2027-07-5"),
        (("dated", "effective_date"), "20270715"),
        (("dated", "effective_date"), "2027-W28-4"),
        (("dated", "effective_date"), "2027-02-29"),
        (("dated", "effective_date"), "2027-02-30"),
        (("dated", "effective_date"), "2027-13-01"),
        (("dated", "effective_date"), "2027-00-10"),
        (("dated", "effective_date"), "2027-07-15T00:00:00Z"),
        (("dated", "effective_date"), ""),
        (("dated", "effective_date"), "\x00"),
        (("dated", "effective_date"), 20270715),
        (("dated", "effective_date"), []),
        (("dated", "active_streak"), True),
        (("dated", "active_streak"), False),
        (("dated", "active_streak"), 7.0),
        (("dated", "active_streak"), -1),
        (("dated", "active_streak"), 8),  # disagrees with outer streak=7
        (("dated", "best_streak"), True),
        (("dated", "best_streak"), False),
        (("dated", "best_streak"), 6),  # best < active
        (("dated", "best_streak"), 58),
        (("dated", "best_streak"), 9.0),
        (("dated", "best_streak"), None),
        (("dated", "saver_state"), "uninitialized"),
        (("dated", "saver_state"), []),
        (("dated", "saver_state"), {}),
        (("dated", "saver_state"), True),
    ]
    for path, val in bad_mutations:
        c = copy.deepcopy(base_dated)
        tgt = c if len(path) == 1 else c["dated"]
        tgt[path[-1]] = val
        cases.append((f"bad_{'.'.join(path)}={val!r}", c))

    # automatic at 58 and non-dated kinds with dated != None
    cases.append(("auto_58", {
        "schema": "bts_pick_chosen_on_v1", "required": True, "kind": "automatic",
        "streak": 58, "saver_available": False, "dated": None,
    }))
    for non_dated_kind in ("automatic", "legacy_manual", "model"):
        cases.append((f"{non_dated_kind}_with_dated", {
            "schema": "bts_pick_chosen_on_v1", "required": True, "kind": non_dated_kind,
            "streak": 7, "saver_available": False, "dated": copy.deepcopy(base_dated["dated"]),
        }))

    return cases


def run_item_3_oracle(trap_files, file_opens) -> dict:
    from bts import contest_state, picks
    from probe_1_2_5_6_7 import set_covered_2027_auto, set_dated_obs, write_account

    corpus = build_record_corpus()

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p_dated = root / "dated"
        set_dated_obs(p_dated, now=NOW, streak=7, best=9, saver="used", reason="initial",
                      recorded_at="2027-07-15T12:00:00Z")
        st_dated = contest_state.load_decision_streak_state(p_dated, now=NOW, require_contest_state=True)

        # Dated state differing ONLY in recorded_at and reason!
        p_dated_meta = root / "dated_meta"
        set_dated_obs(p_dated_meta, now=NOW, streak=7, best=9, saver="used", reason="updated reason",
                      recorded_at="2027-07-15T22:15:00Z")
        st_dated_meta = contest_state.load_decision_streak_state(p_dated_meta, now=NOW, require_contest_state=True)
        assert st_dated.dated_identity == st_dated_meta.dated_identity

        p_auto = root / "auto"
        set_covered_2027_auto(p_auto, now=NOW, streak=25)
        st_auto = contest_state.load_decision_streak_state(p_auto, now=NOW, require_contest_state=True)

        p_leg = root / "legacy"
        write_account(p_leg, "contest_streak.manual.json", {"active_streak": 7, "source": "mlb_bts_profile"})
        st_legacy = contest_state.load_decision_streak_state(p_leg, now=NOW, require_contest_state=False)

        p_mod = root / "model"
        p_mod.mkdir(parents=True, exist_ok=True)
        st_model = contest_state.load_decision_streak_state(p_mod, now=NOW, require_contest_state=False)

        states = [
            ("live_dated", st_dated),
            ("live_dated_meta_changed", st_dated_meta),
            ("live_auto", st_auto),
            ("opt_legacy", st_legacy),
            ("opt_model", st_model),
        ]

    valid_count = 0
    invalid_count = 0
    pair_count = 0
    sample_pick = picks.DailyPick(
        date=DAY, run_time=NOW.isoformat(),
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                        0.82, [], False, 101, f"{DAY}T23:10:00Z"),
        double_down=None, runner_up=None,
    )

    clock_boom = Mock(side_effect=AssertionError("judge accessed clock"))
    file_opens.clear()
    trap_files[0] = True
    try:
        with patch.object(picks, "datetime", clock_boom):
            for label, cand in corpus:
                expected_valid = oracle_valid_record(cand)
                got = picks._valid_chosen_on(copy.deepcopy(cand))
                assert (got is not None) is expected_valid, (label, cand, got, expected_valid)
                if expected_valid:
                    valid_count += 1
                    assert got == cand
                else:
                    invalid_count += 1

                for st_name, st_obj in states:
                    pair_count += 1
                    sample_pick.chosen_on = copy.deepcopy(cand)
                    exp_match = oracle_matches(cand, st_obj)
                    msg = picks.saved_pick_refusal(sample_pick, st_obj)
                    if exp_match:
                        assert msg is None, (label, st_name, msg)
                    else:
                        assert isinstance(msg, str), (label, st_name, msg)
                        # Check exact 4-part structure, order, no command name, and all 4 dated values on both sides!
                        assert msg.startswith(f"Saved pick for {DAY} was not delivered because ")
                        assert msg.endswith("A later scheduled check or fallback refresh may choose a replacement, if one remains.")
                        assert "bts " not in msg and "set-contest-streak" not in msg and "season-contract" not in msg
                        chosen_part, live_part = msg.split(". The live state is ", 1)
                        if not expected_valid:
                            assert "there is no valid record of what it was chosen on" in chosen_part
                        elif not cand["required"]:
                            assert "it was chosen without the live check (for example by the preview)" in chosen_part
                        elif cand["kind"] == "dated":
                            d = cand["dated"]
                            expected_desc = (
                                f"dated (effective date {d['effective_date']}, active streak {d['active_streak']}, "
                                f"best streak {d['best_streak']}, saver {d['saver_state']})"
                            )
                            assert expected_desc in chosen_part, (label, chosen_part)
                        else:
                            assert f"it was chosen on {cand['kind']}" in chosen_part

                        if st_obj.observation_kind == "dated":
                            ld = st_obj.dated_identity
                            expected_live = (
                                f"dated (effective date {ld['effective_date']}, active streak {ld['active_streak']}, "
                                f"best streak {ld['best_streak']}, saver {ld['saver_state']})"
                            )
                            assert live_part.startswith(expected_live + ". "), live_part
                        else:
                            assert live_part.startswith(st_obj.observation_kind + ". "), live_part
    finally:
        trap_files[0] = False

    assert len(file_opens) == 0, f"Judge/validator opened files: {file_opens}"
    assert not clock_boom.called

    # Specifically verify dated match is blind to recorded_at and reason on a real chosen_on_record
    rec_from_first = picks.chosen_on_record(st_dated, required=True)
    sample_pick.chosen_on = rec_from_first
    assert picks.saved_pick_refusal(sample_pick, st_dated_meta) is None

    return {
        "corpus_size": len(corpus),
        "valid_records": valid_count,
        "invalid_records": invalid_count,
        "pairs_checked": pair_count,
        "blind_to_recorded_at_and_reason": True,
        "zero_file_and_clock_reads": True,
    }


def run_item_4_record_lifecycle() -> dict:
    """Verify every save path preserves chosen_on (and never invents one on unrecorded reuse), plus old file bytes."""
    from click.testing import CliRunner
    from bts import cli, contest_state, orchestrator, picks, scheduler
    from probe_1_2_5_6_7 import make_sched_state, offline_harness, set_dated_obs

    runner = CliRunner()
    real_run_result_polling = scheduler.run_result_polling
    out: dict = {}

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with offline_harness(root) as h:
            set_dated_obs(h.picks_dir, now=NOW, streak=10, best=14, saver="active")
            # 1. Fresh choice via run_and_pick
            _, sel, _ = orchestrator.run_and_pick(h.config, DAY)
            daily = sel.pick_result.daily
            expected_rec = copy.deepcopy(daily.chosen_on)
            assert expected_rec["required"] is True and expected_rec["kind"] == "dated"
            assert expected_rec["saver_available"] is True

            # 2. attach_provenance + save_pick
            picks.attach_provenance(daily)
            picks.save_pick(daily, h.picks_dir)
            assert picks.load_pick(DAY, h.picks_dir).chosen_on == expected_rec

            # 3. preview run afterwards when games are Warmup -> re-saves without overwriting live record!
            h.statuses[101] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
            h.statuses[102] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
            assert runner.invoke(cli.cli, [
                "preview", "--date", DAY, "--picks-dir", str(h.picks_dir),
                "--models-dir", h.config["orchestrator"]["models_dir"],
            ], catch_exceptions=False).exit_code == 0
            assert picks.load_pick(DAY, h.picks_dir).chosen_on == expected_rec

            # 4. scheduler delivery (delivery_attempted -> notification_sent)
            st = make_sched_state()
            loaded = picks.load_pick(DAY, h.picks_dir)
            with redirect_stderr(io.StringIO()):
                assert scheduler._deliver_and_lock_pick(loaded, h.config, h.picks_dir, st, DAY, "fallback") is True
            assert picks.load_pick(DAY, h.picks_dir).chosen_on == expected_rec

            # 5. save_nonterminal_result ("unresolved")
            scheduler.save_nonterminal_result(DAY, h.picks_dir, "unresolved")
            assert picks.load_pick(DAY, h.picks_dir).result == "unresolved"
            assert picks.load_pick(DAY, h.picks_dir).chosen_on == expected_rec

            # 6. run_result_polling final grading save
            with (
                patch.object(scheduler, "poll_game_result", return_value="final"),
                patch.object(picks, "resolve_daily_slot_results", return_value={"primary": "hit", "double_down": "hit"} if loaded.double_down else {"primary": "hit"}),
                redirect_stderr(io.StringIO()),
            ):
                poll_res = real_run_result_polling(101, DAY, h.picks_dir)
            assert poll_res == "final"
            graded = picks.load_pick(DAY, h.picks_dir)
            assert graded.result == "hit" and graded.chosen_on == expected_rec

            # 7. reconcile_results correction save (hit -> miss) AND slot_results update save
            next_morning = REAL_DATETIME.fromisoformat("2027-07-16T02:00:00-04:00")
            with patch.object(picks, "resolve_daily_slot_results", return_value={"primary": "miss", "double_down": "hit"} if loaded.double_down else {"primary": "miss"}):
                corr = picks.reconcile_results(h.picks_dir, lookback_days=2, clock=lambda: next_morning)
            assert len(corr) == 1
            reconciled = picks.load_pick(DAY, h.picks_dir)
            assert reconciled.result == "miss" and reconciled.chosen_on == expected_rec

            # 8. Scheduler archives (deferred_fallback, stale_pick, refused_delivery)
            for prefix in ("deferred_fallback", "stale_pick", "refused_delivery"):
                arch = scheduler._archive_and_remove_pick(
                    h.picks_dir, DAY, reconciled, prefix=prefix, key=prefix, at_key="at", reason="test"
                )
                arch_json = json.loads(arch.read_text())
                assert list(arch_json.keys())[-2:] == ["chosen_on", prefix]
                assert arch_json["chosen_on"] == expected_rec

            # 9. Shadow pick save, load (with omitted double_down/runner_up), and check-results shadow grading
            sh = copy.deepcopy(daily)
            sh.result = None
            sh.slot_results = None
            sh_path = picks.save_shadow_pick(sh, h.picks_dir)
            assert picks.load_shadow_pick(DAY, h.picks_dir).chosen_on == expected_rec
            with patch.object(picks, "resolve_daily_slot_results", return_value={"primary": "hit", "double_down": "hit"} if sh.double_down else {"primary": "hit"}):
                assert runner.invoke(cli.cli, [
                    "check-results", "--date", DAY, "--picks-dir", str(h.picks_dir),
                ], catch_exceptions=False).exit_code == 0
            assert picks.load_shadow_pick(DAY, h.picks_dir).result == "hit"
            assert picks.load_shadow_pick(DAY, h.picks_dir).chosen_on == expected_rec

    # 10. Old pick file without chosen_on: exact byte diff on load + save_pick, and never given a record on reuse!
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with offline_harness(root) as h:
            set_dated_obs(h.picks_dir, now=NOW, streak=7, best=9, saver="used")
            d = picks.DailyPick(
                date=DAY, run_time=NOW.isoformat(),
                pick=picks.pick_from_row(h.frame.iloc[0]), double_down=None, runner_up=None,
            )
            legacy_dict = {k: v for k, v in asdict(d).items() if k != "chosen_on"}
            legacy_bytes = json.dumps(legacy_dict, indent=2).encode("utf-8")
            pfile = h.picks_dir / f"{DAY}.json"
            pfile.write_bytes(legacy_bytes)

            loaded, held_bytes = picks.load_pick_bytes(DAY, h.picks_dir)
            assert held_bytes == legacy_bytes and loaded.chosen_on is None
            picks.save_pick(loaded, h.picks_dir)
            resaved_bytes = pfile.read_bytes()
            expected_resaved = legacy_bytes[:-2] + b',\n  "chosen_on": null\n}'
            assert resaved_bytes == expected_resaved

            # Reuse in Warmup across run_and_pick, preview, bts run (closed window) -> chosen_on stays None!
            pfile.write_bytes(legacy_bytes)
            h.statuses[101] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
            h.statuses[102] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
            h.should_post.return_value = False
            _, sel_r, _ = orchestrator.run_and_pick(h.config, DAY)
            assert sel_r.pick_result.daily.chosen_on is None
            assert runner.invoke(cli.cli, [
                "preview", "--date", DAY, "--picks-dir", str(h.picks_dir),
                "--models-dir", h.config["orchestrator"]["models_dir"],
            ], catch_exceptions=False).exit_code == 0
            assert picks.load_pick(DAY, h.picks_dir).chosen_on is None
            assert runner.invoke(cli.cli, [
                "run", "--date", DAY, "--picks-dir", str(h.picks_dir),
                "--models-dir", h.config["orchestrator"]["models_dir"],
            ], catch_exceptions=False).exit_code == 0
            assert picks.load_pick(DAY, h.picks_dir).chosen_on is None

    out["all_lifecycle_checks_passed"] = True
    return out


def run_item_8_readers() -> dict:
    """Exercise pick-file readers across src/ and scripts/ on valid, null, and absent chosen_on."""
    from bts import contest_state, picks, shadow_eval, web
    from bts.entry_receipt import EntryReceipt
    from bts.experiment.artifacts import load_production_pick_snapshot
    from bts.health import (
        calibration as h_cal,
        late_delivery as h_late,
        mdp_policy_alignment as h_mdp,
        pitcher_sparsity as h_ps,
        post_failure as h_pf,
        predicted_vs_realized as h_pvr,
        projected_lineup as h_proj,
        result_resolution as h_rr,
        same_team_corr as h_stc,
    )
    import pandas as pd
    from bts.model.calibrate import _resolve_pick_outcomes
    from bts.state.export import export_initial_state
    from scripts import (
        audit_mdp_live_aggressiveness,
        canonicalize_realized_picks,
        live_forward_capture_once,
        live_forward_provenance_inventory,
        measure_raw_rebin_gate,
        production_live_boundary_feasibility,
    )
    from scripts.audit import boundary_shadow_census
    from scripts.audit.season_ledger.sources.pick_files import parse_pick_file

    valid_rec = {
        "schema": "bts_pick_chosen_on_v1",
        "required": True,
        "kind": "dated",
        "streak": 7,
        "saver_available": False,
        "dated": {"effective_date": DAY, "active_streak": 7, "best_streak": 9, "saver_state": "used"},
    }
    base_pick = picks.DailyPick(
        date=DAY,
        run_time=NOW.isoformat(),
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
                        0.82, [], False, 101, f"{DAY}T23:10:00Z", pitcher_team="NYY"),
        double_down=None,
        runner_up={"batter_name": "Juan Soto", "p_game_hit": 0.81},
        bluesky_posted=True,
        bluesky_uri="at://post/1",
        delivered_at=NOW.isoformat(),
        result="hit",
        slot_results={"primary": "hit"},
        shadow_model_version="context_lgbm_v2",
    )

    variants = {}
    for vname in ("absent", "null", "valid"):
        body = asdict(base_pick)
        if vname == "absent":
            body.pop("chosen_on", None)
        elif vname == "null":
            body["chosen_on"] = None
        else:
            body["chosen_on"] = copy.deepcopy(valid_rec)
        variants[vname] = body

    reader_outputs = {}
    for vname, body in variants.items():
        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            raw_bytes = json.dumps(body, indent=2).encode("utf-8")
            pfile = pdir / f"{DAY}.json"
            pfile.write_bytes(raw_bytes)
            (pdir / f"{DAY}.shadow.json").write_bytes(raw_bytes)
            (pdir / "streak.json").write_text(json.dumps({"season": 2027, "streak": 1, "saver_available": True}))

            lp = picks.load_pick(DAY, pdir)
            lpb, held = picks.load_pick_bytes(DAY, pdir)
            lsp = picks.load_shadow_pick(DAY, pdir)
            assert held == raw_bytes
            assert lp.chosen_on == (valid_rec if vname == "valid" else None)
            assert lpb.chosen_on == (valid_rec if vname == "valid" else None)
            assert lsp.chosen_on == (valid_rec if vname == "valid" else None)

            # Exercising readers across src and scripts
            latest_res = contest_state.latest_resolved_pick_date(pdir)
            gap_res = contest_state.resolved_pick_settlement_gap(pdir, dt_mod.date(2027, 7, 14))
            miss_res = contest_state._has_unconfirmed_miss(pdir, dt_mod.date(2027, 7, 14))
            replay_res = picks._replay_season_streak(pdir, 2027, "2027-07-16")
            pa_df = pd.DataFrame([{"batter_id": 680757, "date": DAY, "is_hit": 1}])
            cal_samples = _resolve_pick_outcomes(pdir, pa_df, dt_mod.date(2027, 7, 15), 7)
            snap = load_production_pick_snapshot(pfile, expected_date=DAY)
            exp = export_initial_state(pdir, pdir / "sub" / "export.json")

            # Health checks
            from bts.simulate.mdp import DEFAULT_POLICY_PATH
            today_d = dt_mod.date(2027, 7, 15)
            h_results = {
                "cal": [(a.level, a.source) for a in h_cal.check(pdir, today=today_d)],
                "mdp": [(a.level, a.source) for a in h_mdp.check(pdir, DEFAULT_POLICY_PATH, today=today_d)],
                "stc": [(a.level, a.source) for a in h_stc.check(pdir, today=today_d)],
                "pvr": [(a.level, a.source) for a in h_pvr.check(pdir, today=today_d)],
                "proj": [(a.level, a.source) for a in h_proj.check(pdir, today=today_d)],
                "ps": [(a.level, a.source) for a in h_ps.check(pdir, today=today_d)],
                "late": [(a.level, a.source) for a in h_late.check(pdir, today=today_d, now=NOW)],
                "pf": [(a.level, a.source) for a in h_pf.check(pdir, today=today_d, now=NOW)],
                "rr": [(a.level, a.source) for a in h_rr.check(pdir, today=today_d)],
            }
            sh_status = shadow_eval.build_shadow_cycle_status(pdir)

            # Receipt & ledger
            rec = EntryReceipt(et_date=DAY, started_at=NOW, clock=lambda: NOW,
                               expected_username=None, picks_dir=pdir)
            rec.selection(lpb, held, None, None)
            rec_sel = rec.record()["selection"]
            ledger_row, = parse_pick_file(pfile.name, raw_bytes).rows
            assert rec_sel["pick_file_sha256"] == ledger_row["content_sha256"] == hashlib.sha256(raw_bytes).hexdigest()

            # Analysis scripts & dashboard
            slot_rows = production_live_boundary_feasibility.load_pick_slot_rows(pdir, today=dt_mod.date(2027, 7, 16))
            with patch.object(measure_raw_rebin_gate, "_day_hit_lookup", return_value={(680757, dt_mod.date(2027, 7, 15)): 1}):
                rebin_rows = measure_raw_rebin_gate.load_resolved_pair_rows(pdir, pdir / "unused.parquet", dt_mod.date(2027, 7, 16))
            aggr_rows = audit_mdp_live_aggressiveness.load_live_pick_rows(pdir, today=dt_mod.date(2027, 7, 16))
            census_ps = boundary_shadow_census._flat_primary_ps(pdir)
            canon_rows = canonicalize_realized_picks.extract_picks(body, pfile.name)
            with patch.object(web, "PICKS_DIR", pdir):
                web_hist = [{k: v for k, v in row.items() if k != "chosen_on"} for row in web.load_all_picks()]

            reader_outputs[vname] = {
                "latest_res": latest_res.isoformat(),
                "gap_res": gap_res,
                "miss_res": miss_res,
                "replay_res": replay_res,
                "cal_rows": len(cal_samples),
                "snap_slots": snap["slots"],
                "exp_picks": [{k: v for k, v in p.items() if k != "exported_at"} for p in exp["historical_picks"]],
                "h_results": h_results,
                "sh_status": {k: v for k, v in sh_status.items() if k not in {"generated_at", "picks_dir"}},
                "slot_rows": len(slot_rows),
                "rebin_rows": len(rebin_rows),
                "aggr_rows": len(aggr_rows),
                "census_count": len(census_ps),
                "canon_rows": len(canon_rows),
                "web_hist": web_hist,
                "sha256": rec_sel["pick_file_sha256"],
            }

    # Verify all non-hash reader outputs are identical across absent, null, and valid!
    for vname in ("null", "valid"):
        a = {k: v for k, v in reader_outputs["absent"].items() if k != "sha256"}
        b = {k: v for k, v in reader_outputs[vname].items() if k != "sha256"}
        assert a == b, (vname, a, b)
    assert len({reader_outputs[v]["sha256"] for v in ("absent", "null", "valid")}) == 3

    # Verify both live-forward scripts end-to-end (decision_snapshot AND manifest alignment helpers!)
    cap_abs = live_forward_capture_once.decision_snapshot(variants["absent"])
    cap_null = live_forward_capture_once.decision_snapshot(variants["null"])
    cap_val = live_forward_capture_once.decision_snapshot(variants["valid"])
    inv_abs = live_forward_provenance_inventory._decision_snapshot(variants["absent"])
    inv_null = live_forward_provenance_inventory._decision_snapshot(variants["null"])
    inv_val = live_forward_provenance_inventory._decision_snapshot(variants["valid"])
    assert cap_abs == cap_null and cap_abs != cap_val
    assert inv_abs == inv_null and inv_abs != inv_val

    valid_opt = copy.deepcopy(variants["valid"])
    valid_opt["chosen_on"]["required"] = False
    assert live_forward_capture_once.decision_snapshot(valid_opt) != cap_val
    assert live_forward_provenance_inventory._decision_snapshot(valid_opt) != inv_val

    # Test existing_snapshot_state and _pick_alignment_summary when legacy file (absent) is re-saved with null!
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        pfile = d / f"{DAY}.json"
        pfile.write_text(json.dumps(variants["absent"], indent=2))
        snap_absent = load_production_pick_snapshot(pfile, expected_date=DAY)
        manifest_path = d / "manifest.json"
        manifest_path.write_text(json.dumps({"production_pick_snapshot": snap_absent}, indent=2))

        # Re-save with chosen_on: null -> byte sha differs, decision_matches_current_pick stays True!
        pfile.write_text(json.dumps(variants["null"], indent=2))
        st_null = live_forward_capture_once.existing_snapshot_state(
            manifest_path=manifest_path, pick_path=pfile, expected_date=DAY
        )
        align_null = live_forward_provenance_inventory._current_pick_snapshot_summary(
            snapshot={**snap_absent, "present": True}, picks_dir=d, date_key=DAY
        )
        assert st_null["snapshot_matches_current_pick"] is False
        assert st_null["snapshot_decision_matches_current_pick"] is True
        assert st_null["stale_pick_snapshot"] is False
        assert align_null["matches_current_pick"] is False
        assert align_null["decision_matches_current_pick"] is True

        # Re-save with non-null chosen_on -> decision_matches_current_pick becomes False (drift detected)!
        pfile.write_text(json.dumps(variants["valid"], indent=2))
        st_val = live_forward_capture_once.existing_snapshot_state(
            manifest_path=manifest_path, pick_path=pfile, expected_date=DAY
        )
        align_val = live_forward_provenance_inventory._current_pick_snapshot_summary(
            snapshot={**snap_absent, "present": True}, picks_dir=d, date_key=DAY
        )
        assert st_val["snapshot_decision_matches_current_pick"] is False
        assert st_val["stale_pick_snapshot"] is True
        assert align_val["decision_matches_current_pick"] is False

    return {
        "readers_verified": 21,
        "live_forward_manifest_alignment_verified": True,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

    trap_files, file_opens = install_network_guard()
    r3 = run_item_3_oracle(trap_files, file_opens)
    r4 = run_item_4_record_lifecycle()
    r8 = run_item_8_readers()
    summary = {"item_3": r3, "item_4": r4, "item_8": r8, "status": "ALL_ASSERTIONS_PASSED"}
    Path(__file__).with_name("probe_3_4_8_results.json").write_text(json.dumps(summary, indent=2))
    print(f"probe_3_4_8 PASSED: {r3['corpus_size']} record shapes ({r3['pairs_checked']} pairs), "
          f"full lifecycle & shadow verified, {r8['readers_verified']} readers & live-forward alignment verified.")


if __name__ == "__main__":
    main()
