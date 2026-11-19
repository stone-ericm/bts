"""Probe for Item 3 (Codex 630 + Carbon 435 corpora on Base c2039541 vs R2 aa61bb43 + independent fold)
and Item 4 (complete 4,480-cell Rule 2 truth table vs Amendment 3 + test coverage audit).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from io import BytesIO
import itertools
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ET = ZoneInfo("America/New_York")
RUN_ET = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _archive_src(rev: str, dest: Path) -> Path:
    raw = subprocess.check_output(["git", "archive", rev, "src"], cwd=REPO, timeout=20)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(raw)) as tf:
        tf.extractall(dest, filter="data")
    return dest / "src"


# ---------------------------------------------------------------------------
# Clean-room from-scratch fold implementing Amendment 3 (Rules 1 & 2)
# ---------------------------------------------------------------------------
def _scratch_read_regular(path: Path) -> str | None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(mode):
        raise OSError("non-regular")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError("non-regular fd")
        with open(fd, "r", encoding="utf-8", closefd=False) as fh:
            return fh.read()
    finally:
        os.close(fd)


def _scratch_valid_chosen_on_preview(val: object) -> bool:
    """True iff val is a valid bts_pick_chosen_on_v1 dict with required == False."""
    req_keys = {"schema", "required", "kind", "streak", "saver_available", "dated"}
    if type(val) is not dict or set(val.keys()) != req_keys:
        return False
    if val["schema"] != "bts_pick_chosen_on_v1" or type(val["required"]) is not bool:
        return False
    if val["kind"] not in {"dated", "automatic", "legacy_manual", "model"}:
        return False
    if type(val["streak"]) is not int or val["streak"] < 0 or type(val["saver_available"]) is not bool:
        return False
    if val["kind"] in {"dated", "automatic"} and val["streak"] > 57:
        return False
    if val["kind"] != "dated":
        if val["dated"] is not None:
            return False
    else:
        d = val["dated"]
        dkeys = {"effective_date", "active_streak", "best_streak", "saver_state"}
        if type(d) is not dict or set(d.keys()) != dkeys:
            return False
        if (type(d["effective_date"]) is not str or type(d["active_streak"]) is not int
                or type(d["best_streak"]) is not int or d["active_streak"] != val["streak"]
                or not (0 <= d["active_streak"] <= d["best_streak"] <= 57)
                or d["saver_state"] not in {"not_earned", "active", "used"}):
            return False
    return val["required"] is False


def scratch_fold_season(picks_dir: Path, season: int, today: str) -> tuple[int, bool] | None:
    """Independent from-scratch fold over picks_dir for (season, today) under Amendment 3."""
    # Read streak markers
    sp = picks_dir / "streak.json"
    st_text = _scratch_read_regular(sp)  # raises OSError if non-regular (matches NonRegularReplayFile)
    markers: list[str] = []
    if st_text is not None:
        try:
            st_obj = json.loads(st_text)
        except Exception:
            st_obj = None
        if isinstance(st_obj, dict):
            if "season" not in st_obj or (type(st_obj["season"]) is int and st_obj["season"] == season):
                raw_m = st_obj.get("applied_dates")
                if isinstance(raw_m, list):
                    markers = [x for x in raw_m if isinstance(x, str)]

    dates = {p.stem for p in picks_dir.glob(f"{season}-*.json")}
    dates.update(p.parent.name for p in picks_dir.glob(f"{season}-*/decision.json"))

    pick_keys = {
        "batter_name", "batter_id", "team", "lineup_position", "pitcher_name",
        "pitcher_id", "p_game_hit", "flags", "projected_lineup", "game_pk",
        "game_time", "pitcher_team",
    }
    req_pick_keys = pick_keys - {"pitcher_team"}

    streak, saver = 0, True
    for day in sorted(dates):
        if day > today or not ISO_RE.fullmatch(day):
            continue
        # 1. Read pick
        try:
            ptext = _scratch_read_regular(picks_dir / f"{day}.json")
            pdata = None if ptext is None else json.loads(ptext)
            if ptext is not None and not isinstance(pdata, dict):
                return None
        except Exception:
            return None

        has_nonterminal_pick = False
        if pdata is not None:
            r = pdata.get("result")
            if r == "void":
                continue
            if r == "miss":
                if saver and 10 <= streak <= 15:
                    saver = False
                else:
                    streak = 0
                continue
            if r == "hit":
                # Legacy load_pick requirements
                try:
                    for topk in ("date", "run_time", "pick", "double_down", "runner_up"):
                        if topk not in pdata:
                            return None
                    p1 = dict(pdata["pick"])
                    p1.setdefault("pitcher_team", None)
                    if set(p1.keys()) != pick_keys:
                        return None
                    dd = pdata["double_down"]
                    dd_present = bool(dd)
                    if dd_present:
                        dd2 = dict(dd)
                        dd2.setdefault("pitcher_team", None)
                        if set(dd2.keys()) != pick_keys:
                            return None
                    sr = pdata.get("slot_results")
                    if sr is not None:
                        inc = sum(1 for v in sr.values() if v == "hit")
                    else:
                        inc = 2 if dd_present else 1
                except Exception:
                    return None
                streak += inc
                continue

            # Nonterminal pick validation
            has_nonterminal_pick = True
            try:
                if not isinstance(pdata.get("pick"), dict):
                    return None
                if pdata.get("double_down") is not None and not isinstance(pdata["double_down"], dict):
                    return None
                for topk in ("date", "run_time", "pick", "double_down", "runner_up"):
                    if topk not in pdata:
                        return None
                if pdata["date"] != day or not isinstance(pdata["run_time"], str):
                    return None
                if r not in (None, "suspended", "unresolved"):
                    return None
                for bkey in ("bluesky_posted", "notification_sent", "delivery_attempted"):
                    if type(pdata.get(bkey, False)) is not bool:
                        return None
                if pdata.get("notification_id") is not None and not isinstance(pdata["notification_id"], str):
                    return None
                sr = pdata.get("slot_results")
                if sr is not None:
                    if not isinstance(sr, dict) or any(v not in ("hit", "miss", "void") for v in sr.values()):
                        return None
                slots_to_check = [pdata["pick"]]
                if pdata["double_down"]:
                    slots_to_check.append(pdata["double_down"])
                for sl in slots_to_check:
                    ks = set(sl.keys())
                    if not req_pick_keys.issubset(ks) or not ks.issubset(pick_keys):
                        return None
                    if (type(sl["batter_id"]) is not int or not isinstance(sl["batter_name"], str)
                            or not isinstance(sl["team"], str)
                            or (sl["game_pk"] is not None and type(sl["game_pk"]) is not int)):
                        return None
            except Exception:
                return None

        # 2. Read decision
        try:
            dtext = _scratch_read_regular(picks_dir / day / "decision.json")
            dec = None
            if dtext is not None:
                dobj = json.loads(dtext)
                if (not isinstance(dobj, dict)
                        or dobj.get("schema_version") not in (
                            "bts_daily_decision_v1", "bts_daily_decision_v2", "bts_daily_decision_v3")
                        or dobj.get("action") not in {"skip", "single", "double"}
                        or type(dobj.get("scoreable")) is not bool
                        or dobj.get("date") != day):
                    return None
                dec = dobj
        except Exception:
            return None

        if not has_nonterminal_pick:
            if dec is not None and dec["scoreable"]:
                return None
            continue

        if day in markers:
            return None

        if day < today:
            if dec is not None and dec["scoreable"]:
                return None
            if any(pdata.get(k) is not None and not isinstance(pdata.get(k), str)
                   for k in ("bluesky_uri", "delivered_at")):
                return None
            sig = any((
                pdata.get("bluesky_posted", False),
                pdata.get("bluesky_uri"),
                pdata.get("notification_sent", False),
                pdata.get("notification_id"),
                pdata.get("delivery_attempted", False),
                pdata.get("delivered_at"),
            ))
            if sig:
                return None
            try:
                stext = _scratch_read_regular(picks_dir / day / "scheduler_state.json")
                if stext is not None:
                    sobj = json.loads(stext)
                    if (not isinstance(sobj, dict) or sobj.get("date") != day
                            or type(sobj.get("pick_locked")) is not bool):
                        return None
                    if sobj["pick_locked"]:
                        return None
            except Exception:
                return None
            if dec is None and not _scratch_valid_chosen_on_preview(pdata.get("chosen_on")):
                return None
    return (streak, saver)


# ---------------------------------------------------------------------------
# Worker process for replay evaluation
# ---------------------------------------------------------------------------
def _replay_worker(src_dir: str) -> int:
    sys.path.insert(0, src_dir)
    import bts.picks as picks

    def _deny_net(*args, **kwargs):
        raise AssertionError("Network forbidden")

    socket.socket.connect = _deny_net  # type: ignore[method-assign]
    socket.getaddrinfo = _deny_net  # type: ignore[assignment]

    class _Timeout(BaseException):
        pass

    signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_Timeout()))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        signal.alarm(3)
        try:
            val = picks._replay_season_streak(Path(req["dir"]), req["season"], req["today"])
        except _Timeout:
            out = {"status": "timeout"}
        except Exception as exc:  # noqa: BLE001
            out = {"status": "raised", "exc": type(exc).__name__}
        else:
            out = {"status": "none"} if val is None else {"status": "ok", "value": [val[0], val[1]]}
        finally:
            signal.alarm(0)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


def _run_corpora(scratch: Path) -> dict:
    base_src = _archive_src("c2039541", scratch / "base_c2039541")
    head_src = REPO / "src"

    env_base = os.environ | {"PYTHONPATH": str(base_src), "PYTHONDONTWRITEBYTECODE": "1", "TZ": "America/New_York"}
    env_head = os.environ | {"PYTHONPATH": str(head_src), "PYTHONDONTWRITEBYTECODE": "1", "TZ": "America/New_York"}

    p_base = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--replay-worker", str(base_src)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env_base,
    )
    p_head = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--replay-worker", str(head_src)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env_head,
    )

    def _ask(proc, d: Path, season: int, today: str) -> dict:
        assert proc.stdin and proc.stdout
        proc.stdin.write(json.dumps({"dir": str(d), "season": season, "today": today}) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline())

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(head_src))
    sys.path.insert(0, str(HERE / "round1"))
    import tests.test_wr_c3_corpus as codex_corpus
    import probe_corpus_and_fold as carbon_corpus

    results = {}
    try:
        # 1. Codex 630 histories
        codex_diffs_base_ok = []
        codex_diffs_head_ok = []
        codex_all_diffs = Counter()
        codex_scratch_mismatches = []
        codex_total = 0
        codex_equal = 0
        for case in codex_corpus.cases():
            for idx in range(len(codex_corpus.ORDINARY)):
                codex_total += 1
                hdir = scratch / "codex_h" / f"{case.name}_{idx}"
                hdir.mkdir(parents=True)
                oracle = codex_corpus.build_history(hdir, case, idx)
                b = _ask(p_base, hdir, 2026, codex_corpus.TODAY)
                h = _ask(p_head, hdir, 2026, codex_corpus.TODAY)
                sf = scratch_fold_season(hdir, 2026, codex_corpus.TODAY)
                sf_norm = {"status": "none"} if sf is None else {"status": "ok", "value": [sf[0], sf[1]]}
                if h != sf_norm:
                    codex_scratch_mismatches.append({"name": f"{case.name}-{idx}", "head": h, "scratch": sf_norm})
                if b == h:
                    codex_equal += 1
                else:
                    codex_all_diffs[case.name] += 1
                    if b["status"] == "ok":
                        codex_diffs_base_ok.append({
                            "name": f"{case.name}-{idx}", "case": case.name,
                            "base": b, "head": h, "scratch": sf_norm,
                        })
                    if h["status"] == "ok" and b["status"] != "ok":
                        codex_diffs_head_ok.append({
                            "name": f"{case.name}-{idx}", "case": case.name,
                            "base": b, "head": h, "oracle": oracle, "scratch": sf_norm,
                            "matches_oracle_and_scratch": (h["value"] == oracle and h == sf_norm),
                        })

        results["codex_630"] = {
            "total": codex_total,
            "equal": codex_equal,
            "diff_count": codex_total - codex_equal,
            "diff_by_case": dict(codex_all_diffs),
            "base_returned_value_but_head_differed": codex_diffs_base_ok,
            "base_unavailable_or_raised_and_head_returned_value": codex_diffs_head_ok,
            "scratch_mismatches": codex_scratch_mismatches,
        }

        # 2. Carbon 435 histories
        carbon_cases = carbon_corpus._build_deterministic_cases() + carbon_corpus._build_seeded_cases()
        carbon_diffs_base_ok = []
        carbon_diffs_head_ok = []
        carbon_all_diffs = []
        carbon_scratch_mismatches = []
        carbon_equal = 0
        for idx, spec in enumerate(carbon_cases):
            hdir = scratch / "carbon_h" / f"c_{idx:04d}"
            hdir.mkdir(parents=True)
            spec.builder(hdir)
            b = _ask(p_base, hdir, spec.season, spec.today)
            h = _ask(p_head, hdir, spec.season, spec.today)
            sf = scratch_fold_season(hdir, spec.season, spec.today)
            sf_norm = {"status": "none"} if sf is None else {"status": "ok", "value": [sf[0], sf[1]]}
            if h != sf_norm:
                carbon_scratch_mismatches.append({"name": spec.name, "head": h, "scratch": sf_norm})
            if b == h:
                carbon_equal += 1
            else:
                carbon_all_diffs.append({"name": spec.name, "base": b, "head": h, "note": spec.note})
                if b["status"] == "ok":
                    carbon_diffs_base_ok.append({
                        "name": spec.name, "base": b, "head": h, "scratch": sf_norm, "note": spec.note,
                    })
                if h["status"] == "ok" and b["status"] != "ok":
                    carbon_diffs_head_ok.append({
                        "name": spec.name, "base": b, "head": h, "scratch": sf_norm,
                        "matches_scratch": (h == sf_norm), "note": spec.note,
                    })

        results["carbon_435"] = {
            "total": len(carbon_cases),
            "equal": carbon_equal,
            "diff_count": len(carbon_all_diffs),
            "all_diffs": carbon_all_diffs,
            "base_returned_value_but_head_differed": carbon_diffs_base_ok,
            "base_unavailable_or_raised_and_head_returned_value": carbon_diffs_head_ok,
            "scratch_mismatches": carbon_scratch_mismatches,
        }
    finally:
        for p in (p_base, p_head):
            if p.stdin:
                p.stdin.close()
            p.wait(timeout=5)
    return results


# ---------------------------------------------------------------------------
# Item 4: Full 4,480-cell Rule 2 table audit
# ---------------------------------------------------------------------------
def _expected_amendment3_cell(dec: str, sig: str, chosen: str, sstate: str, marker: str, day_rel: str) -> str | None:
    """Independent specification of Amendment 3 for all 4,480 cells.

    Returns None for skip (0 contribution), or the expected dated blocker reason.
    """
    # Step 1: Decision record read runs unconditionally for nonterminal picks (past or today)
    if dec == "unreadable":
        return "unreadable or invalid decision record"
    # Step 2: Applied-date marker check runs before past/today split (same_season or tagless holds; other_season ignored)
    if marker in ("same_season", "tagless"):
        return "applied-date marker without a result"
    # Step 3: Today's unmarked unresolved pick (with readable/absent decision) is outside today's fold -> skip (None)
    if day_rel == "today":
        return None
    # Step 4: Past unresolved pick -> Rule 2
    if dec == "scoreable":
        return "committed pick awaiting a result"
    if sig != "none":
        if dec in ("skip", "non_scoreable_other"):
            return "skip record contradicts a delivery"
        if sig == "delivery_attempted":
            return "delivery attempt without a decision or result"
        return "committed pick awaiting a result"
    if sstate == "unreadable":
        return "unreadable or invalid scheduler state"
    if sstate == "locked":
        return "scheduler state marks the pick locked"
    # No delivery signal, no lock, readable/absent state, non-scoreable or absent decision:
    if dec in ("skip", "non_scoreable_other"):
        return None
    # dec == "absent"
    if chosen == "required_false":
        return None
    return "no decision or preview stamp"


def _run_rule2_table(scratch: Path) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from bts import picks
    from bts.contest_state import DecisionStreakState
    from bts.daily_decision import decision_path, write_decision

    decisions = ("absent", "skip", "non_scoreable_other", "scoreable", "unreadable")
    signals = (
        "none", "bluesky_posted", "bluesky_uri", "notification_sent",
        "notification_id", "delivery_attempted", "delivered_at",
    )
    chosens = ("absent", "required_true", "required_false", "invalid")
    sstates = ("absent", "unlocked", "locked", "unreadable")
    markers = ("none", "same_season", "tagless", "other_season")
    days = ("past", "today")

    today_iso = "2027-04-20"
    past_iso = "2027-04-16"
    model = DecisionStreakState(0, True, True, "model", "fresh", 0, True)

    base_slot = {
        "batter_name": "Synthetic batter", "batter_id": 1, "team": "X", "lineup_position": 1,
        "pitcher_name": "Synthetic pitcher", "pitcher_id": 2, "p_game_hit": 0.8, "flags": [],
        "projected_lineup": False, "game_pk": 1, "game_time": "2027-04-16T23:00:00Z",
        "pitcher_team": None,
    }

    # Pre-build decision payloads for speed
    tmp_dec_dir = scratch / "_dec_templates"
    tmp_dec_dir.mkdir()
    dec_templates: dict[tuple[str, str], str] = {}
    for d_iso in (past_iso, today_iso):
        write_decision(d_iso, tmp_dec_dir, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
        dec_templates[("skip", d_iso)] = decision_path(d_iso, tmp_dec_dir).read_text()
        write_decision(d_iso, tmp_dec_dir, action="single", source="test", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
        dec_templates[("non_scoreable_other", d_iso)] = decision_path(d_iso, tmp_dec_dir).read_text()
        write_decision(d_iso, tmp_dec_dir, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN_ET)
        dec_templates[("scoreable", d_iso)] = decision_path(d_iso, tmp_dec_dir).read_text()

    chosen_payloads = {
        "absent": None,
        "required_true": picks.chosen_on_record(model, required=True),
        "required_false": picks.chosen_on_record(model, required=False),
        "invalid": {"schema": "bad"},
    }

    mismatches = []
    reason_counts = Counter()
    total_cells = 0

    cell_root = scratch / "cells"
    cell_root.mkdir()

    for idx, (dec, sig, chosen, sstate, marker, day_rel) in enumerate(
        itertools.product(decisions, signals, chosens, sstates, markers, days)
    ):
        total_cells += 1
        d_iso = past_iso if day_rel == "past" else today_iso
        cdir = cell_root / f"c{idx}"
        cdir.mkdir()

        pdata = {
            "date": d_iso,
            "run_time": f"{d_iso}T19:00:00Z",
            "pick": dict(base_slot, game_time=f"{d_iso}T23:00:00Z"),
            "double_down": None,
            "runner_up": None,
            "bluesky_posted": False,
            "bluesky_uri": None,
            "notification_sent": False,
            "notification_channel": None,
            "notification_id": None,
            "delivery_attempted": False,
            "delivered_at": None,
            "result": None,
            "slot_results": None,
            "chosen_on": chosen_payloads[chosen],
        }
        if sig != "none":
            pdata[sig] = True if sig in ("bluesky_posted", "notification_sent", "delivery_attempted") else "syn"
        (cdir / f"{d_iso}.json").write_text(json.dumps(pdata))

        if dec != "absent":
            ddir = cdir / d_iso
            ddir.mkdir(exist_ok=True)
            (ddir / "decision.json").write_text("{" if dec == "unreadable" else dec_templates[(dec, d_iso)])

        if sstate != "absent":
            ddir = cdir / d_iso
            ddir.mkdir(exist_ok=True)
            if sstate == "unreadable":
                (ddir / "scheduler_state.json").write_text("{")
            else:
                (ddir / "scheduler_state.json").write_text(
                    json.dumps({"date": d_iso, "pick_locked": sstate == "locked"})
                )

        st_dict: dict = {}
        if marker == "same_season":
            st_dict = {"streak": 5, "saver_available": True, "season": 2027, "applied_dates": [d_iso]}
        elif marker == "tagless":
            st_dict = {"streak": 5, "saver_available": True, "applied_dates": [d_iso]}
        elif marker == "other_season":
            st_dict = {"streak": 5, "saver_available": True, "season": 2026, "applied_dates": [d_iso]}
        if st_dict:
            (cdir / "streak.json").write_text(json.dumps(st_dict))

        state_read = picks._read_reconcile_streak(cdir / "streak.json")
        m_list = picks._replay_markers(state_read or {}, 2027)
        val, blocker = picks._fold_season(cdir, 2027, today_iso, m_list)
        actual_reason = None if val == (0, True) and blocker is None else blocker[1]
        if blocker is not None:
            assert blocker[0] == d_iso

        expected_reason = _expected_amendment3_cell(dec, sig, chosen, sstate, marker, day_rel)
        reason_counts[str(actual_reason)] += 1
        if actual_reason != expected_reason:
            mismatches.append({
                "cell": [dec, sig, chosen, sstate, marker, day_rel],
                "actual": actual_reason,
                "expected": expected_reason,
            })

    # Audit which cells tests/test_wr_c3_*.py actually exercise
    # Map the 23 rows of test_past_pick_requires_positive_skip_evidence + other tests in test_wr_c3_*.py
    tested_cells: set[tuple[str, str, str, str, str, str]] = set()
    table_23 = [
        ("absent", "none", "absent", "absent", "none", "past"),
        ("absent", "none", "invalid", "absent", "none", "past"),
        ("absent", "none", "required_true", "absent", "none", "past"),
        ("absent", "none", "required_false", "absent", "none", "past"),
        ("absent", "none", "required_false", "unlocked", "none", "past"),
        ("skip", "none", "absent", "absent", "none", "past"),
        ("non_scoreable_other", "none", "absent", "unlocked", "none", "past"),
        ("scoreable", "none", "required_false", "absent", "none", "past"),
        ("skip", "none", "required_false", "absent", "same_season", "past"),
        ("absent", "notification_id", "required_false", "absent", "none", "past"),
        ("absent", "delivery_attempted", "required_false", "absent", "none", "past"),
        ("skip", "bluesky_posted", "absent", "absent", "none", "past"),
        ("skip", "bluesky_uri", "absent", "absent", "none", "past"),
        ("skip", "notification_sent", "absent", "absent", "none", "past"),
        ("skip", "notification_id", "absent", "absent", "none", "past"),
        ("skip", "delivery_attempted", "absent", "absent", "none", "past"),
        ("skip", "delivered_at", "absent", "absent", "none", "past"),
        ("absent", "none", "required_false", "locked", "none", "past"),
        ("skip", "none", "absent", "locked", "none", "past"),
        ("skip", "none", "absent", "unreadable", "none", "past"),
        ("unreadable", "none", "required_false", "absent", "none", "past"),
    ]
    tested_cells.update(table_23)
    # Other tests in test_wr_c3_reconcile.py / corpus / delivery:
    tested_cells.update([
        # test_reconcile_names_first_blocking_date
        ("absent", "notification_sent", "absent", "absent", "none", "past"),  # delivered uses sent+id+attempted
        ("scoreable", "none", "absent", "absent", "none", "past"),  # private
        ("absent", "delivery_attempted", "absent", "absent", "none", "past"),  # attempt
        ("unreadable", "none", "absent", "absent", "none", "past"),  # decision_json / decision_fifo
        # scorer_crash_marker_holds_across_midnight (past: 2027-04-19 when RUN=2027-04-20)
        ("absent", "none", "absent", "absent", "same_season", "past"),
        ("absent", "none", "absent", "absent", "tagless", "past"),
        # test_new_year_uses_locked_day_and_ignores_foreign_marker (today: 2027-01-01)
        ("absent", "none", "absent", "absent", "other_season", "today"),
        # corpus cases:
        ("absent", "bluesky_posted", "absent", "absent", "none", "today"),  # today_committed_unresolved
        ("absent", "none", "absent", "absent", "same_season", "today"),     # today_marker_matched
        ("absent", "none", "absent", "absent", "tagless", "today"),         # today_marker_tagless
        ("absent", "none", "absent", "absent", "other_season", "past"),     # past_marker_foreign
        # delivery tests:
        ("absent", "none", "required_true", "unlocked", "none", "past"),    # manual_run_crash (no state) / lost_ack
        ("absent", "none", "absent", "locked", "none", "past"),             # private_crash
    ])

    return {
        "total_cells": total_cells,
        "amendment3_mismatches": mismatches,
        "outcome_distribution": dict(reason_counts),
        "explicitly_tested_cell_count": len(tested_cells),
        "untested_cell_count": total_cells - len(tested_cells),
        "notable_untested_slices": {
            "non_scoreable_other_with_each_signal": "0/6 tested (only skip + signal is tested; action='single', scoreable=False + signal untested)",
            "absent_decision_with_individual_signals": "2/6 tested alone (only notification_id and delivery_attempted tested alone with decision=None; bluesky_posted, bluesky_uri, notification_sent, delivered_at alone with decision=None untested in table)",
            "past_tagless_or_other_season_marker_with_skip_or_preview_evidence": "0 tested (past_marker_foreign in corpus had no skip evidence so blocked on 'no decision or preview stamp', never testing that foreign marker does NOT block a valid preview/skip!)",
            "today_with_unreadable_decision": "0 tested (blocks replay even on today)",
            "today_with_unreadable_or_locked_scheduler_state": "0 tested (skipped on today without reading scheduler_state.json)",
            "today_with_skip_contradicting_delivery": "0 tested (skipped on today; only blocks once day < today)",
            "skip_decision_with_chosen_on_required_true_or_invalid": "0 tested in table (skip + required_true and skip + invalid chosen_on both skip)",
        },
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--replay-worker":
        return _replay_worker(sys.argv[2])

    with TemporaryDirectory(dir=HERE, prefix="_p34_") as td:
        scratch = Path(td)
        corpora = _run_corpora(scratch)
        table = _run_rule2_table(scratch)

    out = {"corpora": corpora, "rule2_table": table}
    (HERE / "item3_and_4_results.json").write_text(json.dumps(out, indent=2) + "\n")

    c630 = corpora["codex_630"]
    c435 = corpora["carbon_435"]
    print("=== CODEX 630 CORPUS ===")
    print(f"Total: {c630['total']}, Equal: {c630['equal']}, Diffs: {c630['diff_count']}, Scratch mismatches: {len(c630['scratch_mismatches'])}")
    print("Diffs by case:", json.dumps(c630["diff_by_case"], sort_keys=True))
    print("Base returned value but HEAD differed (cases):", sorted({x["case"] for x in c630["base_returned_value_but_head_differed"]}))
    print("Base unavailable/raised -> HEAD value (all match oracle+scratch?):",
          len(c630["base_unavailable_or_raised_and_head_returned_value"]),
          all(x["matches_oracle_and_scratch"] for x in c630["base_unavailable_or_raised_and_head_returned_value"]))

    print("\n=== CARBON 435 CORPUS ===")
    print(f"Total: {c435['total']}, Equal: {c435['equal']}, Diffs: {c435['diff_count']}, Scratch mismatches: {len(c435['scratch_mismatches'])}")
    print("Base returned value but HEAD differed:", [x["name"] for x in c435["base_returned_value_but_head_differed"]])
    print("Base unavailable/raised -> HEAD value (count, all match scratch?):",
          len(c435["base_unavailable_or_raised_and_head_returned_value"]),
          all(x["matches_scratch"] for x in c435["base_unavailable_or_raised_and_head_returned_value"]))

    print("\n=== RULE 2 TABLE (4,480 CELLS) ===")
    print(f"Total cells: {table['total_cells']}, Amendment 3 mismatches: {len(table['amendment3_mismatches'])}")
    print("Outcome distribution:", json.dumps(table["outcome_distribution"], indent=2, sort_keys=True))
    print(f"Tested cells: {table['explicitly_tested_cell_count']} / {table['total_cells']} ({table['untested_cell_count']} untested)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
