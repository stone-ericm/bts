"""Items 1 & 2 probe: Base (c2039541) vs HEAD (c70effe9) corpus comparison + independent fold.

Builds a deterministic corpus of >400 season histories covering every real writer era,
legacy backfill shape, decision schema, streak-saver boundary, and hostile/malformed shape.
Runs Base (extracted read-only via `git archive c2039541 src`) and HEAD in isolated worker
processes with network access blocked, compares against an independent from-scratch fold,
and classifies every Base-vs-HEAD difference into:
  (a) required by the contract (skip never-committed unresolved pick / ignore foreign-season marker)
  (b) a blocker the contract names
  (c) NOT in the contract
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import stat
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TERMINAL_RESULTS = ("hit", "miss", "void")
ACCEPTED_DECISION_SCHEMAS = (
    "bts_daily_decision_v1",
    "bts_daily_decision_v2",
    "bts_daily_decision_v3",
)
PICK_DATACLASS_KEYS = {
    "batter_name",
    "batter_id",
    "team",
    "lineup_position",
    "pitcher_name",
    "pitcher_id",
    "p_game_hit",
    "flags",
    "projected_lineup",
    "game_pk",
    "game_time",
    "pitcher_team",
}
REQUIRED_PICK_KEYS = PICK_DATACLASS_KEYS - {"pitcher_team"}


def _worker_main(src_dir: str) -> int:
    sys.path.insert(0, src_dir)

    def _deny_net(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    socket.socket = _deny_net  # type: ignore[assignment]
    socket.create_connection = _deny_net  # type: ignore[assignment]

    import bts.picks as picks

    class _Alarm(BaseException):
        pass

    def _on_alarm(*_args):
        raise _Alarm()

    signal.signal(signal.SIGALRM, _on_alarm)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        picks_dir = Path(req["picks_dir"])
        season = req["season"]
        today = req["today"]
        signal.alarm(3)
        try:
            res = picks._replay_season_streak(picks_dir, season, today)
        except _Alarm:
            out = {"status": "timeout"}
        except Exception as exc:  # noqa: BLE001
            out = {"status": "raised", "exc": type(exc).__name__, "msg": str(exc)}
        else:
            if res is None:
                out = {"status": "none"}
            else:
                s, sv = res
                assert type(s) is int and type(sv) is bool, f"Unexpected return types: {res!r}"
                out = {"status": "ok", "value": [s, sv]}
        finally:
            signal.alarm(0)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# Independent from-scratch reference fold (Item 2)
# ---------------------------------------------------------------------------

def _indep_read_regular_utf8(path: Path) -> str | None:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(st.st_mode):
        raise IsADirectoryError(f"Non-regular path: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise IsADirectoryError(f"Non-regular fd: {path}")
        with open(fd, "r", encoding="utf-8", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def _indep_validate_slot(slot: object) -> bool:
    if not isinstance(slot, dict):
        return False
    keys = set(slot.keys())
    if not REQUIRED_PICK_KEYS.issubset(keys):
        return False
    if not keys.issubset(PICK_DATACLASS_KEYS):
        return False
    if type(slot["batter_id"]) is not int:
        return False
    if not isinstance(slot["batter_name"], str) or not isinstance(slot["team"], str):
        return False
    if slot["game_pk"] is not None and type(slot["game_pk"]) is not int:
        return False
    return True


def _indep_read_pick(picks_dir: Path, day: str) -> dict | None:
    raw = _indep_read_regular_utf8(picks_dir / f"{day}.json")
    if raw is None:
        return None
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        raise ValueError("pick not object")
    for req_key in ("date", "run_time", "pick", "double_down", "runner_up"):
        if req_key not in doc:
            raise KeyError(req_key)
    if doc["date"] != day or not isinstance(doc["run_time"], str):
        raise ValueError("bad date or run_time")
    if not _indep_validate_slot(doc["pick"]):
        raise ValueError("bad primary slot")
    dd = doc["double_down"]
    if dd is not None:
        if not isinstance(dd, dict):
            raise ValueError("dd not dict")
        if dd and not _indep_validate_slot(dd):
            raise ValueError("bad dd slot")
        if not dd:
            # Empty dict `{}`: in _daily_pick_from_data, `Pick(**data["double_down"]) if data["double_down"] else None`
            # evaluates `{}` as falsy -> double_down becomes None! Check exact parity!
            dd = None
    res = doc.get("result")
    if res not in (*TERMINAL_RESULTS, None, "suspended", "unresolved"):
        raise ValueError("bad result")
    bp = doc.get("bluesky_posted", False)
    ns = doc.get("notification_sent", False)
    da = doc.get("delivery_attempted", False)
    if any(type(flag) is not bool for flag in (bp, ns, da)):
        raise ValueError("bad bool flag")
    nid = doc.get("notification_id")
    if nid is not None and not isinstance(nid, str):
        raise ValueError("bad notification_id")
    sr = doc.get("slot_results")
    if sr is not None:
        if not isinstance(sr, dict) or any(v not in TERMINAL_RESULTS for v in sr.values()):
            raise ValueError("bad slot_results")
    return {
        "date": day,
        "has_dd": bool(dd),
        "result": res,
        "slot_results": sr,
        "delivered": bool(bp or (ns and nid)),
        "attempted": da,
    }


def _indep_read_decision(picks_dir: Path, day: str) -> dict | None:
    raw = _indep_read_regular_utf8(picks_dir / day / "decision.json")
    if raw is None:
        return None
    doc = json.loads(raw)
    if (
        not isinstance(doc, dict)
        or doc.get("schema_version") not in ACCEPTED_DECISION_SCHEMAS
        or doc.get("action") not in {"skip", "single", "double"}
        or not isinstance(doc.get("scoreable"), bool)
        or doc.get("date") != day
    ):
        raise ValueError("bad decision")
    return doc


def independent_fold(picks_dir: Path, season: int, today_iso: str) -> dict:
    streak_path = picks_dir / "streak.json"
    try:
        s_raw = _indep_read_regular_utf8(streak_path)
    except IsADirectoryError as exc:
        return {"status": "raised", "exc": "NonRegularReplayFile", "msg": str(exc)}
    except OSError:
        s_doc = None
    else:
        if s_raw is None:
            s_doc = {}
        else:
            try:
                parsed = json.loads(s_raw)
                s_doc = parsed if isinstance(parsed, dict) else None
            except (ValueError, RecursionError):
                s_doc = None

    markers: list[str] = []
    if isinstance(s_doc, dict):
        if "season" in s_doc and not (type(s_doc["season"]) is int and s_doc["season"] == season):
            markers = []
        elif isinstance(s_doc.get("applied_dates"), list):
            markers = [m for m in s_doc["applied_dates"] if isinstance(m, str)]

    dates = {p.stem for p in picks_dir.glob(f"{season}-*.json")}
    dates.update(p.parent.name for p in picks_dir.glob(f"{season}-*/decision.json"))

    streak = 0
    saver = True
    for day in sorted(dates):
        if day > today_iso or not ISO_DATE_RE.fullmatch(day):
            continue
        try:
            pick = _indep_read_pick(picks_dir, day)
        except Exception:  # noqa: BLE001
            return {"status": "none", "blocker": [day, "unreadable or invalid pick file"]}
        try:
            dec = _indep_read_decision(picks_dir, day)
        except Exception:  # noqa: BLE001
            return {"status": "none", "blocker": [day, "unreadable or invalid decision record"]}

        scoreable = dec["scoreable"] if dec is not None else bool(pick and pick["delivered"])

        if pick is None:
            if scoreable:
                return {"status": "none", "blocker": [day, "committed pick file missing"]}
            continue

        res = pick["result"]
        if res in TERMINAL_RESULTS:
            if res == "void":
                continue
            if res == "hit":
                if pick["slot_results"] is not None:
                    inc = sum(1 for v in pick["slot_results"].values() if v == "hit")
                else:
                    inc = 2 if pick["has_dd"] else 1
                streak += inc
            else:
                if saver and 10 <= streak <= 15:
                    saver = False
                else:
                    streak = 0
        elif day in markers:
            return {"status": "none", "blocker": [day, "applied-date marker without a result"]}
        elif day < today_iso:
            if scoreable:
                return {"status": "none", "blocker": [day, "committed pick awaiting a result"]}
            if dec is None and pick["attempted"]:
                return {"status": "none", "blocker": [day, "delivery attempt without a decision or result"]}
            # Uncommitted past pick -> skipped (contributes 0)
        else:
            # Unmarked unresolved today -> excluded from fold
            pass

    return {"status": "ok", "value": [streak, saver]}


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

def _slot_dict(
    day: str,
    *,
    batter_id=660271,
    batter_name="Shohei Ohtani",
    team="LAD",
    game_pk=745100,
    include_pitcher_team=True,
    pitcher_id=543037,
) -> dict:
    d = {
        "batter_name": batter_name,
        "batter_id": batter_id,
        "team": team,
        "lineup_position": 1,
        "pitcher_name": "Gerrit Cole",
        "pitcher_id": pitcher_id,
        "p_game_hit": 0.81,
        "flags": [],
        "projected_lineup": False,
        "game_pk": game_pk,
        "game_time": f"{day}T23:10:00Z",
    }
    if include_pitcher_team:
        d["pitcher_team"] = "NYY"
    return d


def _pick_dict(
    day: str,
    result: str | None = "hit",
    *,
    dd: bool = False,
    slot_results: dict | None = "auto",  # type: ignore[assignment]
    legacy_no_delivery: bool = False,
    bluesky_posted: object = False,
    notification_sent: object = False,
    notification_id: object = None,
    delivery_attempted: object = False,
    include_pitcher_team: bool = True,
    game_pk: object = 745100,
) -> dict:
    p1 = _slot_dict(day, game_pk=game_pk, include_pitcher_team=include_pitcher_team)
    p2 = (
        _slot_dict(
            day,
            batter_id=605141,
            batter_name="Mookie Betts",
            team="LAD",
            game_pk=(None if game_pk is None else 745101),
            include_pitcher_team=include_pitcher_team,
        )
        if dd
        else None
    )
    if slot_results == "auto":
        if result in TERMINAL_RESULTS:
            sr = {"pick": result, **({"double_down": result} if dd else {})}
        else:
            sr = None
    else:
        sr = slot_results

    doc = {
        "date": day,
        "run_time": f"{day}T18:00:00+00:00",
        "pick": p1,
        "double_down": p2,
        "runner_up": {"batter_name": "Freddie Freeman", "p_game_hit": 0.77},
        "result": result,
    }
    if sr is not None or not legacy_no_delivery:
        doc["slot_results"] = sr
    if not legacy_no_delivery:
        doc.update(
            {
                "bluesky_posted": bluesky_posted,
                "bluesky_uri": "at://did:plc:test/post/1" if bluesky_posted is True else None,
                "notification_sent": notification_sent,
                "notification_channel": "bluesky_dm" if notification_sent is True else None,
                "notification_id": notification_id,
                "delivery_attempted": delivery_attempted,
                "delivered_at": f"{day}T18:01:00-04:00" if (bluesky_posted is True or notification_sent is True) else None,
            }
        )
    return doc


def _decision_dict(
    day: str,
    *,
    schema: str = "bts_daily_decision_v3",
    action: str = "single",
    scoreable: bool = True,
    delivery_status: str = "delivered",
    objective: str = "reach57",
) -> dict:
    rec = {
        "schema_version": schema,
        "date": day,
        "action": action,
        "source": "mdp",
        "primary": {"batter_id": 660271, "batter_name": "Shohei Ohtani", "team": "LAD", "game_pk": 745100, "p_game_hit": 0.81},
        "double_down": None,
        "delivery_status": delivery_status,
        "scoreable": scoreable,
        "finalized_at": f"{day}T23:59:00Z",
    }
    if schema in ("bts_daily_decision_v2", "bts_daily_decision_v3"):
        rec.update({"second_candidate": None, "streak": 0, "saver_available": True, "state_source": "contest", "state_status": "fresh", "allow_double": True, "contest_source_date": day})
    if schema == "bts_daily_decision_v3":
        rec.update({"objective": objective, "best_streak": 0, "best_status": "fresh", "effective_best": 0, "tail_policy_sha256": None, "degraded_reason": None})
    return rec


@dataclass
class CaseSpec:
    name: str
    season: int
    today: str
    expected_category: str  # "same", "a", "b", or "c"
    real_writer_possible: bool
    note: str
    builder: object  # Callable[[Path], None]


def _build_deterministic_cases() -> list[CaseSpec]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from bts.state.regenerate import HistoricalPickRecord, Timeline, _hist_to_pick_file, _record_to_pick_file, compose_state_from_snapshot_and_timeline
    from bts.state.export import export_initial_state

    cases: list[CaseSpec] = []

    def add(name: str, season: int, today: str, cat: str, real: bool, note: str, fn):
        cases.append(CaseSpec(name=name, season=season, today=today, expected_category=cat, real_writer_possible=real, note=note, builder=fn))

    # --- 1. Real writer eras (must all be "same" when terminal) ---
    def c_6798410(root: Path):
        for i, res in enumerate(("hit", "miss", "hit")):
            d = f"2026-04-0{i + 1}"
            doc = {
                "date": d,
                "run_time": f"{d}T18:00:00Z",
                "pick": _slot_dict(d, include_pitcher_team=False),
                "double_down": None,
                "runner_up": None,
                "streak": i,
                "bluesky_posted": True,
                "bluesky_uri": f"at://post/{i}",
                "result": res,
            }
            (root / f"{d}.json").write_text(json.dumps(doc))

    add("era_6798410_extra_streak_key", 2026, "2026-04-10", "same", True, "Commit 6798410 top-level streak field + no pitcher_team", c_6798410)

    def c_mar29_30_null_game_pk(root: Path):
        for d, res, dd in (("2026-03-29", "hit", False), ("2026-03-30", "miss", False), ("2026-03-31", "hit", True)):
            doc = _pick_dict(d, res, dd=dd, slot_results=None, legacy_no_delivery=True, include_pitcher_team=False, game_pk=None)
            doc["bluesky_posted"] = True
            (root / f"{d}.json").write_text(json.dumps(doc))

    add("era_mar29_30_null_game_pk", 2026, "2026-04-05", "same", True, "Retroactive 3/29-3/30 null game_pk (I-008/I-094)", c_mar29_30_null_game_pk)

    def c_regen_records(root: Path):
        recs = [
            HistoricalPickRecord("2026-04-01", "Nico Hoerner", "CHC", False, None, None, "at://1", "hit", 1),
            HistoricalPickRecord("2026-04-02", "Jose Altuve", "HOU", True, "Kyle Tucker", None, "at://2", "hit", 3),
            HistoricalPickRecord("2026-04-03", "Aaron Judge", "NYY", False, None, None, "at://3", "miss", 0),
        ]
        for r in recs:
            (root / f"{r.date}.json").write_text(json.dumps(_record_to_pick_file(r)))

    add("era_regenerate_record_to_pick_file", 2026, "2026-04-10", "same", True, "state/regenerate.py _record_to_pick_file single + teamless DD", c_regen_records)

    def c_regen_hist_valid(root: Path):
        for d, res in (("2026-04-01", "hit"), ("2026-04-02", "miss"), ("2026-04-03", "void")):
            hist = {
                "date": d,
                "pick": {
                    "batter_name": "Legacy Batter",
                    "batter_id": None,
                    "team": "NYY",
                    "pitcher_name": None,
                    "pitcher_id": None,
                    "game_pk": None,
                    "game_time": None,
                    "p_game_hit": None,
                },
                "double_down": None,
                "result": res,
            }
            (root / f"{d}.json").write_text(json.dumps(_hist_to_pick_file(hist)))

    add("era_regenerate_hist_to_pick_file_backfilled", 2026, "2026-04-10", "same", True, "state/regenerate.py _hist_to_pick_file with None backfills", c_regen_hist_valid)

    for res in ("hit", "miss", "void"):
        def _mk_missing_pid(r=res):
            def _fn(root: Path):
                hist = {
                    "date": "2026-04-01",
                    "pick": {"batter_name": "B", "batter_id": 1, "team": "NYY", "pitcher_name": "P", "p_game_hit": 0.8, "game_pk": 1, "game_time": "2026-04-01T23:00:00Z"},
                    "double_down": None,
                    "result": r,
                }
                (root / "2026-04-01.json").write_text(json.dumps(_hist_to_pick_file(hist)))
            return _fn

        add(
            f"hist_missing_pitcher_id_{res}",
            2026,
            "2026-04-10",
            "b",
            False,
            f"Hand-crafted snapshot missing required Pick.pitcher_id ({res}): export_initial_state always writes pitcher_id",
            _mk_missing_pid(),
        )

    def c_export_compose_roundtrip(root: Path):
        src_dir = root / "_src_export"
        src_dir.mkdir()
        for i in range(12):
            d = (date(2026, 5, 1) + timedelta(days=i)).isoformat()
            res = "miss" if i == 10 else "hit"
            (src_dir / f"{d}.json").write_text(json.dumps(_pick_dict(d, res, bluesky_posted=True)))
        (src_dir / "streak.json").write_text(json.dumps({"streak": 11, "saver_available": False}))
        snap = root / "_snap.json"
        export_initial_state(src_dir, snap)
        compose_state_from_snapshot_and_timeline(snap, Timeline(), root)
        snap.unlink()

    add("era_export_and_regenerate_roundtrip", 2026, "2026-05-20", "same", True, "export_initial_state -> compose_state_from_snapshot_and_timeline", c_export_compose_roundtrip)

    def c_partial_voids_and_dd(root: Path):
        (root / "2026-06-01.json").write_text(json.dumps(_pick_dict("2026-06-01", "hit", dd=True, slot_results=None, legacy_no_delivery=True)))
        (root / "2026-06-02.json").write_text(json.dumps(_pick_dict("2026-06-02", "hit", dd=True, slot_results={"pick": "void", "double_down": "hit"})))
        (root / "2026-06-03.json").write_text(json.dumps(_pick_dict("2026-06-03", "hit", dd=True, slot_results={"pick": "hit", "double_down": "void"})))
        (root / "2026-06-04.json").write_text(json.dumps(_pick_dict("2026-06-04", "void", dd=True, slot_results={"pick": "void", "double_down": "void"})))
        (root / "2026-06-05.json").write_text(json.dumps(_pick_dict("2026-06-05", "void", dd=False, slot_results={"pick": "void"})))

    add("era_partial_voids_and_legacy_dd", 2026, "2026-06-10", "same", True, "Legacy DD (+2), partial void DDs (+1 each), full voids (+0)", c_partial_voids_and_dd)

    def c_decision_schemas_v1_v2_v3(root: Path):
        for idx, (sch, obj) in enumerate([
            ("bts_daily_decision_v1", "reach57"),
            ("bts_daily_decision_v2", "reach57"),
            ("bts_daily_decision_v3", "reach57"),
            ("bts_daily_decision_v3", "emax_season_best"),
        ], start=1):
            d = f"2026-07-0{idx}"
            (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, "hit", notification_sent=True, notification_id=f"dm-{idx}")))
            (root / d).mkdir()
            (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, schema=sch, objective=obj)))

    add("era_decision_schemas_v1_v2_v3", 2026, "2026-07-10", "same", True, "All 3 accepted decision schemas + both objectives", c_decision_schemas_v1_v2_v3)

    def c_non_pick_files_ignored(root: Path):
        (root / "2026-05-01.json").write_text(json.dumps(_pick_dict("2026-05-01", "hit")))
        (root / "._2026-05-02.json").write_bytes(b"\x00\x05\x16\x07bad")
        (root / "2026-05-02.shadow.json").write_text("{corrupt shadow")
        (root / "lineup_evolution_2026-05-02.jsonl").write_text("{}\n")
        (root / "2026-05-02").mkdir()
        (root / "2026-05-02" / "deferred_fallback_20260502T180000-0400.json").write_text("{bad archive")
        (root / "2026-05-20.json").write_text("{future corrupt preview")
        (root / "2025-09-20.json").write_text("{prior season corrupt")

    add("non_canonical_and_future_ignored", 2026, "2026-05-10", "same", True, "AppleDouble, shadow, archive, future, prior season ignored", c_non_pick_files_ignored)

    # --- 2. Category (a): uncommitted nonterminal past picks skipped; foreign-season marker ignored ---
    for sub_name, sub_res, with_skip_dec in [
        ("refused_preview_none", None, False),
        ("uncommitted_suspended", "suspended", False),
        ("uncommitted_unresolved", "unresolved", False),
        ("skip_leftover_none", None, True),
        ("skip_leftover_unresolved", "unresolved", True),
    ]:
        def _mk_skip(sr=sub_res, wsd=with_skip_dec):
            def _fn(root: Path):
                (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", notification_sent=True, notification_id="m1")))
                (root / "2027-04-02.json").write_text(json.dumps(_pick_dict("2027-04-02", sr)))
                if wsd:
                    (root / "2027-04-02").mkdir()
                    (root / "2027-04-02" / "decision.json").write_text(json.dumps(_decision_dict("2027-04-02", action="skip", scoreable=False, delivery_status="not_applicable")))
                (root / "2027-04-03.json").write_text(json.dumps(_pick_dict("2027-04-03", "hit", notification_sent=True, notification_id="m3")))
            return _fn

        add(f"cat_a_{sub_name}", 2027, "2027-04-05", "a", True, f"Uncommitted past pick ({sub_name}) skipped", _mk_skip())

    def c_skip_decision_only(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", notification_sent=True, notification_id="m1")))
        (root / "2027-04-02").mkdir()
        (root / "2027-04-02" / "decision.json").write_text(json.dumps(_decision_dict("2027-04-02", action="skip", scoreable=False, delivery_status="not_applicable")))

    add("skip_decision_without_pick_file", 2027, "2027-04-05", "same", True, "Skip decision with no root pick contributes 0 on both Base and HEAD", c_skip_decision_only)

    def c_foreign_marker_today(root: Path):
        (root / "2027-01-01.json").write_text(json.dumps(_pick_dict("2027-01-01", None)))
        (root / "streak.json").write_text(json.dumps({"streak": 9, "saver_available": True, "season": 2026, "applied_dates": ["2027-01-01"]}))

    add("cat_a_foreign_season_marker_on_today", 2027, "2027-01-01", "a", True, "Foreign-season marker ignored on today's unresolved preview", c_foreign_marker_today)

    # --- 3. Category (b) & field-type rejection probes across hit / miss / void ---
    def _add_mutated_pick_probes(label: str, mutate_fn, cat_if_diff: str, real_possible: bool, note: str, results=("hit", "miss", "void")):
        for res in results:
            def _fn(root: Path, r=res, mf=mutate_fn):
                (root / "2026-05-01.json").write_text(json.dumps(_pick_dict("2026-05-01", "hit", bluesky_posted=True)))
                doc = _pick_dict("2026-05-02", r, bluesky_posted=True)
                raw = mf(doc, "2026-05-02")
                if isinstance(raw, bytes):
                    (root / "2026-05-02.json").write_bytes(raw)
                elif isinstance(raw, str):
                    (root / "2026-05-02.json").write_text(raw)
                else:
                    (root / "2026-05-02.json").write_text(json.dumps(raw))

            add(f"shape_{label}_{res}", 2026, "2026-05-10", cat_if_diff, real_possible, f"{note} (result={res})", _fn)

    _add_mutated_pick_probes("batter_id_numeric_str", lambda d, _: (d["pick"].update(batter_id="660271") or d), "b", False, "batter_id as numeric string")
    _add_mutated_pick_probes("game_pk_numeric_str", lambda d, _: (d["pick"].update(game_pk="745100") or d), "b", False, "game_pk as numeric string")
    _add_mutated_pick_probes("batter_id_float", lambda d, _: (d["pick"].update(batter_id=660271.0) or d), "b", False, "batter_id as float")
    _add_mutated_pick_probes("batter_id_bool", lambda d, _: (d["pick"].update(batter_id=True) or d), "b", False, "batter_id as bool")
    _add_mutated_pick_probes("game_pk_bool", lambda d, _: (d["pick"].update(game_pk=False) or d), "b", False, "game_pk as bool")
    _add_mutated_pick_probes("batter_name_null", lambda d, _: (d["pick"].update(batter_name=None) or d), "b", False, "batter_name null")
    _add_mutated_pick_probes("team_null", lambda d, _: (d["pick"].update(team=None) or d), "b", False, "team null")
    _add_mutated_pick_probes("bluesky_posted_null", lambda d, _: (d.update(bluesky_posted=None) or d), "b", False, "bluesky_posted explicit null")
    _add_mutated_pick_probes("notification_sent_null", lambda d, _: (d.update(notification_sent=None) or d), "b", False, "notification_sent explicit null")
    _add_mutated_pick_probes("delivery_attempted_null", lambda d, _: (d.update(delivery_attempted=None) or d), "b", False, "delivery_attempted explicit null")
    _add_mutated_pick_probes("bluesky_posted_int", lambda d, _: (d.update(bluesky_posted=1) or d), "b", False, "bluesky_posted as int 1")
    _add_mutated_pick_probes("notification_id_int", lambda d, _: (d.update(notification_id=999) or d), "b", False, "notification_id as int")
    _add_mutated_pick_probes("run_time_null", lambda d, _: (d.update(run_time=None) or d), "b", False, "run_time null")
    _add_mutated_pick_probes("runner_up_missing", lambda d, _: ({k: v for k, v in d.items() if k != "runner_up"}), "b", False, "runner_up key omitted")
    _add_mutated_pick_probes("date_mismatch", lambda d, _: (d.update(date="2026-05-01") or d), "b", False, "date field differs from filename")
    _add_mutated_pick_probes("slot_results_nonterminal", lambda d, _: (d.update(slot_results={"pick": "hit", "double_down": None}) or d), "b", False, "slot_results contains None")
    _add_mutated_pick_probes("slot_results_list", lambda d, _: (d.update(slot_results=["hit"]) or d), "b", False, "slot_results is a list")
    _add_mutated_pick_probes("extra_top_level_key", lambda d, _: (d.update(extra_unknown_field={"x": 1}) or d), "same", True, "extra unknown top-level key")
    _add_mutated_pick_probes("extra_slot_key_p_hit_pa", lambda d, _: (d["pick"].update(p_hit_pa=0.31) or d), "b", False, "extra unknown key inside pick (test_state_export fixture shape)")
    _add_mutated_pick_probes("utf8_bom", lambda d, _: b"\xef\xbb\xbf" + json.dumps(d).encode("utf-8"), "same", False, "UTF-8 BOM prefix (both Base and HEAD reject via json.loads)")

    # Empty dict `{}` as double_down: check parity!
    _add_mutated_pick_probes("empty_dict_double_down", lambda d, _: (d.update(double_down={}) or d), "same", False, "double_down as empty dict {}")

    # Empty dict `{}` as slot_results on hit:
    _add_mutated_pick_probes("empty_dict_slot_results", lambda d, _: (d.update(slot_results={}) or d), "same", False, "slot_results as empty dict {}")

    # Raw malformed files
    for raw_label, raw_bytes, exp_cat in [
        ("json_array", b"[]", "b"),
        ("json_string", b'"just a string"', "b"),
        ("json_int", b"123", "b"),
        ("json_null", b"null", "b"),
        ("truncated_json", b'{"date": "2026-05-02"', "same"),
        ("invalid_utf8", b"\xff\xfe\x00\x00", "same"),
        ("only_result_miss", b'{"result": "miss"}', "b"),
        ("only_result_void", b'{"result": "void"}', "b"),
        ("only_result_hit", b'{"result": "hit"}', "b"),
        ("unknown_result_value", None, "b"),
        ("giant_int_in_pick", b'{"result": "miss", "x": ' + (b"9" * 5000) + b"}", "same"),
    ]:
        def _mk_raw(rb=raw_bytes, rl=raw_label):
            def _fn(root: Path):
                (root / "2026-05-01.json").write_text(json.dumps(_pick_dict("2026-05-01", "hit", bluesky_posted=True)))
                if rl == "unknown_result_value":
                    doc = _pick_dict("2026-05-02", "hit", bluesky_posted=True)
                    doc["result"] = "postponed_forever"
                    (root / "2026-05-02.json").write_text(json.dumps(doc))
                else:
                    assert rb is not None
                    (root / "2026-05-02.json").write_bytes(rb)
            return _fn

        add(f"raw_{raw_label}", 2026, "2026-05-10", exp_cat, False, f"Raw file shape: {raw_label}", _mk_raw())

    # Blockers on decision / missing pick / markers / today / special files
    def c_missing_root_scoreable(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", notification_sent=True, notification_id="m1")))
        (root / "2027-04-02").mkdir()
        (root / "2027-04-02" / "decision.json").write_text(json.dumps(_decision_dict("2027-04-02", scoreable=True)))

    add("blocker_scoreable_decision_missing_pick", 2027, "2027-04-05", "b", True, "Scoreable decision with missing root pick blocks on HEAD (Base skipped!)", c_missing_root_scoreable)

    def c_corrupt_decision_beside_hit(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", notification_sent=True, notification_id="m1")))
        (root / "2027-04-01").mkdir()
        (root / "2027-04-01" / "decision.json").write_text("{}")

    add("blocker_invalid_decision_beside_terminal_hit", 2027, "2027-04-05", "b", False, "Invalid decision {} beside terminal hit blocks on HEAD", c_corrupt_decision_beside_hit)

    def c_date_mismatch_decision_beside_hit(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", notification_sent=True, notification_id="m1")))
        (root / "2027-04-01").mkdir()
        (root / "2027-04-01" / "decision.json").write_text(json.dumps(_decision_dict("2027-03-31", scoreable=True)))

    add("blocker_date_mismatched_decision", 2027, "2027-04-05", "b", False, "Date-mismatched decision blocks on HEAD", c_date_mismatch_decision_beside_hit)

    def c_unhashable_action_decision(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", notification_sent=True, notification_id="m1")))
        (root / "2027-04-01").mkdir()
        dec = _decision_dict("2027-04-01", scoreable=True)
        dec["action"] = ["single"]  # type: ignore[assignment]
        (root / "2027-04-01" / "decision.json").write_text(json.dumps(dec))

    add("blocker_unhashable_action_decision", 2027, "2027-04-05", "b", False, "Decision with list action caught by TypeError", c_unhashable_action_decision)

    for m_label, s_tag, exp_cat in [
        ("same_season_marker_past", 2027, "same"),
        ("tagless_marker_past", None, "same"),
        ("foreign_season_marker_past_uncommitted", 2026, "a"),
    ]:
        def _mk_marker(tag=s_tag):
            def _fn(root: Path):
                (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", None)))
                s_doc = {"streak": 4, "saver_available": True, "applied_dates": ["2027-04-01"]}
                if tag is not None:
                    s_doc["season"] = tag
                (root / "streak.json").write_text(json.dumps(s_doc))
            return _fn

        add(f"marker_{m_label}", 2027, "2027-04-05", exp_cat, True, f"Past unresolved uncommitted pick with {m_label}", _mk_marker())

    def c_today_malformed_minimal(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", bluesky_posted=True)))
        (root / "2027-04-05.json").write_text(json.dumps({"date": "2027-04-05", "result": None}))

    add("blocker_today_malformed_minimal", 2027, "2027-04-05", "b", False, "Malformed minimal pick on today blocks on HEAD (Base skipped today before load_pick)", c_today_malformed_minimal)

    def c_today_valid_preview(root: Path):
        (root / "2027-04-01.json").write_text(json.dumps(_pick_dict("2027-04-01", "hit", bluesky_posted=True)))
        (root / "2027-04-05.json").write_text(json.dumps(_pick_dict("2027-04-05", None)))

    add("today_valid_unmarked_preview", 2027, "2027-04-05", "same", True, "Valid unmarked preview on today excluded on both Base and HEAD", c_today_valid_preview)

    def c_today_terminal_hit_and_miss_saver(root: Path):
        for i in range(10):
            d = (date(2027, 4, 1) + timedelta(days=i)).isoformat()
            (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, "hit", bluesky_posted=True)))
        (root / "2027-04-11.json").write_text(json.dumps(_pick_dict("2027-04-11", "miss", bluesky_posted=True)))

    add("today_terminal_miss_consumes_saver", 2027, "2027-04-11", "same", True, "Today's terminal miss at streak 10 included and consumes saver", c_today_terminal_miss_saver := c_today_terminal_hit_and_miss_saver)

    return cases


def _build_seeded_cases() -> list[CaseSpec]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from bts.state.regenerate import HistoricalPickRecord, _hist_to_pick_file, _record_to_pick_file

    rng = random.Random(20261119)
    cases: list[CaseSpec] = []

    # 210 valid real-writer season histories (no uncommitted past picks, no missing root picks, no foreign today markers)
    for idx in range(210):
        season = 2026 if idx < 120 else 2027
        n_days = rng.randint(8, 45)
        start = date(season, 4, 1) + timedelta(days=rng.randint(0, 60))
        today_dt = start + timedelta(days=n_days)
        today_iso = today_dt.isoformat()
        seed_val = rng.randrange(1 << 30)

        def _make_valid(s=season, st=start, nd=n_days, td_iso=today_iso, sv=seed_val):
            def _fn(root: Path):
                r = random.Random(sv)
                for step in range(nd + 1):
                    cur = st + timedelta(days=step)
                    d = cur.isoformat()
                    if step == nd:
                        # Today: either no pick, valid unmarked preview, or terminal result
                        t_mode = r.choice(("none", "preview", "hit", "miss", "void"))
                        if t_mode == "none":
                            continue
                        res = None if t_mode == "preview" else t_mode
                        (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, res, notification_sent=res is not None, notification_id="dm-t" if res is not None else None)))
                        continue

                    mode = r.choice(("absent", "skip_dec_only", "legacy_single", "legacy_dd_no_sr", "null_gpk", "regen_rec", "regen_hist", "modern_dm", "partial_void_dd", "full_void", "private_terminal"))
                    if mode == "absent":
                        continue
                    if mode == "skip_dec_only":
                        (root / d).mkdir()
                        (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, schema=r.choice(ACCEPTED_DECISION_SCHEMAS), action="skip", scoreable=False, delivery_status="not_applicable")))
                        continue

                    res = r.choices(("hit", "miss", "void"), weights=(72, 20, 8))[0]
                    if mode == "legacy_single":
                        doc = _pick_dict(d, res, slot_results=None, legacy_no_delivery=True, include_pitcher_team=r.choice((True, False)))
                        if r.random() < 0.2:
                            doc["streak"] = step
                        (root / f"{d}.json").write_text(json.dumps(doc))
                    elif mode == "legacy_dd_no_sr":
                        doc = _pick_dict(d, res, dd=True, slot_results=None, legacy_no_delivery=True, include_pitcher_team=False)
                        (root / f"{d}.json").write_text(json.dumps(doc))
                    elif mode == "null_gpk":
                        doc = _pick_dict(d, res, dd=r.choice((False, True)), slot_results=None, legacy_no_delivery=True, game_pk=None)
                        (root / f"{d}.json").write_text(json.dumps(doc))
                    elif mode == "regen_rec":
                        if res == "void":
                            res = "hit"
                        rec = HistoricalPickRecord(d, "Batter R", "CHC", r.choice((False, True)), "Batter D", r.choice((None, "BOS")), "at://r", res, 1)
                        (root / f"{d}.json").write_text(json.dumps(_record_to_pick_file(rec)))
                    elif mode == "regen_hist":
                        hist = {
                            "date": d,
                            "pick": {"batter_name": "Snap B", "batter_id": None, "team": "LAD", "pitcher_name": None, "pitcher_id": None, "game_pk": None, "game_time": None, "p_game_hit": None},
                            "double_down": None,
                            "result": res,
                        }
                        (root / f"{d}.json").write_text(json.dumps(_hist_to_pick_file(hist)))
                    elif mode == "modern_dm":
                        dd = r.random() < 0.25
                        (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, res, dd=dd, notification_sent=True, notification_id=f"dm-{step}", delivery_attempted=True)))
                        if r.random() < 0.6:
                            (root / d).mkdir()
                            (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, schema=r.choice(ACCEPTED_DECISION_SCHEMAS), action=("double" if dd else "single"), scoreable=True)))
                    elif mode == "partial_void_dd":
                        other = r.choice(("hit", "miss"))
                        eff = other
                        sr = {"pick": "void", "double_down": other} if r.random() < 0.5 else {"pick": other, "double_down": "void"}
                        (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, eff, dd=True, slot_results=sr, notification_sent=True, notification_id=f"dm-{step}")))
                    elif mode == "full_void":
                        (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, "void", dd=True, slot_results={"pick": "void", "double_down": "void"}, bluesky_posted=True)))
                    elif mode == "private_terminal":
                        (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, res)))
                        (root / d).mkdir()
                        (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, action="single", scoreable=True, delivery_status="private_locked")))

                # Optional future preview and non-pick artifacts
                fut = (st + timedelta(days=nd + 1)).isoformat()
                if r.random() < 0.4:
                    (root / f"{fut}.json").write_text(json.dumps(_pick_dict(fut, None)))
                if r.random() < 0.3:
                    (root / f"{st.isoformat()}.shadow.json").write_text(json.dumps(_pick_dict(st.isoformat(), None)))
            return _fn

        cases.append(CaseSpec(f"seeded_valid_{idx:03d}", season, today_iso, "same", True, "Seeded valid legacy+modern season history", _make_valid()))

    # 120 mixed histories injecting uncommitted skips, committed blockers, markers, or malformed records
    for idx in range(120):
        season = 2027
        n_days = rng.randint(12, 35)
        start = date(season, 4, 1) + timedelta(days=rng.randint(0, 40))
        today_iso = (start + timedelta(days=n_days)).isoformat()
        flavor = ("uncommitted_only", "committed_blocker", "missing_pick_blocker", "bad_decision_blocker", "malformed_pick_blocker", "marker_blocker")[idx % 6]
        seed_val = rng.randrange(1 << 30)

        exp_cat = "a" if flavor == "uncommitted_only" else "b"
        real_pos = flavor not in ("bad_decision_blocker", "malformed_pick_blocker")

        def _make_mixed(st=start, nd=n_days, fl=flavor, sv=seed_val):
            def _fn(root: Path):
                r = random.Random(sv)
                special_step = r.randint(1, nd - 2)
                for step in range(nd):
                    d = (st + timedelta(days=step)).isoformat()
                    if step == special_step:
                        if fl == "uncommitted_only":
                            kind = r.choice(("refused_preview", "skip_leftover", "uncommitted_suspended"))
                            res = "suspended" if kind == "uncommitted_suspended" else None
                            (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, res)))
                            if kind == "skip_leftover":
                                (root / d).mkdir()
                                (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, action="skip", scoreable=False, delivery_status="not_applicable")))
                        elif fl == "committed_blocker":
                            kind = r.choice(("dm_pending", "public_pending", "private_pending", "locked_unconfirmed", "bare_attempt"))
                            (root / f"{d}.json").write_text(json.dumps(_pick_dict(
                                d,
                                r.choice((None, "suspended", "unresolved")),
                                bluesky_posted=(kind == "public_pending"),
                                notification_sent=(kind == "dm_pending"),
                                notification_id=("dm-x" if kind == "dm_pending" else None),
                                delivery_attempted=(kind in ("dm_pending", "locked_unconfirmed", "bare_attempt")),
                            )))
                            if kind in ("private_pending", "locked_unconfirmed"):
                                (root / d).mkdir()
                                (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, scoreable=True, delivery_status=kind)))
                        elif fl == "missing_pick_blocker":
                            (root / d).mkdir()
                            (root / d / "decision.json").write_text(json.dumps(_decision_dict(d, scoreable=True)))
                        elif fl == "bad_decision_blocker":
                            (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, "hit", notification_sent=True, notification_id="dm-ok")))
                            (root / d).mkdir()
                            (root / d / "decision.json").write_text(r.choice(("{bad", "{}", json.dumps(_decision_dict("2027-03-01", scoreable=True)))))
                        elif fl == "malformed_pick_blocker":
                            bad_choice = r.choice(("miss_minimal", "array", "bad_batter_id_str", "date_mismatch"))
                            if bad_choice == "miss_minimal":
                                (root / f"{d}.json").write_text('{"result": "miss"}')
                            elif bad_choice == "array":
                                (root / f"{d}.json").write_text("[]")
                            elif bad_choice == "bad_batter_id_str":
                                doc = _pick_dict(d, "miss")
                                doc["pick"]["batter_id"] = "123"
                                (root / f"{d}.json").write_text(json.dumps(doc))
                            else:
                                doc = _pick_dict(d, "void")
                                doc["date"] = "2027-03-01"
                                (root / f"{d}.json").write_text(json.dumps(doc))
                        elif fl == "marker_blocker":
                            (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, None)))
                            s_doc = {"streak": 5, "saver_available": True, "applied_dates": [d]}
                            if r.random() < 0.5:
                                s_doc["season"] = 2027
                            (root / "streak.json").write_text(json.dumps(s_doc))
                        continue

                    # Ordinary resolved day (including saver-window hits/misses)
                    res = "miss" if step == 11 else r.choices(("hit", "miss", "void"), weights=(80, 12, 8))[0]
                    (root / f"{d}.json").write_text(json.dumps(_pick_dict(d, res, notification_sent=True, notification_id=f"dm-{step}")))
            return _fn

        cases.append(CaseSpec(f"seeded_mixed_{idx:03d}_{flavor}", season, today_iso, exp_cat, real_pos, f"Seeded mixed history ({flavor})", _make_mixed()))

    return cases


def _classify_actual_diff(spec: CaseSpec, base_out: dict, head_out: dict) -> str:
    if base_out == head_out:
        return "same"
    # Base blocked ("none"), HEAD folded ("ok") -> uncommitted nonterminal skipped or foreign marker ignored
    if base_out["status"] == "none" and head_out["status"] == "ok" and spec.expected_category == "a":
        return "a"
    # HEAD blocked ("none") where Base folded ("ok") or crashed ("raised") -> named contract blocker
    if head_out["status"] == "none" and base_out["status"] in ("ok", "raised") and spec.expected_category == "b":
        return "b"
    return "c"


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return _worker_main(sys.argv[2])

    with TemporaryDirectory(dir=REVIEW_DIR, prefix="_corpus_run_") as scratch_str:
        scratch = Path(scratch_str)
        base_root = scratch / "base_c2039541"
        base_root.mkdir()
        with subprocess.Popen(
            ["git", "archive", "c2039541", "src"],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
        ) as archive:
            assert archive.stdout is not None
            with tarfile.open(fileobj=archive.stdout, mode="r|") as tf:
                tf.extractall(base_root, filter="data")
            assert archive.wait() == 0, "git archive c2039541 failed"

        all_cases = _build_deterministic_cases() + _build_seeded_cases()
        assert len(all_cases) >= 300, f"Expected >=300 histories, built {len(all_cases)}"

        base_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(base_root / "src")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        head_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(REPO_ROOT / "src")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )

        def _ask(proc: subprocess.Popen, picks_dir: Path, season: int, today: str) -> dict:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps({"picks_dir": str(picks_dir), "season": season, "today": today}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            assert line, "Worker exited unexpectedly"
            return json.loads(line)

        summary_counts = {"total": len(all_cases), "same": 0, "a": 0, "b": 0, "c": 0, "indep_mismatches": 0}
        divergences: list[dict] = []
        indep_mismatches: list[dict] = []

        try:
            for idx, spec in enumerate(all_cases):
                case_dir = scratch / f"case_{idx:04d}"
                case_dir.mkdir()
                spec.builder(case_dir)

                base_res = _ask(base_proc, case_dir, spec.season, spec.today)
                head_res = _ask(head_proc, case_dir, spec.season, spec.today)
                indep_res = independent_fold(case_dir, spec.season, spec.today)

                # Compare HEAD vs independent fold
                head_norm = {"status": head_res["status"], **({"value": head_res["value"]} if head_res["status"] == "ok" else {})}
                indep_norm = {"status": indep_res["status"], **({"value": indep_res["value"]} if indep_res["status"] == "ok" else {})}
                if head_norm != indep_norm:
                    summary_counts["indep_mismatches"] += 1
                    indep_mismatches.append({
                        "name": spec.name,
                        "head": head_res,
                        "indep": indep_res,
                    })

                cat = _classify_actual_diff(spec, base_res, head_res)
                summary_counts[cat] += 1
                if cat != "same":
                    divergences.append({
                        "name": spec.name,
                        "category": cat,
                        "real_writer_possible": spec.real_writer_possible,
                        "base": base_res,
                        "head": head_res,
                        "indep_blocker": indep_res.get("blocker"),
                        "note": spec.note,
                    })
        finally:
            for proc in (base_proc, head_proc):
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=5)

        out_report = {
            "summary": summary_counts,
            "indep_mismatches": indep_mismatches,
            "cat_c": [d for d in divergences if d["category"] == "c"],
            "divergences_sample_by_kind": divergences[:80],
            "all_divergences_count": len(divergences),
        }
        out_path = REVIEW_DIR / "corpus_results.json"
        out_path.write_text(json.dumps(out_report, indent=2) + "\n")
        print(json.dumps(summary_counts, indent=2))
        if indep_mismatches:
            print("INDEPENDENT FOLD MISMATCHES:", json.dumps(indep_mismatches, indent=2))
            return 1
        if out_report["cat_c"]:
            print("CATEGORY C DIVERGENCES:", json.dumps(out_report["cat_c"], indent=2))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
