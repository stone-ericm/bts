"""Synthetic inputs for the W2.1/W2.2 field-product tests (no tests here). A final-grab directory is produced by the
REAL grab (`run_grab`) over the scripted fake transport, so status.json, raw page receipts, cohort.json, identity.json
and id-keyed pick files have exactly the on-disk formats the analysis reads."""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timezone
from pathlib import Path

from bts.leaderboard.models import PickRow
from bts.leaderboard.storage import append_user_picks
from scripts.final_leaderboard_grab import GrabConfig, run_grab
from tests.scripts.test_final_leaderboard_grab import FakeTransport, _board_page, _rank_row

UPDATED = "2026-09-27T18:51:54-04:00"
GRAB_NOW = datetime(2026, 9, 27, 23, 0, tzinfo=timezone.utc)
ROUND_DATES = {1000 + i: date(2026, 9, 18 + i) for i in range(10)}


def board_rows(n: int, *, best=lambda i: max(0, 30 - i // 10), names=None) -> list[dict]:
    """n rank rows, user ids 1000+i, competition ranks from the season best (ties share a rank)."""
    vals = [best(i) for i in range(n)]
    rows = []
    for i, v in enumerate(vals):
        rank = 1 + sum(1 for w in vals if w > v)
        rows.append(_rank_row(1000 + i, rank, v, username=(names or {}).get(1000 + i)))
    return rows


def pages(rows: list[dict], *, participants=None, updated=UPDATED, updated_per_page=None) -> list[tuple]:
    chunks = [rows[i:i + 300] for i in range(0, len(rows), 300)]
    out = []
    for k, chunk in enumerate(chunks):
        upd = updated_per_page[k] if updated_per_page else updated
        out.append((200, _board_page(chunk, next_page=k < len(chunks) - 1,
                                     participants=len(rows) if participants is None else participants, updated=upd)))
    return out


def manifest(path: Path, members: list[tuple[int, str]]) -> Path:
    users = [{"order": i + 1, "user_id": uid, "usernames_2026_05_01": [name], "ranks_2026_05_01": {"all_season": i + 1},
              "streak_fields_2026_05_01": {}} for i, (uid, name) in enumerate(members)]
    path.write_text(json.dumps({"schema_version": "bts_early_cohort_manifest_v1", "frozen_at": "2026-09-22",
                                "definition": "synthetic", "n_users": len(users), "users": users,
                                "source_fixture_sha256": {"x": "0" * 64}}))
    return path


def pred(round_id: int, streak, slots: list[tuple]) -> dict:
    """slots: (number, unitId, playerId, result)."""
    return {"roundId": round_id, "streak": streak, "result": None, "streakIncrease": None,
            "roundPredictions": [{"number": n, "unitId": u, "playerId": p, "result": r,
                                  "atBats": None if r is None else 4, "hits": None if r is None else 1}
                                 for n, u, p, r in slots]}


def profile(preds: list[dict], best=5) -> dict:
    return {"success": {"seasonBestStreak": best, "activeStreak": 0, "accuracy": 55, "favouriteBatter": None,
                        "predictions": preds}}


STATICS_BASE = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game"


def statics() -> dict:
    rounds = [{"id": rid, "date": f"{d.isoformat()}T08:00:00-04:00"} for rid, d in ROUND_DATES.items()]
    players = [{"id": p, "squadId": 1 + p % 2, "feedId": 600000 + p, "name": f"P{p}"} for p in range(1, 40)]
    units = [{"id": 5, "homeSquadId": 1, "awaySquadId": 2}]
    return {f"{STATICS_BASE}/json/rounds.json": {"rounds": rounds},
            f"{STATICS_BASE}/json/players.json": {"players": players},
            f"{STATICS_BASE}/json/units.json": {"units": units},
            f"{STATICS_BASE}/json/squads.json": {"squads": [{"id": 1, "abbreviation": "NYM"},
                                                           {"id": 2, "abbreviation": "ATL"}]}}


def make_grab(root: Path, *, board_pages, early: list[tuple[int, str]], profiles: dict | None = None,
              cohort_a=2, cohort_b=2) -> Path:
    """Run the real grab against the fake transport; return the final_grab_20260927 directory."""
    lb = root / "data" / "leaderboard"
    lb.mkdir(parents=True, exist_ok=True)
    early_path = manifest(root / "early.json", early)
    t = FakeTransport(board_pages=board_pages, profiles=profiles or {}, statics=statics())
    cfg = GrabConfig(date="2026-09-27", season=2026, leaderboard_dir=lb, run_root=lb / "final_grab_20260927",
                     early_cohort_path=early_path, cohort_a=cohort_a, cohort_b=cohort_b, transport=t,
                     cookies_loader=lambda: ({"oktaid": "abc"}, {"source": "test", "sha256": "0" * 64}),
                     sleeper=lambda s: None, rng=random.Random(7), now=lambda: GRAB_NOW, code_sha="deadbeef",
                     final_round_date=date(2026, 9, 27))
    run_grab(cfg)
    return cfg.run_root


def make_ledger(root: Path, rows: list[dict], contest_rows: list[dict]) -> Path:
    """A synthetic accepted W1.1 build directory in the compiler's exact formats: the six build files (schemas from
    season_ledger.compile), build.json counts computed like the compiler's, and the accept step's ACCEPTED.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from collections import Counter

    from scripts.audit.field_products import ledger as LG
    from scripts.audit.season_ledger.compile import CONTEST_SCHEMA, LEDGER_SCHEMA

    d = root / LG.ACCEPTED_BUILD
    d.mkdir(parents=True)
    full = lambda rs, schema: pa.Table.from_pylist([{f.name: r.get(f.name) for f in schema} for r in rs],
                                                   schema=schema)
    pq.write_table(full(rows, LEDGER_SCHEMA), d / "season_2026_ledger.parquet")
    pq.write_table(full(contest_rows, CONTEST_SCHEMA), d / "season_2026_ledger_contest_slots.parquet")
    for name in ("season_2026_ledger_occurrences.parquet", "season_2026_ledger_reconciliation.parquet",
                 "season_2026_ledger_summary.md"):
        (d / name).write_bytes(b"synthetic")
    counts = lambda rs, k: dict(sorted(Counter(str(r.get(k)) for r in rs).items()))
    sels = [r for r in rows if r.get("row_kind") == "selection"]
    build = {"code_sha": LG.ACCEPTED_CODE_SHA, "rules_fingerprint": LG.EXPECTED_RULES_FINGERPRINT,
             "builder_version": "season-ledger-phase1/3", "row_kinds": counts(rows, "row_kind"),
             "commit_status": counts(sels, "commit_status"), "entry_status": counts(sels, "entry_status"),
             "match": counts(contest_rows, "match")}
    (d / "season_2026_ledger_build.json").write_text(json.dumps(build, indent=1, sort_keys=True) + "\n")
    receipt = {"run": LG.ACCEPTED_RUN, "accepted_at_utc": "2026-09-29T02:29:08+00:00",
               "files": sorted(LG.BUILD_FILES), "compared_with": "/tmp/ledger_check", "rules_fingerprint":
               LG.EXPECTED_RULES_FINGERPRINT}
    (d / "ACCEPTED.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return d


def ledger_selection(sid, d, slot, grade, *, match="evidenced", round_id=1, unit=1) -> dict:
    return {"row_id": sid, "row_kind": "selection", "date": d, "round_id": round_id, "slot": slot,
            "selection_id": sid, "batter_id": 5, "game_pk": 9, "commit_status": "committed_evidenced",
            "entry_status": "confirmed", "match": match, "match_reason": "unit_capture", "unit_id": unit,
            "contest_slot_grade_raw": grade}


def daily_pick(round_id, pick_number, *, cap, pick_date, result="hit", unit=10, player=20, streak=1,
               team="NYM", ha="home") -> PickRow:
    return PickRow(captured_at=cap, round_id=round_id, pick_date=pick_date, pick_number=pick_number, unit_id=unit,
                   bts_player_id=player, result=result, at_bats=4, hits=1 if result == "hit" else 0,
                   streak_after=streak, batter_id=600000 + player, batter_name="B", batter_team=team,
                   opponent_team="ATL", home_or_away=ha)


def write_daily(dir_: Path, stem: str, batches: list[list[PickRow]]) -> Path:
    """Append each batch (one daily capture) to user_picks/<stem>.parquet with the real writer."""
    p = dir_ / f"{stem}.parquet"
    for b in batches:
        append_user_picks(p, b)
    return p
