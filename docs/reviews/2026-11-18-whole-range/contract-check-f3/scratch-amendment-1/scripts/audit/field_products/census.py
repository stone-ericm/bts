"""W2.1 item 1: the whole-field season-best census (design r1 edit B-E5; code review r1 F5).

The board is a census only when the retained receipts verify it. Two kinds of failure are kept apart:
- INTEGRITY (the evidence itself is wrong): a raw page archive that does not hash to its recorded
  ``archived_sha256``, an unexpected page set, a board parquet that does not hash to its recorded sha, is not unique
  by user id, is not all ``all_season`` or differs from the rows re-parsed from the raw pages, conflicting duplicate
  rows for one user id in the raw walk, and any id excluded by qualification (review r2 R2-2: a name-only duplicate);
- COVERAGE (the evidence is sound but incomplete): the grab's own status not census, a failed request, a walk not
  exhausted, an unstable/invalid participant count or ``updatedAt`` (``final_leaderboard_grab._population``, reused),
  listed != reported for the board or for the QUALIFIED rows actually summarised.
Summaries are always computed from QUALIFIED raw rows — rows of hash-verified raw pages, exact duplicates collapsed,
user ids with conflicting duplicates excluded and counted — never from the board parquet, which is only checked.
Basis: ``census`` (no failure: N, percentile interval, exact threshold counts); ``observed_lower_bound`` (coverage
failure only); ``qualified_raw_lower_bound`` (integrity failure). Percentiles exist only under a census.

Season best is the C-01 tab-semantic ``season_best_streak``; a missing value is counted, never imputed 0: threshold
counts are known-value counts with an upper bound (known + missing) only under a census, the maximum is the maximum of
known values, and the field maximum is reported only when nothing is missing. No sampling bootstrap is attached."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from bts.leaderboard.scraper import LeaderboardEnvelopeError, parse_leaderboard_response
from scripts.final_leaderboard_grab import _population, _validate_page_meta

THRESHOLDS = (20, 30, 40)
INTEGRITY = ("raw_page_hashes", "raw_page_set", "board_output_hash", "board_unique_ids", "board_tab_all_season",
             "board_rows_equal_raw", "recomputed_conflict_free", "qualified_conflict_free")
COVERAGE = ("retained_status_census", "board_requests_ok", "recomputed_walk_exhausted", "recomputed_population",
            "board_rows_equal_reported", "qualified_rows_equal_reported")
CITATIONS = {
    "C-01": "docs/audit/2026-09-corrections-index.md C-01: historical all_season/all_time snapshot rows held the "
            "ACTIVE streak; the 9/27 walk was parsed after the fix (season_best_streak tab-semantic).",
    "C-04/X-17": "docs/audit/2026-09-corrections-index.md C-04 and register X-17: the 9/28 post-cutoff board-only "
                 "pass was identical to final_grab_20260927 for every user (final-board equality evidence).",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_board_receipts(grab_dir: Path, read=None) -> dict:
    """Everything the gate reads from a final-grab directory, through ``read`` (the frozen bytes; default: disk)."""
    grab_dir = Path(grab_dir)
    read = read or (lambda p: Path(p).read_bytes())
    status = json.loads(read(grab_dir / "status.json"))
    pages = []
    for e in [e for e in status.get("requests", []) if e.get("class") == "board"]:
        try:
            raw = read(grab_dir / e["raw_path"])
        except FileNotFoundError:
            raw = None
        body = gzip.decompress(raw) if raw is not None else None
        pages.append({"name": e["name"], "entry": e, "exists": raw is not None, "body": body})
    on_disk = sorted(str(p.relative_to(grab_dir)) for p in (grab_dir / "raw" / "board").glob("*.json.gz"))
    art = (status.get("artifacts") or {}).get("leaderboard_snapshot") or {}
    board_rel = art.get("path")
    try:
        board_bytes = read(grab_dir / board_rel) if board_rel else None
    except FileNotFoundError:
        board_bytes = None
    board = pd.read_parquet(io.BytesIO(board_bytes)) if board_bytes is not None else pd.DataFrame()
    return {"status": status, "pages": pages, "raw_files_on_disk": on_disk, "board_relpath": board_rel,
            "board_sha256": _sha(board_bytes) if board_bytes is not None else None, "board": board}


def _parse(body: bytes | None):
    try:
        obj = json.loads(body)
        return obj, parse_leaderboard_response(obj, tab="all_season", captured_at=datetime(1970, 1, 1))
    except (TypeError, ValueError, LeaderboardEnvelopeError):
        return None, None


def recompute_walk(bodies: list[bytes | None], http: list[int | None], limit: int) -> tuple[dict, dict]:
    """The grab's walk/termination logic replayed over the archived page bodies, in request order. Returns the board
    record ``_population`` reads and {user_id: (rank, season_best)} at each id's first occurrence (as the grab kept)."""
    board = {"unique_user_ids": 0, "per_page": [], "walk_exhausted": False, "duplicate_conflicts": [],
             "termination_reason": None}
    first: dict[int, tuple] = {}
    signatures: set[str] = set()
    participants: list[int] = []
    for k, (body, code) in enumerate(zip(bodies, http), start=1):
        obj, rows = _parse(body) if code == 200 else (None, None)
        if rows is None:
            board["termination_reason"] = "error"
            break
        meta = _validate_page_meta(obj["success"])
        if meta["all_participants_count"] is not None:
            participants.append(meta["all_participants_count"])
        for r in rows:
            prev = first.get(r.user_id)
            if prev is not None and prev != (r.rank, r.season_best_streak):
                board["duplicate_conflicts"].append(r.user_id)
            first.setdefault(r.user_id, (r.rank, r.season_best_streak))
        board["per_page"].append({"page": k, "all_participants_count": meta["all_participants_count"],
                                  "updated_at": meta["updated_at"]})
        sig = _sha(",".join(str(u) for u in sorted(r.user_id for r in rows)).encode()) if rows else f"empty-{k}"
        if sig in signatures:
            board["termination_reason"] = "repeated_page"
            break
        signatures.add(sig)
        raw_count = len(obj["success"].get("ranks") or [])
        terminal = meta["next_page"] is False     # the grab: a short page needs nextPage False to be terminal
        if raw_count < limit and not terminal:
            board["termination_reason"] = "short_page_not_terminal"
            break
        if terminal:
            board["walk_exhausted"] = k == len(bodies)
            board["termination_reason"] = "terminal" if k == len(bodies) else "pages_after_terminal"
            break
    board["unique_user_ids"] = len(first)
    board["population"] = _population(board, participants)
    return board, first


def qualified_rows(pages: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Rows of hash-verified, parseable raw pages; exact duplicates collapsed; ids with conflicting rows excluded."""
    rows, stats = [], {"pages_excluded_hash": 0, "pages_excluded_unparseable": 0, "rows_excluded_unverified": 0}
    for p in pages:
        ok = p["exists"] and _sha(p["body"]) == p["entry"].get("archived_sha256")
        obj, parsed = _parse(p["body"]) if p["exists"] and p["entry"].get("http_status") == 200 else (None, None)
        if not ok:
            stats["pages_excluded_hash"] += 1
            stats["rows_excluded_unverified"] += len(parsed or [])
            continue
        if parsed is None:
            stats["pages_excluded_unparseable"] += int(p["entry"].get("http_status") == 200)
            continue
        rows += [(r.user_id, r.rank, r.season_best_streak, r.username) for r in parsed]
    df = pd.DataFrame(rows, columns=["user_id", "rank", "season_best_streak", "username"]).drop_duplicates()
    conflicting = set(df.loc[df["user_id"].duplicated(keep=False), "user_id"])
    stats.update(conflicting_ids_excluded=len(conflicting),
                 rows_excluded_conflicting=int(df["user_id"].isin(conflicting).sum()))
    df = df[~df["user_id"].isin(conflicting)].reset_index(drop=True)
    df["season_best_streak"] = pd.array(df["season_best_streak"], dtype="Int64")
    return df, stats


def census_gate(rec: dict) -> dict:
    status, pages, board = rec["status"], rec["pages"], rec["board"]
    sb = status.get("board") or {}
    checks: dict[str, bool] = {}
    checks["retained_status_census"] = bool((sb.get("population") or {}).get("status") == "census"
                                            and sb.get("walk_complete") is True and sb.get("population_complete") is True)
    checks["board_requests_ok"] = bool(pages) and all(p["entry"].get("http_status") == 200
                                                      and p["entry"].get("outcome") == "success" for p in pages)
    checks["raw_page_hashes"] = bool(pages) and all(p["exists"] and _sha(p["body"]) == p["entry"].get("archived_sha256")
                                                    for p in pages)
    checks["raw_page_set"] = sorted(p["entry"]["raw_path"] for p in pages) == rec["raw_files_on_disk"] and \
        [p["name"] for p in pages] == [f"page_{k:03d}" for k in range(1, len(pages) + 1)]
    art = (status.get("artifacts") or {}).get("leaderboard_snapshot") or {}
    checks["board_output_hash"] = rec["board_sha256"] is not None and rec["board_sha256"] == art.get("sha256")
    walk, raw_first = recompute_walk([p["body"] for p in pages], [p["entry"].get("http_status") for p in pages],
                                     int(sb.get("limit") or 300))
    checks["recomputed_walk_exhausted"] = bool(walk["walk_exhausted"])
    checks["recomputed_conflict_free"] = not walk["duplicate_conflicts"]
    pop = walk["population"]
    checks["recomputed_population"] = pop["status"] == "census"
    ids = board["user_id"] if len(board) else pd.Series(dtype="Int64")
    checks["board_unique_ids"] = bool(len(board)) and ids.notna().all() and ids.is_unique
    checks["board_tab_all_season"] = bool(len(board)) and (board["tab"] == "all_season").all()
    out_rows = {int(r.user_id): (int(r.rank), None if pd.isna(r.season_best_streak) else int(r.season_best_streak))
                for r in board.itertuples()} if checks["board_unique_ids"] else {}
    checks["board_rows_equal_raw"] = bool(out_rows) and out_rows == raw_first
    checks["board_rows_equal_reported"] = pop.get("reported") is not None and len(board) == pop.get("reported")
    # R2-2: the census must cover the qualified rows actually summarised (an excluded id is never silently certified)
    qualified, qstats = qualified_rows(pages)
    checks["qualified_conflict_free"] = qstats["conflicting_ids_excluded"] == 0
    checks["qualified_rows_equal_reported"] = pop.get("reported") is not None and len(qualified) == pop["reported"]
    failures = [k if k != "recomputed_population" else f"recomputed_population:{pop['status']}"
                for k, v in checks.items() if not v]
    integrity_ok = all(checks[k] for k in INTEGRITY)
    coverage_ok = all(checks[k] for k in COVERAGE)
    return {"census": integrity_ok and coverage_ok, "integrity_ok": integrity_ok, "coverage_ok": coverage_ok,
            "checks": checks, "failures": failures, "reported": pop.get("reported"), "listed": int(len(qualified)),
            "board_parquet_rows": int(len(board)), "recomputed_population": pop,
            "retained_population": sb.get("population"), "termination_reason": walk["termination_reason"],
            "qualification": qstats, "qualified": qualified}


def season_best_summary(rows: pd.DataFrame, gate: dict, our_user_id: int | None) -> dict:
    """Distribution, thresholds, maxima, listing floor and our ID-bound rank/percentile over the QUALIFIED raw rows.
    ``N``, threshold upper bounds and the percentile interval exist only under a census; missing season-best values
    widen the interval's upper end: [100·below/N, 100·(below+equal+missing)/N]."""
    census = bool(gate.get("census"))
    basis = "census" if census else ("observed_lower_bound" if gate.get("integrity_ok", True)
                                     else "qualified_raw_lower_bound")
    best = pd.to_numeric(rows["season_best_streak"], errors="coerce") if len(rows) else pd.Series(dtype=float)
    missing = int(best.isna().sum())
    vals = best.dropna().astype(int)
    N = int(gate["reported"]) if census else None
    dist = vals.value_counts().sort_index()
    max_known = int(vals.max()) if len(vals) else None
    out = {"basis": basis, "N": N, "listed": int(len(rows)), "missing_season_best": missing,
           "distribution": [{"season_best": int(k), "users": int(v)} for k, v in dist.items()],
           "thresholds": {f"ge_{t}": {"known": int((vals >= t).sum()),
                                      "upper_bound": int((vals >= t).sum()) + missing if census else None}
                          for t in THRESHOLDS},
           "thresholds_note": ("census: known counts, each at most upper_bound (missing values unplaced)" if census
                               else "lower bounds from qualified raw rows (board not verified as a census)"),
           "max_known": max_known, "tie_at_max_known": int((vals == max_known).sum()) if len(vals) else None,
           "field_max": max_known if census and missing == 0 else None,
           "listing_floor": int(vals.min()) if len(vals) else None,
           "citations": CITATIONS, "bootstrap": "none (a census distribution has no sampling interval)"}
    own = {"user_id": our_user_id, "matched_by_user_id": False, "season_best": None, "stored_rank": None,
           "below": None, "equal": None, "above": None, "percentile_interval": None, "tied_stored_ranks": None,
           "rank_note": "the board's stored rank (its own tie semantics); not a percentile convention"}
    match = rows[rows["user_id"] == our_user_id] if our_user_id is not None and len(rows) else rows.iloc[0:0]
    if len(match) == 1 and pd.notna(match["season_best_streak"].iloc[0]):
        ours = int(match["season_best_streak"].iloc[0])
        below, equal, above = int((vals < ours).sum()), int((vals == ours).sum()), int((vals > ours).sum())
        own.update(matched_by_user_id=True, season_best=ours, stored_rank=int(match["rank"].iloc[0]),
                   below=below, equal=equal, above=above,
                   tied_stored_ranks=sorted({int(r) for r in rows.loc[best == ours, "rank"]}),
                   percentile_interval=[100 * below / N, 100 * (below + equal + missing) / N] if census else None)
    out["own"] = own
    return out
