"""W2.1 item 1: the whole-field season-best census (design r1 edit B-E5).

The board is called a census only when the retained receipts verify it: the grab's own status says census; every
board request was an HTTP 200 whose raw archive hashes to its recorded ``archived_sha256``; the board parquet hashes to
its recorded sha and equals the rows re-parsed from the raw pages; and the walk recomputed from the raw pages is
exhausted, conflict-free, with a stable valid participant count and one valid identical ``updatedAt`` on every page,
listed == reported (``final_leaderboard_grab._population``, reused, not re-implemented). Equal row and participant
counts alone are insufficient. Otherwise threshold counts are observed lower bounds and percentiles are unavailable.

Season best is the C-01 tab-semantic ``season_best_streak`` of the all_season walk; a missing value is counted, never
imputed 0. No sampling bootstrap is attached to the census distribution."""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from bts.leaderboard.scraper import LeaderboardEnvelopeError, parse_leaderboard_response
from scripts.final_leaderboard_grab import _population, _validate_page_meta

THRESHOLDS = (20, 30, 40)
CITATIONS = {
    "C-01": "docs/audit/2026-09-corrections-index.md C-01: historical all_season/all_time snapshot rows held the "
            "ACTIVE streak; the 9/27 walk was parsed after the fix (season_best_streak tab-semantic).",
    "C-04/X-17": "docs/audit/2026-09-corrections-index.md C-04 and register X-17: the 9/28 post-cutoff board-only "
                 "pass was identical to final_grab_20260927 for every user (final-board equality evidence).",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_board_receipts(grab_dir: Path) -> dict:
    """Everything the gate reads from a final-grab directory, with the sha256 of every file read."""
    grab_dir = Path(grab_dir)
    files: dict[str, str] = {}
    status_bytes = (grab_dir / "status.json").read_bytes()
    files["status.json"] = _sha(status_bytes)
    status = json.loads(status_bytes)
    entries = [e for e in status.get("requests", []) if e.get("class") == "board"]
    pages = []
    for e in entries:
        path = grab_dir / e["raw_path"]
        raw = path.read_bytes() if path.exists() else None
        if raw is not None:
            files[e["raw_path"]] = _sha(raw)
        body = gzip.decompress(raw) if raw is not None else None
        pages.append({"name": e["name"], "entry": e, "exists": raw is not None, "body": body})
    on_disk = sorted(str(p.relative_to(grab_dir)) for p in (grab_dir / "raw" / "board").glob("*.json.gz"))
    art = (status.get("artifacts") or {}).get("leaderboard_snapshot") or {}
    board_rel = art.get("path")
    board_bytes = (grab_dir / board_rel).read_bytes() if board_rel and (grab_dir / board_rel).exists() else None
    if board_bytes is not None:
        files[board_rel] = _sha(board_bytes)
    board = pd.read_parquet(grab_dir / board_rel) if board_bytes is not None else pd.DataFrame()
    return {"status": status, "pages": pages, "raw_files_on_disk": on_disk, "board_relpath": board_rel,
            "board_sha256": _sha(board_bytes) if board_bytes is not None else None, "board": board, "files": files}


def recompute_walk(bodies: list[bytes | None], http: list[int | None], limit: int) -> tuple[dict, dict]:
    """The grab's walk/termination logic replayed over the archived page bodies, in request order. Returns the board
    record ``_population`` reads and {user_id: (rank, season_best)} from the raw rows."""
    board = {"unique_user_ids": 0, "per_page": [], "walk_exhausted": False, "duplicate_conflicts": [],
             "termination_reason": None}
    rows_by_id: dict[int, tuple] = {}
    signatures: set[str] = set()
    participants: list[int] = []
    for k, (body, code) in enumerate(zip(bodies, http), start=1):
        if code != 200 or body is None:
            board["termination_reason"] = "error"
            break
        try:
            obj = json.loads(body)
            rows = parse_leaderboard_response(obj, tab="all_season", captured_at=datetime(1970, 1, 1))
        except (ValueError, LeaderboardEnvelopeError):
            board["termination_reason"] = "error"
            break
        meta = _validate_page_meta(obj["success"])
        if meta["all_participants_count"] is not None:
            participants.append(meta["all_participants_count"])
        for r in rows:
            prev = rows_by_id.get(r.user_id)
            if prev is not None and prev != (r.rank, r.season_best_streak):
                board["duplicate_conflicts"].append(r.user_id)
            rows_by_id.setdefault(r.user_id, (r.rank, r.season_best_streak))
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
    board["unique_user_ids"] = len(rows_by_id)
    board["population"] = _population(board, participants)
    return board, rows_by_id


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
    walk, raw_rows = recompute_walk([p["body"] for p in pages], [p["entry"].get("http_status") for p in pages],
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
    checks["board_rows_equal_raw"] = bool(out_rows) and out_rows == raw_rows
    checks["board_rows_equal_reported"] = pop.get("reported") is not None and len(board) == pop.get("reported")
    failures = [k if k != "recomputed_population" else f"recomputed_population:{pop['status']}"
                for k, v in checks.items() if not v]
    return {"census": not failures, "checks": checks, "failures": failures, "reported": pop.get("reported"),
            "listed": int(len(board)), "recomputed_population": pop,
            "retained_population": sb.get("population"), "termination_reason": walk["termination_reason"]}


def season_best_summary(board: pd.DataFrame, gate: dict, our_user_id: int | None) -> dict:
    """Distribution, thresholds, max/tie, listing floor and our ID-bound rank/percentile. ``N`` and the percentile
    interval exist only under a census; missing season-best values widen the interval's upper end (their position is
    unknown): [100·below/N, 100·(below+equal+missing)/N], which is the design's interval when nothing is missing."""
    census = bool(gate.get("census"))
    best = pd.to_numeric(board["season_best_streak"], errors="coerce") if len(board) else pd.Series(dtype=float)
    missing = int(best.isna().sum())
    vals = best.dropna().astype(int)
    N = int(gate["reported"]) if census else None
    dist = vals.value_counts().sort_index()
    out = {"basis": "census" if census else "observed_lower_bound", "N": N, "listed": int(len(board)),
           "missing_season_best": missing,
           "distribution": [{"season_best": int(k), "users": int(v)} for k, v in dist.items()],
           "thresholds": {f"ge_{t}": int((vals >= t).sum()) for t in THRESHOLDS},
           "thresholds_note": ("exact census counts; missing values could add at most missing_season_best"
                               if census else "observed lower bounds (board not verified as a census)"),
           "max": int(vals.max()) if len(vals) else None,
           "tie_at_max": int((vals == vals.max()).sum()) if len(vals) else None,
           "listing_floor": int(vals.min()) if len(vals) else None,
           "citations": CITATIONS, "bootstrap": "none (a census distribution has no sampling interval)"}
    own = {"user_id": our_user_id, "matched_by_user_id": False, "season_best": None, "stored_rank": None,
           "below": None, "equal": None, "above": None, "percentile_interval": None, "tied_stored_ranks": None,
           "rank_note": "the board's stored rank (its own tie semantics); not a percentile convention"}
    match = board[board["user_id"] == our_user_id] if our_user_id is not None and len(board) else board.iloc[0:0]
    if len(match) == 1 and pd.notna(match["season_best_streak"].iloc[0]):
        ours = int(match["season_best_streak"].iloc[0])
        below, equal, above = int((vals < ours).sum()), int((vals == ours).sum()), int((vals > ours).sum())
        own.update(matched_by_user_id=True, season_best=ours, stored_rank=int(match["rank"].iloc[0]),
                   below=below, equal=equal, above=above,
                   tied_stored_ranks=sorted({int(r) for r in board.loc[best == ours, "rank"]}),
                   percentile_interval=[100 * below / N, 100 * (below + equal + missing) / N] if census else None)
    out["own"] = own
    return out
