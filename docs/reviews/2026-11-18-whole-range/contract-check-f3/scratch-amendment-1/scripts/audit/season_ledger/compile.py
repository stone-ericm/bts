"""Offline compilation from a sealed bundle (spec §3–§10). Reads only the bundle; no network. Every table is
built (and type-checked) before any file is written, and each build writes a fresh output directory."""
from __future__ import annotations

import json
import platform
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa

from . import BUILDER_VERSION
from .bundle import open_bundle
from .contest import (entered_rounds, line_round_streaks, match_slot, players_lookup, resolve_duplicate_links,
                      rounds_lookup, slot_history, streak_before, team_games, unit_status_history, units_lookup)
from .ids import canonical_json, sha256_hex, utc_iso
from .io import build_table, write_table
from .outcomes import derived_single_result, normalize_contest, normalize_local, slot_disagreement
from .reconcile import (PARSERS, RULES, account, assign_dispositions, check_invariants, check_sources,
                        evaluate_rules, recipe_labels, route, rules_fingerprint)
from .rows import day_rows
from .sources.pick_files import pick_file_state

SEASON_DATES = [(date(2026, 3, 25) + timedelta(days=i)).isoformat() for i in range(187)]   # 3/25 → 9/27
_SEASON = frozenset(SEASON_DATES)
POSTPONED_UNIT_STATUSES = frozenset({"postponed"})
DAY_KINDS = frozenset({"pick_file", "archive", "manual_archive", "repair_archive", "decision", "scheduler_state",
                       "lineup_evolution"})
OBSERVATION_KINDS = ("archive", "manual_archive", "repair_archive", "lineup_evolution")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
S, I, B, F = pa.string(), pa.int64(), pa.bool_(), pa.float64()

LEDGER_SCHEMA = pa.schema([
    ("row_id", S), ("row_kind", S), ("date", S), ("round_id", I), ("slot", S), ("selection_id", S), ("reason", S),
    ("scheduled_games", I),
    ("batter_id", I), ("batter_name", S), ("team_at_pick", S), ("game_pk", I), ("game_time", S),
    ("lineup_position", I), ("projected_lineup", B), ("pitcher_id", I), ("p_stated", F),
    ("finalization", S), ("pick_view_batter_id", I), ("pick_view_game_pk", I), ("pick_view_action", S),
    ("pick_file_complete", B),
    ("commit_status", S), ("commit_basis", S), ("history_status", S), ("scheduler_commit_flag", B),
    ("action", S), ("action_source_raw", S), ("action_source", S), ("objective", S), ("pick_policy_objective", S),
    ("degraded_reason", S), ("decision_streak", I), ("decision_state_source", S), ("decision_state_status", S),
    ("declined_batter_id", I), ("declined_game_pk", I),
    ("predicted_at", S), ("locked_at", S), ("delivery_attempted", B), ("delivery_attempted_at", S),
    ("delivery_confirmed", B), ("delivery_basis", S), ("delivery_evidence_conflict", B), ("delivered_at", S),
    ("game_eligibility", S), ("game_eligibility_at", S), ("game_eligibility_basis", S),
    ("entry_status", S), ("match", S), ("match_reason", S), ("unit_id", I), ("player_id", I), ("slot_number", I),
    ("entry_observed_at", S), ("contest_slot_grade_raw", S), ("bts_outcome", S), ("bts_outcome_status", S),
    ("contest_round_result", S), ("streak_before", I), ("streak_after", I), ("saver_available_before", B),
    ("local_slot_result_raw", S), ("local_day_result_raw", S), ("local_slot_result_derived", S),
    ("derivation_source", S), ("local_norm", S), ("contest_norm", S), ("local_vs_contest_disagreement", B),
    ("decision_obs_id", S), ("pick_obs_id", S), ("pick_view_obs_id", S), ("state_obs_id", S), ("contest_obs_id", S),
])
CONTEST_SCHEMA = pa.schema([
    ("round_id", I), ("unit_id", I), ("player_id", I), ("date", S), ("batter_id", I), ("game_pk", I),
    ("selection_id", S), ("match", S), ("match_reason", S), ("first_seen", S), ("last_seen", S),
    ("n_observations", I), ("changed", B), ("dropped_later", B), ("slot_number", I), ("slot_result", S),
    ("slot_result_state", S),
    ("hits", I), ("hits_state", S), ("at_bats", I), ("at_bats_state", S), ("round_result", S),
    ("round_streak", I), ("round_streak_increase", I), ("last_obs_id", S)])
OCCURRENCE_SCHEMA = pa.schema([("source_path", S), ("locator", S), ("obs_id", S), ("kind", S), ("state", S),
                               ("reason", S), ("disposition", S), ("content_sha256", S), ("fields_json", S),
                               ("record_raw_json", S)])
RECONCILIATION_SCHEMA = pa.schema([
    ("rule_id", S), ("recipe", S), ("source_path", S), ("slot", S), ("recipe_slot_key", S), ("file_date", S),
    ("recipe_value", S), ("included", B), ("exclusion_reason", S), ("historical_membership", S),
    ("source_mtime_utc", S), ("mtime_after_recipe_date", B), ("frozen_at_utc", S), ("source_obs_id", S),
    ("source_state", S), ("source_reason", S), ("occurrence_disposition", S), ("selection_id", S),
    ("canonical_bts_outcome", S)])


def _group(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return out


def _counts(rows, key: str) -> dict:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def _eligibility(row: dict, match: dict | None, unit_status: dict, refusals: dict) -> tuple:
    """§9 and Interpretation I6: only evidence timed before a KNOWN lock counts, and the latest such evidence
    decides. A refusal binds when its archive names this date, slot, batter and game, carries a reason, and was
    written before the lock; without a known lock nothing can be shown to precede it."""
    locked = row["locked_at"]
    if locked is None:
        return "unknown", None, None
    if match is not None and match["match"] == "evidenced":
        before = [(t, s) for t, s in unit_status.get(match["unit_id"], []) if t is not None and t < locked]
        if before and before[-1][1] in POSTPONED_UNIT_STATUSES:
            return "postponed_evidenced", before[-1][0], "unit_capture_status_postponed"
    timed = sorted((t, reason) for t, reason in refusals.get((row["date"], row["slot"], row["batter_id"],
                                                                row["game_pk"]), [])
                   if t is not None and reason and t < locked)
    if timed:
        return "refused_evidenced", timed[-1][0], timed[-1][1]
    return "unknown", None, None


def _outcome_status(slot_result, slot_result_state: str) -> str:
    """I11 with I13: a string grade is `graded`; a source null is `matched_ungraded`; a present value of the wrong
    type is `unknown` — never presented as a source null."""
    if slot_result_state == "type_mismatch":
        return "unknown"
    return "graded" if slot_result is not None else "matched_ungraded"


def compile_bundle(bundle_root, out_dir, *, uv_lock_sha256: str | None, code_sha: str | None = None) -> dict:
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out} (each build writes a new directory)")
    manifest, files = open_bundle(bundle_root)
    # Codex code r1 #7: the offset-required time policy applies to manifest mtimes too (a naive one is unknown).
    mtimes = {e["rel_path"]: utc_iso(e.get("source_mtime_utc")) for e in manifest["entries"]}
    routed = {rel: kind for rel, data in files.items() if data is not None and (kind := route(rel))}
    parsed = {rel: PARSERS[kind](rel, files[rel]) for rel, kind in routed.items()}

    def rows_of(*kinds: str) -> list[dict]:
        return [r for rel, k in routed.items() if k in kinds for r in parsed[rel].rows]

    for rel, kind in routed.items():
        if kind in DAY_KINDS:
            m = _DATE.search(rel)
            for r in parsed[rel].rows:
                r["file_date"] = m.group(0) if m else None
    accounting = account(files, routed, parsed)
    check_sources(files, routed, accounting)          # §8 anti-join, before any canonical row exists

    # §5 day rows
    decisions = {r["file_date"]: r for r in rows_of("decision")}
    states = {r["file_date"]: r for r in rows_of("scheduler_state")}
    picks_by_date = _group(rows_of("pick_file"), "file_date")
    pick_state_by_date = {_DATE.search(rel).group(0): pick_file_state(files[rel], parsed[rel])
                          for rel, k in routed.items() if k == "pick_file"}
    obs_by_date = _group(rows_of(*OBSERVATION_KINDS), "file_date")
    unusable_by_date: dict[str, set[str]] = {}       # day evidence present but quarantined (Codex plan r3 #5)
    for rel, kind in routed.items():
        if kind in DAY_KINDS and kind != "pick_file" and parsed[rel].quarantined and (m := _DATE.search(rel)):
            unusable_by_date.setdefault(m.group(0), set()).add(kind)
    ledger: list[dict] = []
    for d in SEASON_DATES:
        ledger += day_rows(d, decision=decisions.get(d), pick_rows=picks_by_date.get(d, []),
                           pick_file=pick_state_by_date.get(d), state=states.get(d),
                           observations=obs_by_date.get(d, []), unusable=frozenset(unusable_by_date.get(d, ())))
    selections = [r for r in ledger if r["row_kind"] == "selection"]

    # §6 contest evidence and matching
    contest_rows = rows_of("contest_ledger")
    streaks_by_line, entered = line_round_streaks(contest_rows), entered_rounds(contest_rows)
    schedule_rows = rows_of("schedule")
    schedule_status = {rel[len("schedules/"):-len(".json")]: "incomplete" if parsed[rel].quarantined else "complete"
                       for rel, k in routed.items() if k == "schedule"}
    games_by_date: dict[str, set] = {}
    for g in schedule_rows:
        games_by_date.setdefault(g["query_date"], set()).add(g["game_pk"])
    rounds = rounds_lookup(rows_of("rounds"))
    lookups = {"rounds": rounds, "players": players_lookup(rows_of("players")),
               "units": units_lookup(rows_of("units")), "team_games": team_games(schedule_rows),
               "schedule_status": schedule_status}
    matches = resolve_duplicate_links([{**h, **match_slot(h, local_selections=selections, **lookups)}
                                       for h in slot_history(contest_rows)])
    round_of_date: dict[str, list[int]] = {}
    for rid, dates in rounds.items():
        if len(dates) == 1:
            round_of_date.setdefault(next(iter(dates)), []).append(rid)

    def day_context(d: str | None) -> dict:
        ids = round_of_date.get(d, [])
        complete = schedule_status.get(d) == "complete"
        return {"round_id": ids[0] if len(ids) == 1 else None,
                "scheduled_games": len(games_by_date.get(d, ())) if complete else None}

    for row in ledger:
        row.update(day_context(row["date"]))

    # §7 outcomes onto selections
    unit_status = unit_status_history(rows_of("units"))
    refusals: dict[tuple, list] = {}
    for r in rows_of("archive"):
        if r["archive_prefix"] == "refused_delivery":
            refusals.setdefault((r["file_date"], r["slot"], r["batter_id"], r["game_pk"]), []).append(
                (utc_iso(r["archived_at"]), r["archive_reason"]))
    picks_by_obs = {r["obs_id"]: r for r in rows_of("pick_file")}
    contest_by_obs = {r["obs_id"]: r for r in contest_rows}
    linked = {m["selection_id"]: m for m in matches if m["selection_id"]}

    def grade_raw(m: dict) -> str | None:
        if m["slot_result_state"] != "type_mismatch":
            return m["slot_result"]
        return canonical_json(json.loads(contest_by_obs[m["last_obs_id"]]["record_raw_json"])["result"])
    unlinked_keys = {(m["date"], m["batter_id"]) for m in matches
                     if not m["selection_id"] and m["match"] in ("ambiguous", "evidenced")}
    for row in selections:
        pick = picks_by_obs.get(row["pick_obs_id"])
        derived, source = derived_single_result(pick)
        row.update(local_slot_result_raw=pick["slot_result_raw"] if pick else None,
                   local_day_result_raw=pick["day_result_raw"] if pick else None,
                   local_slot_result_derived=derived, derivation_source=source, saver_available_before=None)
        row["local_norm"] = normalize_local(row["local_slot_result_raw"] or derived)
        m = linked.get(row["selection_id"])
        eligibility, at, basis = _eligibility(row, m, unit_status, refusals)
        row.update(game_eligibility=eligibility, game_eligibility_at=at, game_eligibility_basis=basis)
        if m is None:
            row["entry_status"] = "unknown"
            row["bts_outcome_status"] = ("match_ambiguous" if (row["date"], row["batter_id"]) in unlinked_keys
                                         else "unknown")
            continue
        row.update(entry_status="confirmed", match=m["match"], match_reason=m["match_reason"], round_id=m["round_id"],
                   unit_id=m["unit_id"], player_id=m["player_id"], slot_number=m["slot_number"],
                   entry_observed_at=m["first_seen"], contest_slot_grade_raw=grade_raw(m),
                   bts_outcome=m["slot_result"], bts_outcome_status=_outcome_status(m["slot_result"],
                                                                                    m["slot_result_state"]),
                   contest_round_result=m["round_result"], streak_after=m["round_streak"],
                   streak_before=streak_before(streaks_by_line.get(m["last_line_no"], {}), m["round_id"], entered),
                   contest_norm=normalize_contest(m["slot_result"]), contest_obs_id=m["last_obs_id"])
        row["local_vs_contest_disagreement"] = slot_disagreement(row["local_norm"], row["contest_norm"])

    # §5 contest-only rows
    for m in matches:
        if m["selection_id"]:
            continue
        status = {"unmapped": "unmapped", "ambiguous": "match_ambiguous"}.get(
            m["match"], _outcome_status(m["slot_result"], m["slot_result_state"]))
        ledger.append({"row_id": f"contest|{m['round_id']}|{m['unit_id']}|{m['player_id']}",
                       "row_kind": "contest_only", "date": m["date"], "slot": None, "selection_id": None,
                       **day_context(m["date"]), "round_id": m["round_id"],
                       "batter_id": m["batter_id"], "game_pk": m["game_pk"], "entry_status": "confirmed",
                       "match": m["match"], "match_reason": m["match_reason"], "unit_id": m["unit_id"],
                       "player_id": m["player_id"], "slot_number": m["slot_number"],
                       "entry_observed_at": m["first_seen"], "contest_slot_grade_raw": grade_raw(m),
                       "bts_outcome": m["slot_result"] if status == "graded" else None, "bts_outcome_status": status,
                       "contest_round_result": m["round_result"], "streak_after": m["round_streak"],
                       "contest_norm": normalize_contest(m["slot_result"]), "contest_obs_id": m["last_obs_id"]})

    # §8 dispositions (canonical links), recipes, then the output-phase checks
    dispositions: dict[str, str] = {}
    for r in rows_of("pick_file"):
        dispositions[r["obs_id"]] = "not_selected" if r["file_date"] in _SEASON else "outside_season_window"
    for r in rows_of("decision"):
        dispositions[r["obs_id"]] = "outside_season_window"
    unresolved_dates = {r["date"] for r in ledger if r.get("finalization") == "unresolved"
                        or r.get("reason") == "decision_unusable"}
    for r in rows_of("pick_file"):
        if r["file_date"] in unresolved_dates:
            dispositions[r["obs_id"]] = "unresolved_pick_file_view"
    for r in ledger:
        if r.get("pick_obs_id"):
            dispositions[r["pick_obs_id"]] = "canonical_selection"
        if r.get("decision_obs_id"):
            dispositions[r["decision_obs_id"]] = "canonical_decision"
    assign_dispositions(accounting, dispositions)
    summary, membership = evaluate_rules(files, RULES, mtimes, frozen_at=manifest["acquired_at_utc"])
    by_locator = {(a["source_path"], a["locator"]): a for a in accounting}
    canonical_of = {r["pick_obs_id"]: r for r in ledger if r.get("pick_obs_id")}
    for m in membership:   # link every recipe row to the occurrence that accounts for its record (Codex plan r2 #5)
        source = by_locator.get((m["source_path"], f"slot={m['slot']}")) or by_locator.get((m["source_path"], "file"))
        link = canonical_of.get(source["obs_id"]) if source else None
        m.update(source_obs_id=source["obs_id"] if source else None, source_state=source["state"] if source else None,
                 source_reason=source["reason"] if source else None,
                 occurrence_disposition=source["disposition"] if source else None,
                 selection_id=link["selection_id"] if link else None,
                 canonical_bts_outcome=link.get("bts_outcome") if link else None)
    check_invariants(files, accounting, matches, ledger, membership, summary, SEASON_DATES)

    tables = {
        "season_2026_ledger.parquet": build_table(ledger, LEDGER_SCHEMA, sort_keys=["row_id"], name="ledger"),
        "season_2026_ledger_occurrences.parquet": build_table(accounting, OCCURRENCE_SCHEMA,
                                                              sort_keys=["source_path", "locator"], name="occurrences"),
        "season_2026_ledger_contest_slots.parquet": build_table(
            [{k: m.get(k) for k in CONTEST_SCHEMA.names} for m in matches], CONTEST_SCHEMA,
            sort_keys=["round_id", "unit_id", "player_id"], name="contest_slots"),
        "season_2026_ledger_reconciliation.parquet": build_table(membership, RECONCILIATION_SCHEMA,
                                                                 sort_keys=["rule_id", "source_path", "slot"],
                                                                 name="reconciliation"),
    }
    sels = [r for r in ledger if r["row_kind"] == "selection"]
    emitted = [a for a in accounting if a["state"] == "emitted"]
    saver = rows_of("saver_transitions")
    build = {"builder_version": BUILDER_VERSION, "code_sha": code_sha, "python_version": platform.python_version(),
             "pyarrow_version": pa.__version__, "environment_lock_sha256": uv_lock_sha256,
             "bundle_manifest_sha256": sha256_hex((Path(bundle_root) / "manifest.json").read_bytes()),
             "bundle_acquired_at_utc": manifest["acquired_at_utc"], "rules_fingerprint": rules_fingerprint(),
             "season_dates": [SEASON_DATES[0], SEASON_DATES[-1]],
             "row_kinds": _counts(ledger, "row_kind"), "finalization": _counts(sels, "finalization"),
             "commit_status": _counts(sels, "commit_status"), "history_status": _counts(sels, "history_status"),
             "entry_status": _counts(sels, "entry_status"),
             "bts_outcome_status": _counts([r for r in ledger if r["row_kind"] in ("selection", "contest_only")],
                                           "bts_outcome_status"),
             "match": _counts(matches, "match"), "match_reason": _counts(matches, "match_reason"),
             "game_eligibility": _counts(sels, "game_eligibility"),
             "local_vs_contest_disagreement": _counts(sels, "local_vs_contest_disagreement"),
             "schedule_status": _counts([{"s": schedule_status.get(d, "missing")} for d in SEASON_DATES], "s"),
             "occurrence_states": _counts(accounting, "state"),
             "exclusion_reasons": _counts([a for a in accounting if a["state"] == "excluded"], "reason"),
             "quarantine_reasons": _counts([a for a in accounting if a["state"] == "quarantined"], "reason"),
             "dispositions": _counts(emitted, "disposition"),
             "type_mismatch_rows_by_kind": _counts([r for rel, k in routed.items() for r in parsed[rel].rows
                                                    if r.get("type_mismatch_fields")], "source_kind"),
             "saver_transition_attempts": {"n": len(saver), "by_outcome": _counts(saver, "attempt_outcome")},
             "recipes": summary, "recipe_labels": recipe_labels(summary)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()                       # reserves the directory: a concurrent or repeated build fails here
    for name, table in tables.items():
        write_table(table, out / name)
    (out / "season_2026_ledger_build.json").write_text(json.dumps(build, indent=1, sort_keys=True) + "\n")
    (out / "season_2026_ledger_summary.md").write_text(_summary_md(build))
    return build


def _summary_md(build: dict) -> str:
    lines = ["# Season 2026 ledger — Phase 1 build summary", "",
             f"Builder `{build['builder_version']}` · code `{build['code_sha']}` · Python {build['python_version']} · "
             f"pyarrow {build['pyarrow_version']} · environment lock `{build['environment_lock_sha256']}` · bundle "
             f"manifest `{build['bundle_manifest_sha256']}` (acquired {build['bundle_acquired_at_utc']}) · recipe "
             f"rules `{build['rules_fingerprint']}`", ""]
    for key in ("row_kinds", "finalization", "commit_status", "history_status", "entry_status", "bts_outcome_status",
                "match", "match_reason", "game_eligibility", "local_vs_contest_disagreement", "schedule_status",
                "occurrence_states", "exclusion_reasons", "quarantine_reasons", "dispositions",
                "type_mismatch_rows_by_kind"):
        lines += [f"## {key}", *(f"- {k}: {v}" for k, v in build[key].items()), ""]
    s = build["saver_transition_attempts"]
    lines += ["## saver_transitions.jsonl (attempts, not consumption times)", f"- rows: {s['n']}",
              *(f"- {k}: {v}" for k, v in s["by_outcome"].items()), "",
              "## Recipe candidates (historical membership unverified)", "",
              "| rule | recipe | files | primary | legs | window | primaries | legs | published | fit |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rule_id']} | {r['recipe']} | {r['files']} | {r['primary_grading']} | {r['leg_grading']} | "
              f"{r['window'][0]}→{r['window'][1]} | {r['primaries']} | {r['legs']} | "
              f"{r['published_primaries']}/{r['published_legs']} | {r['fit']} |" for r in build["recipes"]]
    lines += ["", *(f"- **{k}: {v}**" for k, v in build["recipe_labels"].items()), ""]
    return "\n".join(lines)
