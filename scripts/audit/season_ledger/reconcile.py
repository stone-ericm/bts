"""Path routing, occurrence accounting with an independent structural census (the anti-join), build
invariants, and the pre-declared recipe rules (spec §8)."""
from __future__ import annotations

import inspect
import itertools
import json
import re
import types
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo

from .ids import Parsed, canonical_json, load_json_bytes, obs_id, sha256_hex
from .sources.contest_ledger import parse_contest_ledger, parse_saver_transitions
from .sources.day_records import parse_decision, parse_lineup_evolution, parse_scheduler_state
from .sources.pick_files import parse_archive, parse_pick_file
from .sources.static import parse_players, parse_rounds, parse_schedule, parse_units

_D = r"\d{4}-\d{2}-\d{2}"
_APPLEDOUBLE = re.compile(r"(^|/)\._[^/]*$")
ROUTES = [(re.compile(p), kind) for p, kind in [
    (rf"^picks/{_D}\.json$", "pick_file"),
    (rf"^picks/{_D}/decision\.json$", "decision"),
    (rf"^picks/{_D}/scheduler_state\.json$", "scheduler_state"),
    (rf"^picks/{_D}/(deferred_fallback|refused_delivery|stale_pick)_[^/]+\.json$", "archive"),
    (rf"^picks/archive/{_D}\.json\.postponed$", "manual_archive"),
    (rf"^picks/archive_actual_streak_repair_[^/]+/{_D}\.json(\.before)?$", "repair_archive"),
    (rf"^picks/lineup_evolution_{_D}\.jsonl$", "lineup_evolution"),
    (r"^picks/account_state/contest_ledger\.jsonl$", "contest_ledger"),
    (r"^picks/account_state/saver_transitions\.jsonl$", "saver_transitions"),
    (r"^static/rounds/[^/]+$", "rounds"),
    (r"^static/units/[^/]+$", "units"),
    (r"^static/players/[^/]+$", "players"),
    (r"^static/grab_20260927/[^/]*rounds[^/]*$", "rounds"),
    (r"^static/grab_20260927/[^/]*units[^/]*$", "units"),
    (r"^static/grab_20260927/[^/]*players[^/]*$", "players"),
    (rf"^schedules/{_D}\.json$", "schedule"),
]]
EXCLUSIONS = [(re.compile(p), reason) for p, reason in [
    (r"(^|/)\._[^/]*$", "appledouble_resource_fork"),
    (rf"^picks/{_D}\.shadow\.json$", "shadow_model_out_of_scope"),
    (r"^picks/backup_shadow_[^/]+/", "shadow_model_out_of_scope"),
    (rf"^picks/{_D}\.policy_shadow\.json$", "skip_policy_shadow_out_of_scope"),
    (r"^picks/slates/", "slate_binding_out_of_scope"),
    (r"(^|/)streak(\.before)?\.json$", "state_snapshot_not_a_record"),
    (r"^picks/account_state/(contest_streak|saver_state)[^/]*$", "state_snapshot_not_a_record"),
    (r"(^|/)README\.txt$", "documentation"),
    (r"^picks/\.nrestarts_checkpoint$", "runtime_marker"),
    (r"^logs/", "corroboration_only"),
    (r"^static/grab_20260927/[^/]*squads[^/]*$", "not_used_phase1"),
]]
PARSERS = {
    "pick_file": parse_pick_file, "archive": parse_archive,
    "manual_archive": lambda rel, data: parse_pick_file(rel, data, kind="manual_archive"),
    "repair_archive": lambda rel, data: parse_pick_file(rel, data, kind="repair_archive"),
    "decision": parse_decision, "scheduler_state": parse_scheduler_state,
    "lineup_evolution": parse_lineup_evolution, "contest_ledger": parse_contest_ledger,
    "saver_transitions": parse_saver_transitions, "rounds": parse_rounds, "players": parse_players,
    "units": parse_units, "schedule": parse_schedule,
}
KIND_DISPOSITION = {"archive": "history_evidence", "manual_archive": "history_evidence",
                    "repair_archive": "history_evidence", "lineup_evolution": "history_evidence",
                    "scheduler_state": "day_evidence", "contest_ledger": "contest_evidence",
                    "saver_transitions": "reported_attempt", "rounds": "lookup", "players": "lookup",
                    "units": "lookup", "schedule": "lookup"}   # pick_file / decision come from the ledger
DISPOSITIONS = frozenset(KIND_DISPOSITION.values()) | {"canonical_selection", "unresolved_pick_file_view",
                                                       "not_selected", "outside_season_window", "canonical_decision"}
_OWN_COLUMNS = frozenset({"obs_id", "locator", "source_path", "content_sha256", "record_raw_json"})
_PICK_LIKE = frozenset({"pick_file", "archive", "manual_archive", "repair_archive"})
_ITEM_KEYS = {"rounds": "rounds", "players": "players", "units": "units"}


def route(rel_path: str) -> str | None:
    if _APPLEDOUBLE.search(rel_path):
        return None
    for pattern, kind in ROUTES:
        if pattern.match(rel_path):
            return kind
    return None


def exclusion_reason(rel_path: str) -> str:
    for pattern, reason in EXCLUSIONS:
        if pattern.search(rel_path):
            return reason
    return "unrecognized_path"


def account(files: dict[str, bytes | None], routed: dict[str, str], parsed: dict[str, Parsed]) -> list[dict]:
    """Spec §8 table 1: every occurrence ends emitted (with its parsed fields and raw record), excluded(reason),
    quarantined(reason, with its raw record when it had one) or declared_missing. Dispositions are assigned
    after the ledger exists (`assign_dispositions`)."""
    out = []
    for rel, data in files.items():
        kind = routed.get(rel)
        sha = None if data is None else sha256_hex(data)
        blank = {"source_path": rel, "kind": kind, "content_sha256": sha, "disposition": None,
                 "fields_json": None, "record_raw_json": None}
        if data is None:
            out.append({**blank, "locator": "file", "obs_id": obs_id(rel, "file", "missing"),
                        "state": "declared_missing", "reason": "declared_missing"})
            continue
        if kind is None:
            out.append({**blank, "locator": "file", "obs_id": obs_id(rel, "file", sha), "state": "excluded",
                        "reason": exclusion_reason(rel)})
            continue
        result = parsed[rel]
        out += [{**blank, "locator": q["locator"], "obs_id": obs_id(rel, q["locator"], sha), "state": "quarantined",
                 "reason": q["reason"], "record_raw_json": q.get("record_raw_json")} for q in result.quarantined]
        out += [{**blank, "locator": r["locator"], "obs_id": r["obs_id"], "state": "emitted", "reason": None,
                 "fields_json": canonical_json({k: v for k, v in r.items() if k not in _OWN_COLUMNS}),
                 "record_raw_json": r.get("record_raw_json")} for r in result.rows]
        if not result.rows and not result.quarantined:
            out.append({**blank, "locator": "file", "obs_id": obs_id(rel, "file", sha), "state": "excluded",
                        "reason": "no_records"})
    return out


def assign_dispositions(accounting: list[dict], dispositions: dict[str, str]) -> None:
    for a in accounting:
        if a["state"] == "emitted":
            a["disposition"] = dispositions.get(a["obs_id"]) or KIND_DISPOSITION.get(a["kind"])


def _doc(data: bytes):
    try:
        return load_json_bytes(data)
    except ValueError:
        return None


def _jsonl(data: bytes) -> list[tuple[int, object]] | None:
    """(line number, parsed value or None) for every non-blank line; None when the bytes are not UTF-8."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                out.append((n, json.loads(line)))
            except json.JSONDecodeError:
                out.append((n, None))
    return out


def required_locators(kind: str, data: bytes) -> set[str]:
    """An independent structural census of the records one file holds, walked from its containers without type
    checks (spec §8; Codex plan r3 #1). A file that is itself one record (a decision, a scheduler state, a pick
    file without slots), or whose container cannot be read, holds exactly one record: `file`. Only a readable
    container with no entries holds none — the one case `no_records` may describe."""
    if kind in _PICK_LIKE:
        doc = _doc(data)
        slots = {f"slot={slot}" for slot, key in (("primary", "pick"), ("double_down", "double_down"))
                 if isinstance(doc, dict) and doc.get(key) is not None}
        return slots or {"file"}
    required: set[str] = set()
    if kind in ("lineup_evolution", "saver_transitions", "contest_ledger"):
        lines = _jsonl(data)
        if lines is None:
            return {"file"}
        for n, doc in lines:
            if kind == "lineup_evolution":
                slots = {f"line={n}/slot={s}" for s in ("primary", "double_down")
                         if isinstance(doc, dict) and doc.get(s) is not None}
                required |= slots or {f"line={n}"}
                continue
            required.add(f"line={n}")
            if kind == "contest_ledger" and isinstance(doc, dict) and isinstance(doc.get("predictions"), list):
                for i, rnd in enumerate(doc["predictions"]):
                    required.add(f"line={n}/round={i}")
                    slots = rnd.get("roundPredictions") if isinstance(rnd, dict) else None
                    if isinstance(slots, list):
                        required |= {f"line={n}/round={i}/slot={j}" for j in range(len(slots))}
        return required
    if kind in _ITEM_KEYS:
        doc = _doc(data)
        items = doc.get(_ITEM_KEYS[kind]) if isinstance(doc, dict) else None
        return {f"item={i}" for i in range(len(items))} if isinstance(items, list) else {"file"}
    if kind == "schedule":
        doc = _doc(data)
        dates = doc.get("dates") if isinstance(doc, dict) else None
        if not isinstance(dates, list):
            return {"file"}
        for i, day in enumerate(dates):
            games = day.get("games") if isinstance(day, dict) else None
            required |= ({f"date={i}/game={j}" for j in range(len(games))} if isinstance(games, list) and games
                         else {f"date={i}"})
        return required
    return {"file"}


def _ancestors(locator: str) -> list[str]:
    if locator == "file":
        return []
    parts = locator.split("/")
    return ["file"] + ["/".join(parts[:k]) for k in range(1, len(parts))]


_OVER_RECORDS = {"excluded": "exclusion_over_records", "declared_missing": "declared_missing_over_records"}


def census_problems(files: dict, routed: dict[str, str], accounting: list[dict]) -> list[tuple[str, str, str]]:
    """(problem, source_path, locator) for every breach of exactly-once accounting (Codex plan r2 #1, r3 #1), over
    EVERY bundle path. A declared-missing input, an excluded path and a readable empty container each have exactly
    one `file` row in that state, with its reason. Every other file accounts for each census record exactly once —
    emitted or quarantined at its locator, or under one quarantined ancestor — with no row that is not a record or
    an ancestor, no row in any other state, and no emitted row that is not a record."""
    by_file: dict[str, list[dict]] = {}
    for a in accounting:
        by_file.setdefault(a["source_path"], []).append(a)
    problems = [("source_not_in_bundle", rel, "file") for rel in sorted(set(by_file) - set(files))]
    for rel in sorted(files):
        rows, data, kind = by_file.get(rel, []), files[rel], routed.get(rel)
        required = set() if data is None or kind is None else required_locators(kind, data)
        if not required:
            want = (("declared_missing", "declared_missing") if data is None else
                    ("excluded", exclusion_reason(rel)) if kind is None else ("excluded", "no_records"))
            if [(a["locator"], a["state"], a["reason"]) for a in rows] != [("file", *want)]:
                problems.append(("file_accounting_shape", rel, "file"))
            continue
        allowed = required | {anc for loc in required for anc in _ancestors(loc)}
        accounted = {a["locator"]: a for a in rows}
        for loc, a in sorted(accounted.items()):
            if loc not in allowed:
                problems.append(("phantom_locator", rel, loc))
            elif a["state"] not in ("emitted", "quarantined"):
                problems.append((_OVER_RECORDS.get(a["state"], "unknown_state"), rel, loc))
            elif a["state"] == "emitted" and loc not in required:
                problems.append(("emitted_off_record", rel, loc))
        for loc in sorted(required):
            covers = (loc in accounted) + sum(accounted.get(anc, {}).get("state") == "quarantined"
                                              for anc in _ancestors(loc))
            if covers != 1:
                problems.append(("omitted" if covers == 0 else "covered_twice", rel, loc))
    return problems


class InvariantError(Exception):
    pass


def check_sources(files: dict, routed: dict[str, str], accounting: list[dict]) -> None:
    """Source-phase build failures, run before any canonical row exists: an unaccounted file, a duplicated
    occurrence, any census problem, or an occurrence whose identity does not match its source — its obs_id
    recomputed from (path, locator, content hash), its content hash or its kind (Codex plan r3 #1). Recomputed ids
    are also unique, because (path, locator) pairs are."""
    missing = sorted(set(files) - {a["source_path"] for a in accounting})
    if missing:
        raise InvariantError(f"files not accounted: {missing[:5]}")
    dupes = [k for k, n in Counter((a["source_path"], a["locator"]) for a in accounting).items() if n > 1]
    if dupes:
        raise InvariantError(f"duplicate occurrence rows: {dupes[:5]}")
    problems = census_problems(files, routed, accounting)
    if problems:
        raise InvariantError(f"census: {problems[:5]}")
    shas = {rel: None if data is None else sha256_hex(data) for rel, data in files.items()}   # once per file
    for a in accounting:
        sha = shas[a["source_path"]]
        want = (obs_id(a["source_path"], a["locator"], "missing" if sha is None else sha), sha, routed.get(a["source_path"]))
        if (a["obs_id"], a["content_sha256"], a["kind"]) != want:
            raise InvariantError(f"occurrence identity does not match its source: {a['source_path']} {a['locator']}")


_REF_KINDS = {"pick_obs_id": "pick_file", "pick_view_obs_id": "pick_file", "decision_obs_id": "decision",
              "state_obs_id": "scheduler_state", "contest_obs_id": "contest_ledger"}


def check_invariants(files: dict, accounting: list[dict], matches: list[dict], ledger_rows: list[dict],
                     membership: list[dict], summary: list[dict], season_dates: list[str]) -> None:
    """Output-phase build failures (spec §8): an undisposed or unknown disposition; a ledger reference to an
    occurrence that was not emitted or is of the wrong kind; a canonical disposition no ledger row references; a
    reference to another date's occurrence, a pick of another slot or selection, or a usable decision whose named
    selections are not exactly its ledger rows (Codex code r1 #1); a recipe summary that is not every frozen rule
    exactly as frozen (Codex code r1 #2); recipe membership that does not list every rule × universe slot exactly
    once, does not add up to the reported totals, or links a row to anything but the occurrence that accounts for its
    record (Codex plan r3 #3); a contest slot identity twice; a selection linked to two contest slots; a qualified
    contest slot missing from the ledger, placed twice, or carrying evidence or a grade that is not its own
    occurrence's (Codex code r1 #1); duplicate row ids; a malformed season day."""
    undisposed = [(a["source_path"], a["locator"]) for a in accounting
                  if a["state"] == "emitted" and not a["disposition"]]
    if undisposed:
        raise InvariantError(f"emitted occurrences without a disposition: {undisposed[:5]}")
    unknown = sorted({a["disposition"] for a in accounting if a["disposition"]} - DISPOSITIONS)
    if unknown:
        raise InvariantError(f"unknown disposition values: {unknown}")
    emitted = {a["obs_id"]: a for a in accounting if a["state"] == "emitted"}
    referenced = Counter()
    for r in ledger_rows:
        for column, kind in _REF_KINDS.items():
            value = r.get(column)
            if value is None:
                continue
            if value not in emitted or emitted[value]["kind"] != kind:
                raise InvariantError(f"ledger row {r['row_id']} {column} is not an emitted {kind} occurrence")
            referenced[(column, value)] += 1
    for a in emitted.values():
        if a["disposition"] == "canonical_selection" and referenced[("pick_obs_id", a["obs_id"])] != 1:
            raise InvariantError(f"canonical selection {a['obs_id']} is not referenced by exactly one ledger row")
        if a["disposition"] == "canonical_decision" and not referenced[("decision_obs_id", a["obs_id"])]:
            raise InvariantError(f"canonical decision {a['obs_id']} is referenced by no ledger row")
    # References name the right occurrence (Codex code r1 #1): a decision, state or pick of the row's own date — a pick
    # also of its slot, and an attached pick of its selection — and a usable decision yields exactly what it names.
    facts = {obs: json.loads(a["fields_json"]) for obs, a in emitted.items()
             if a["kind"] in ("decision", "scheduler_state", "pick_file")}
    named_rows: dict[str, list[tuple]] = {}
    for r in ledger_rows:
        for column in ("decision_obs_id", "state_obs_id", "pick_obs_id", "pick_view_obs_id"):
            if r.get(column) is not None and facts[r[column]]["file_date"] != r["date"]:
                raise InvariantError(f"ledger row {r['row_id']} {column} names another date's occurrence")
        for column in ("pick_obs_id", "pick_view_obs_id"):
            if r.get(column) is not None and facts[r[column]]["slot"] != r["slot"]:
                raise InvariantError(f"ledger row {r['row_id']} {column} names another slot's pick")
        pick = facts.get(r.get("pick_obs_id"))
        if pick is not None and (pick["batter_id"], pick["game_pk"]) != (r["batter_id"], r["game_pk"]):
            raise InvariantError(f"ledger row {r['row_id']} attaches a pick of another selection")
        if r["row_kind"] == "selection" and r.get("decision_obs_id") is not None:
            named_rows.setdefault(r["decision_obs_id"], []).append((r["slot"], r["batter_id"], r["game_pk"]))
    # The obligations come from the source date, never from an assigned disposition (Codex code r2 #1): a decision or
    # pick is outside the season window exactly when its source date is, and every in-season decision yields exactly
    # what it names (a skip, exactly one day row).
    in_season = set(season_dates)
    for obs, a in emitted.items():
        if a["kind"] in ("decision", "pick_file") and (
                (a["disposition"] == "outside_season_window") != (facts[obs]["file_date"] not in in_season)):
            raise InvariantError(f"{a['kind']} {obs} is marked {a['disposition']} for source date "
                                 f"{facts[obs]['file_date']}")
        if a["kind"] != "decision" or facts[obs]["file_date"] not in in_season:
            continue
        d = facts[obs]
        slots = {"single": ("primary",), "double": ("primary", "double_down")}.get(d["action"], ())
        want = sorted((s, d[f"{s}_batter_id"], d[f"{s}_game_pk"]) for s in slots)
        if sorted(named_rows.get(obs, [])) != want or (not slots and referenced[("decision_obs_id", obs)] != 1):
            raise InvariantError(f"decision {obs} does not yield exactly its named selections")
    # Every frozen rule, exactly as frozen (Codex code r1 #2): the expected rules come from RULES, never from the
    # summary the evaluation produced.
    if sorted(s["rule_id"] for s in summary) != sorted(RULES):
        raise InvariantError("the recipe summary does not hold every frozen rule exactly once")
    for s in summary:
        rule = RULES[s["rule_id"]]
        if (s["recipe"], s["files"], s["primary_grading"], s["leg_grading"], list(s["window"]),
                [s["published_primaries"], s["published_legs"]]) != (rule["recipe"], rule["files"], rule["primary"],
                                                                    rule["legs"], list(rule["window"]),
                                                                    list(rule["published"])):
            raise InvariantError(f"recipe {s['rule_id']} does not carry its frozen definition")
        fit = {(True, True): "both", (True, False): "primaries_only", (False, True): "legs_only",   # Codex code r2 #2
               (False, False): "none"}[(s["primaries"] == s["published_primaries"], s["legs"] == s["published_legs"])]
        label = ("matches published totals; historical membership unverified" if fit == "both"
                 else f"does not reproduce both totals ({fit})")
        if (s["fit"], s["label"]) != (fit, label):
            raise InvariantError(f"recipe {s['rule_id']} reports a fit its totals do not support")
    keys = Counter((m["rule_id"], m["source_path"], m["slot"]) for m in membership)
    if set(keys) != {(rule_id, rel, slot) for rule_id in RULES for rel, slot in membership_slots(files)} \
            or any(n > 1 for n in keys.values()):
        raise InvariantError("recipe membership does not list every rule and universe slot exactly once")
    for s in summary:
        included = [m["slot"] for m in membership if m["rule_id"] == s["rule_id"] and m["included"]]
        if (included.count("primary"), included.count("double_down")) != (s["primaries"], s["legs"]):
            raise InvariantError(f"recipe {s['rule_id']} membership does not add up to its totals")
    by_locator = {(a["source_path"], a["locator"]): a for a in accounting}
    canonical = {r["pick_obs_id"]: r for r in ledger_rows if r.get("pick_obs_id")}
    for m in membership:
        source = by_locator.get((m["source_path"], f"slot={m['slot']}")) or by_locator.get((m["source_path"], "file"))
        link = canonical.get(source["obs_id"]) if source else None
        got = (m["source_obs_id"], m["source_state"], m["source_reason"], m["occurrence_disposition"],
               m["selection_id"], m["canonical_bts_outcome"])
        if source is None or got != (source["obs_id"], source["state"], source["reason"], source["disposition"],
                                     link["selection_id"] if link else None, link.get("bts_outcome") if link else None):
            raise InvariantError(f"recipe row {m['recipe_slot_key']} is not linked to the occurrence that accounts for it")
    if any(n > 1 for n in Counter((m["round_id"], m["unit_id"], m["player_id"]) for m in matches).values()):
        raise InvariantError("a contest slot identity appears twice")
    if any(n > 1 for n in Counter(m["selection_id"] for m in matches if m["selection_id"]).values()):
        raise InvariantError("a selection is linked to more than one contest slot")
    # Every qualified contest slot reaches the ledger exactly once — on its linked selection or as a contest-only row —
    # with its own occurrence, identity, grade and round label (Codex code r1 #1).
    by_identity = {(m["round_id"], m["unit_id"], m["player_id"]): m for m in matches}
    census: dict[tuple, list[tuple]] = {}      # identity → [((recorded_at, line_no), obs_id, facts)], from the source
    for obs, a in emitted.items():
        if a["kind"] == "contest_ledger" and (f := json.loads(a["fields_json"])).get("row_level") == "slot":
            census.setdefault((f["round_id"], f["unit_id"], f["player_id"]), []).append(
                ((f["recorded_at"], f["line_no"]), obs, f))
    if set(census) != set(by_identity):
        raise InvariantError("the contest slot matches are not exactly the qualified slot identities")
    for identity, seen in census.items():     # Codex code r2 #3: each match carries its latest observation
        latest = max(key for key, _obs, _f in seen)
        last = {obs: f for key, obs, f in seen if key == latest}
        m = by_identity[identity]
        if (m["last_obs_id"] not in last or (m["first_seen"], m["last_seen"], m["last_line_no"], m["n_observations"])
                != (min(key for key, _o, _f in seen)[0], latest[0], latest[1], len(seen))
                or any(m[k] != last[m["last_obs_id"]][k] for k in ("slot_result", "slot_result_state", "round_result"))):
            raise InvariantError(f"contest slot {identity} does not carry its latest observation")
    placed = Counter()
    for r in ledger_rows:
        if r.get("contest_obs_id") is None:
            if r["row_kind"] == "contest_only":
                raise InvariantError(f"contest-only row {r['row_id']} carries no contest occurrence")
            continue
        identity = (r["round_id"], r["unit_id"], r["player_id"])
        m, occ = by_identity.get(identity), json.loads(emitted[r["contest_obs_id"]]["fields_json"])
        if (m is None or r["contest_obs_id"] != m["last_obs_id"]
                or (occ.get("row_level"), occ["round_id"], occ["unit_id"], occ["player_id"]) != ("slot", *identity)
                or (r["row_kind"] == "selection") != bool(m["selection_id"])
                or (r["row_kind"] == "selection" and r["selection_id"] != m["selection_id"])):
            raise InvariantError(f"ledger row {r['row_id']} carries contest evidence that is not its own slot's")
        if (r["bts_outcome"] not in (None, occ["slot_result"]) or r.get("contest_round_result") != occ["round_result"]
                or (r.get("bts_outcome_status") == "graded" and r["bts_outcome"] != occ["slot_result"])):
            raise InvariantError(f"ledger row {r['row_id']} carries a grade that is not its slot's")
        placed[identity] += 1
    if set(placed) != set(by_identity) or any(n != 1 for n in placed.values()):
        raise InvariantError("a qualified contest slot is missing from the ledger or placed twice")
    if any(n > 1 for n in Counter(r["row_id"] for r in ledger_rows).values()):
        raise InvariantError("duplicate ledger row ids")
    by_date: dict[str, list[dict]] = {}
    for r in ledger_rows:
        if r["row_kind"] != "contest_only":
            by_date.setdefault(r["date"], []).append(r)
    for d in season_dates:
        rows = by_date.get(d, [])
        kinds = {r["row_kind"] for r in rows}
        if not rows:
            raise InvariantError(f"no ledger row for {d}")
        if "selection" in kinds and len(kinds) > 1:
            raise InvariantError(f"{d} mixes a day row with selections")
        if "selection" not in kinds and len(rows) != 1:
            raise InvariantError(f"{d} has {len(rows)} day rows")
        slots = [r["slot"] for r in rows if r["row_kind"] == "selection"]
        if len(slots) > 2 or len(slots) != len(set(slots)):
            raise InvariantError(f"{d} has malformed selection slots {slots}")


# ---- recipe rules: fixed before any real count (plan Task 9 text; pinned by rules_fingerprint) -----------
GRADED = frozenset({"hit", "miss"})
FILE_SETS = {"F1": re.compile(rf"^picks/{_D}\.json$"),
             "F2": re.compile(r"^picks/2026-[^/]*\.json$"),
             "F3": re.compile(r"^picks/.*\.json$")}
RECIPE_KINDS = {"scorecard_0911": "prose_rerun", "tally_0914": "candidate_search"}
_SCORECARD = {"recipe": "scorecard_0911", "published": (141, 82), "recipe_date": "2026-09-11",
              "window": ("2026-03-29", "2026-09-10")}
_TALLY = {"recipe": "tally_0914", "published": (191, 157), "recipe_date": "2026-09-14"}
RULES: dict[str, dict] = {}
for _i, (_f, _p, _l) in enumerate(itertools.product(("F1", "F2"), ("G1", "G2"), ("G1", "G2")), start=1):
    RULES[f"S{_i}"] = {**_SCORECARD, "files": _f, "primary": _p, "legs": _l}
for _i, (_f, _g, _end) in enumerate(itertools.product(("F1", "F2", "F3"), ("G1", "G2", "G3", "G4"),
                                                      ("2026-09-13", "2026-09-14")), start=1):
    RULES[f"T{_i}"] = {**_TALLY, "files": _f, "primary": _g, "legs": _g, "window": ("0000-00-00", _end)}
_ET = ZoneInfo("America/New_York")
_FILE_DATE = re.compile(_D)


def slot_value(doc: dict, slot: str, grading: str):
    """The value a grading reads for one slot of a pick-object record (the recipe's own reading)."""
    if slot == "double_down" and doc.get("double_down") is None:
        return None
    if grading == "G2":
        sr = doc.get("slot_results")
        if isinstance(sr, dict):
            return sr.get("pick" if slot == "primary" else "double_down")
        return doc.get("result") if (slot == "primary" and doc.get("double_down") is None) else None
    return doc.get("result")     # G1, G3 and G4 read the day result


def counts(value, grading: str) -> bool:
    """Whether a grading counts a recipe value. G1/G2 count only the string labels in GRADED; G3 counts any
    non-null value and G4 every slot, as those naive rules would — a malformed value never crashes a recipe."""
    if grading == "G4":
        return True
    if grading == "G3":
        return value is not None
    return isinstance(value, str) and value in GRADED


def slot_inclusion(doc: dict, primary_grading: str, leg_grading: str) -> dict[str, tuple]:
    out = {}
    for slot, grading in (("primary", primary_grading), ("double_down", leg_grading)):
        if slot == "double_down" and doc.get("double_down") is None:
            continue
        value = slot_value(doc, slot, grading)
        out[slot] = (value, counts(value, grading))
    return out


def _mtime_after(mtime_utc: str | None, recipe_date: str) -> bool | None:
    """Interpretation I9: the filesystem mtime (suggestive, never proof) against the recipe's ET date — True
    after, False before, None on the same ET day (order unknown) or without an mtime."""
    if mtime_utc is None:
        return None
    day = datetime.fromisoformat(mtime_utc).astimezone(_ET).date().isoformat()
    return None if day == recipe_date else day > recipe_date


def _natural(rule_id: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", rule_id)]


def _universe(files: dict[str, bytes | None]) -> dict[str, tuple[dict, str]]:
    """Every pick-object record under picks/ (any suffix, AppleDouble aside) with its window date."""
    universe: dict[str, tuple[dict, str]] = {}
    for rel, data in files.items():
        if data is None or not rel.startswith("picks/") or _APPLEDOUBLE.search(rel):
            continue
        doc = _doc(data)
        if isinstance(doc, dict) and isinstance(doc.get("pick"), dict):
            m = _FILE_DATE.search(rel.rsplit("/", 1)[-1])
            universe[rel] = (doc, m.group(0) if m else str(doc.get("date") or ""))
    return universe


def membership_slots(files: dict[str, bytes | None]) -> set[tuple[str, str]]:
    """(source_path, slot) for every slot of every universe record — what each rule must list exactly once."""
    return {(rel, slot) for rel, (doc, _day) in _universe(files).items()
            for slot in ("primary", "double_down") if slot == "primary" or doc.get("double_down") is not None}


def evaluate_rules(files: dict[str, bytes | None], rules: dict, mtimes: dict[str, str | None], *,
                   frozen_at: str) -> tuple[list[dict], list[dict]]:
    """Spec §8 table 3. Every rule gets a row per universe slot, included or excluded with a reason, so no candidate
    record disappears. `recipe_slot_key` names the recipe's record; the compiler links each row to the source
    occurrence that accounts for it (Codex plan r2 #5)."""
    universe = _universe(files)
    summary, membership = [], []
    for rule_id in sorted(rules, key=_natural):
        rule = rules[rule_id]
        n = {"primary": 0, "double_down": 0}
        lo, hi = rule["window"]
        for rel in sorted(universe):
            doc, day = universe[rel]
            in_set = FILE_SETS[rule["files"]].match(rel) is not None
            in_window = lo <= day <= hi
            for slot, (value, counted) in slot_inclusion(doc, rule["primary"], rule["legs"]).items():
                reason = (None if (in_set and in_window and counted) else "not_in_file_set" if not in_set
                          else "outside_window" if not in_window else "value_not_counted")
                n[slot] += reason is None
                membership.append({"rule_id": rule_id, "recipe": rule["recipe"], "source_path": rel, "slot": slot,
                                   "recipe_slot_key": f"{rel}#{slot}", "file_date": day,
                                   "recipe_value": None if value is None else str(value), "included": reason is None,
                                   "exclusion_reason": reason, "historical_membership": "unknown",
                                   "source_mtime_utc": mtimes.get(rel),
                                   "mtime_after_recipe_date": _mtime_after(mtimes.get(rel), rule["recipe_date"]),
                                   "frozen_at_utc": frozen_at})
        pub_p, pub_l = rule["published"]
        fit = {(True, True): "both", (True, False): "primaries_only", (False, True): "legs_only",
               (False, False): "none"}[(n["primary"] == pub_p, n["double_down"] == pub_l)]
        summary.append({"rule_id": rule_id, "recipe": rule["recipe"], "files": rule["files"],
                        "primary_grading": rule["primary"], "leg_grading": rule["legs"],
                        "window": list(rule["window"]), "primaries": n["primary"], "legs": n["double_down"],
                        "published_primaries": pub_p, "published_legs": pub_l, "fit": fit,
                        "label": ("matches published totals; historical membership unverified" if fit == "both"
                                  else f"does not reproduce both totals ({fit})")})
    return summary, membership


def recipe_labels(summary: list[dict]) -> dict[str, str]:
    """Spec §8 and Interpretation I9. The 9/11 scorecard rerun is a hypothesis whether or not it reproduces
    its totals; the 9/14 tally is a hypothesis if some pre-declared rule reproduces both totals, else
    unrecoverable. A single matching total pins nothing. `exact` and `partial` need independent per-record
    evidence, which Phase 1 does not have, so they are never emitted."""
    out = {}
    for recipe in sorted({s["recipe"] for s in summary}):
        if RECIPE_KINDS.get(recipe) == "prose_rerun":
            out[recipe] = "hypothesis"
        else:
            out[recipe] = "hypothesis" if any(s["fit"] == "both" for s in summary if s["recipe"] == recipe) \
                else "unrecoverable"
    return out


_RECIPE_ROOTS = ("evaluate_rules", "recipe_labels")
_PACKAGE = __name__.rsplit(".", 1)[0]


def _bound(value):
    """A JSON description of one configuration value the recipe code reads; any other kind fails closed."""
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": int(value.flags)}
    if isinstance(value, ZoneInfo):
        return {"zone": value.key}
    if isinstance(value, dict):
        return {"dict": {canonical_json(_bound(k)): _bound(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"seq": [_bound(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"set": sorted(canonical_json(_bound(v)) for v in value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"rules_fingerprint cannot bind {type(value).__name__}")


def _global_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _global_names(const)
    return names


def recipe_closure() -> dict[str, object]:
    """Everything the recipe evaluation reads (Codex plan r2 #6, r3 #4), by qualified name: the source of every
    repo-local function reachable from `evaluate_rules` and `recipe_labels` (imported ones included), and every
    configuration value those functions read — compiled regexes with their flags, the time zone, tables. Modules
    and outside callables are bound by name; their behaviour is pinned by the versions `build.json` records.
    Anything else fails closed."""
    bound: dict[str, object] = {}
    todo = [globals()[name] for name in _RECIPE_ROOTS]
    while todo:
        fn = todo.pop()
        key = f"{fn.__module__}.{fn.__qualname__}"
        if key in bound:
            continue
        bound[key] = inspect.getsource(fn)
        for name in sorted(_global_names(fn.__code__)):
            if name not in fn.__globals__ or (name.startswith("__") and name.endswith("__")):
                continue          # an attribute, a local, a builtin, or interpreter bookkeeping (e.g. __file__)
            value = fn.__globals__[name]
            module = getattr(value, "__module__", None) or ""
            if isinstance(value, types.FunctionType) and module.startswith(_PACKAGE):
                todo.append(value)
            elif isinstance(value, types.ModuleType):
                bound[f"module:{value.__name__}"] = value.__name__
            elif callable(value) and not module.startswith(_PACKAGE):
                bound[f"callable:{module}.{getattr(value, '__qualname__', name)}"] = name
            else:
                bound[f"{fn.__module__}.{name}"] = _bound(value)
    return bound


def rules_fingerprint() -> str:
    """One digest over the rule table and everything the recipe evaluation reads (`recipe_closure`): any edit that
    can move a recipe total or label — a predicate, the JSON decoder, a regex or its flags, a table, even a
    comment in a reachable function — changes the digest predeclared in the exposure register."""
    return sha256_hex(json.dumps({"rules": _bound(RULES), "closure": recipe_closure()}, sort_keys=True).encode())
