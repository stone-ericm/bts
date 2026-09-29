import json

import pytest

from scripts.audit.season_ledger.ids import Parsed
from scripts.audit.season_ledger.reconcile import (PARSERS, RULES, InvariantError, account, assign_dispositions,
                                                   census_problems, check_invariants, check_sources, evaluate_rules,
                                                   exclusion_reason, recipe_closure, recipe_labels, route,
                                                   rules_fingerprint, slot_inclusion)
from tests.scripts.season_ledger.builders import cand, contest_line, decision_json, dumps, gz, pick_json, rnd, slot

# The recipe candidates frozen by this plan (Task 9 text). Recompute only by editing the plan before the real run.
RULES_SHA256 = "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f"
# Everything the recipe evaluation reads (reconcile.recipe_closure): a new dependency must show up here.
RECIPE_CLOSURE = [
    "callable:datetime.datetime",
    "module:gzip",
    "module:json",
    "module:re",
    "module:zlib",
    "scripts.audit.season_ledger.ids.load_json_bytes",
    "scripts.audit.season_ledger.reconcile.FILE_SETS",
    "scripts.audit.season_ledger.reconcile.GRADED",
    "scripts.audit.season_ledger.reconcile.RECIPE_KINDS",
    "scripts.audit.season_ledger.reconcile._APPLEDOUBLE",
    "scripts.audit.season_ledger.reconcile._ET",
    "scripts.audit.season_ledger.reconcile._FILE_DATE",
    "scripts.audit.season_ledger.reconcile._doc",
    "scripts.audit.season_ledger.reconcile._mtime_after",
    "scripts.audit.season_ledger.reconcile._natural",
    "scripts.audit.season_ledger.reconcile._universe",
    "scripts.audit.season_ledger.reconcile.counts",
    "scripts.audit.season_ledger.reconcile.evaluate_rules",
    "scripts.audit.season_ledger.reconcile.recipe_labels",
    "scripts.audit.season_ledger.reconcile.slot_inclusion",
    "scripts.audit.season_ledger.reconcile.slot_value",
]


@pytest.mark.parametrize("path,kind", [
    ("picks/2026-05-01.json", "pick_file"), ("picks/2026-05-01/decision.json", "decision"),
    ("picks/2026-05-01/scheduler_state.json", "scheduler_state"),
    ("picks/2026-08-30/deferred_fallback_20260830T120000-0400.json", "archive"),
    ("picks/archive/2026-04-11.json.postponed", "manual_archive"),
    ("picks/archive_actual_streak_repair_20260527T101500Z/2026-05-24.json", "repair_archive"),
    ("picks/archive_actual_streak_repair_20260527T101500Z_missed_2/2026-05-24.json.before", "repair_archive"),
    ("picks/lineup_evolution_2026-05-01.jsonl", "lineup_evolution"),
    ("picks/account_state/contest_ledger.jsonl", "contest_ledger"),
    ("picks/account_state/saver_transitions.jsonl", "saver_transitions"),
    ("static/rounds/20260704T030011Z.json", "rounds"), ("static/units/20260801T150000Z.json.gz", "units"),
    ("static/grab_20260927/002_players.json.gz", "players"), ("schedules/2026-05-01.json", "schedule")])
def test_routes(path, kind):
    assert route(path) == kind


@pytest.mark.parametrize("path,reason", [
    ("picks/._2026-05-01.json", "appledouble_resource_fork"), ("static/rounds/._x.json", "appledouble_resource_fork"),
    ("picks/2026-05-01.shadow.json", "shadow_model_out_of_scope"),
    ("picks/backup_shadow_2026-05-09/2026-05-08.shadow.json", "shadow_model_out_of_scope"),
    ("picks/2026-05-01.policy_shadow.json", "skip_policy_shadow_out_of_scope"),
    ("picks/slates/2026-05-01.json", "slate_binding_out_of_scope"),
    ("picks/streak.json", "state_snapshot_not_a_record"),
    ("picks/archive_actual_streak_repair_20260527T101500Z/streak.before.json", "state_snapshot_not_a_record"),
    ("picks/account_state/contest_streak.manual.json.archived_post_auto_20260601T000000Z", "state_snapshot_not_a_record"),
    ("picks/archive_replay_restore_20260601T000000Z_post_contest_state_deploy/README.txt", "documentation"),
    ("picks/.nrestarts_checkpoint", "runtime_marker"), ("logs/cron.log", "corroboration_only"),
    ("static/grab_20260927/004_squads.json.gz", "not_used_phase1"), ("picks/mystery.bin", "unrecognized_path")])
def test_exclusions(path, reason):
    assert route(path) is None and exclusion_reason(path) == reason


def _parse(files, routed):
    return {rel: PARSERS[kind](rel, files[rel]) for rel, kind in routed.items()}


def test_every_file_is_accounted_with_its_locator_fields_and_disposition():
    files = {"picks/2026-05-01.json": pick_json("2026-05-01"), "picks/._2026-05-01.json": b"\x00\x05",
             "picks/2026-05-02.json": b"{", "schedules/2026-05-02.json": None}
    routed = {"picks/2026-05-01.json": "pick_file", "picks/2026-05-02.json": "pick_file"}
    parsed = _parse(files, routed)
    obs = parsed["picks/2026-05-01.json"].rows[0]["obs_id"]
    acc = account(files, routed, parsed)
    assign_dispositions(acc, {obs: "canonical_selection"})
    assert {(a["source_path"], a["locator"], a["state"], a["reason"] or a["disposition"]) for a in acc} == {
        ("picks/2026-05-01.json", "slot=primary", "emitted", "canonical_selection"),
        ("picks/._2026-05-01.json", "file", "excluded", "appledouble_resource_fork"),
        ("picks/2026-05-02.json", "file", "quarantined", "invalid_json:JSONDecodeError"),
        ("schedules/2026-05-02.json", "file", "declared_missing", "declared_missing")}
    (emitted,) = [a for a in acc if a["state"] == "emitted"]
    assert emitted["obs_id"] == obs and json.loads(emitted["fields_json"])["batter_id"] == 101
    assert json.loads(emitted["record_raw_json"])["pick"]["batter_id"] == 101
    check_sources(files, routed, acc)


def test_census_requires_every_record_exactly_once():
    # Codex plan r1 #1 and r2 #1: omitted, phantom, empty-parser, dropped line record, foreign path, double cover.
    pick_rel, contest_rel = "picks/2026-05-02.json", "picks/account_state/contest_ledger.jsonl"
    line = contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")]), rnd(972, "void", 0, 0, [])])
    files = {pick_rel: pick_json("2026-05-02", dd={}), contest_rel: (line + "\n").encode()}
    routed = {pick_rel: "pick_file", contest_rel: "contest_ledger"}
    good = account(files, routed, _parse(files, routed))
    assert census_problems(files, routed, good) == []

    def problems(acc):
        return [(p, loc) for p, _rel, loc in census_problems(files, routed, acc)]

    assert problems([a for a in good if a["locator"] != "slot=double_down"]) == [("omitted", "slot=double_down")]
    assert problems(good + [dict(good[0], locator="slot=triple")]) == [("phantom_locator", "slot=triple")]
    empty = account(files, routed, {pick_rel: Parsed(), contest_rel: _parse(files, routed)[contest_rel]})
    assert problems(empty) == [("exclusion_over_records", "file"), ("omitted", "slot=double_down"),
                               ("omitted", "slot=primary")]
    assert problems([a for a in good if a["locator"] != "line=1"]) == [("omitted", "line=1")]
    assert census_problems(files, routed, good + [dict(good[0], source_path="picks/phantom.json")]) == [
        ("source_not_in_bundle", "picks/phantom.json", "file")]
    overlap = good + [dict(good[-1], locator="file", state="excluded", reason="no_records")]
    assert ("exclusion_over_records", "file") in problems(overlap)


def test_a_quarantined_ancestor_covers_its_records_and_double_cover_fails():
    rel = "picks/account_state/contest_ledger.jsonl"
    good = contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])])
    bad = contest_line("2026-08-22T14:30:00Z", [rnd(973, "hit", 1, 1, [slot(1930, None, "hit")])])
    files, routed = {rel: (good + "\n" + bad + "\n").encode()}, {rel: "contest_ledger"}
    acc = account(files, routed, _parse(files, routed))
    assert census_problems(files, routed, acc) == []          # line 2's round and slot sit under its quarantine
    leaked = acc + [dict(acc[0], locator="line=2/round=0/slot=0", state="emitted")]
    assert [p for p, _r, loc in census_problems(files, routed, leaked)] == ["covered_twice"]


def _acc(*pairs, kind="pick_file", disposition="not_selected", fields=None):
    """Synthetic accounting rows; like compiled ones, their facts carry the source date (the compiler adds file_date)."""
    return [{"source_path": p, "locator": loc, "obs_id": f"o:{p}:{loc}", "state": "emitted", "kind": kind,
             "disposition": disposition, "fields_json": json.dumps(fields or {"file_date": "2026-05-01"})}
            for p, loc in pairs]


# The recipe summary over an empty universe: every frozen rule, zero counts (the output checks require all of them).
EMPTY_SUMMARY = evaluate_rules({}, RULES, {}, frozen_at="t")[0]


def _day(date, kind="unobserved_day", **refs):
    return {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "slot": None, **refs}


def test_source_checks_catch_unaccounted_and_duplicated_files():
    files = {"a": b"1", "b": b"2"}
    acc = account(files, {}, {})
    check_sources(files, {}, acc)
    with pytest.raises(InvariantError, match="not accounted"):
        check_sources(files, {}, acc[:1])
    with pytest.raises(InvariantError, match="duplicate occurrence"):
        check_sources(files, {}, acc + acc[:1])
    rel = "picks/2026-05-02.json"
    pf, routed = {rel: pick_json("2026-05-02", dd={})}, {rel: "pick_file"}
    full = account(pf, routed, _parse(pf, routed))
    with pytest.raises(InvariantError, match="census: \\[\\('omitted'"):
        check_sources(pf, routed, [a for a in full if a["locator"] != "slot=double_down"])


def test_source_checks_reject_the_codex_r3_census_probes():
    # Codex plan r3 #1: an empty single-record parse, a record relabelled missing, a row beneath an excluded path and
    # a forged occurrence id must each fail before any ledger row exists.
    dec_rel, contest_rel = "picks/2026-08-10/decision.json", "picks/account_state/contest_ledger.jsonl"
    shadow_rel, manual_rel = "picks/2026-08-10.shadow.json", "picks/archive/2026-08-10.json.postponed"
    files = {dec_rel: decision_json("2026-08-10", action="single", primary=cand(101, 5001)),
             contest_rel: (contest_line("2026-08-21T14:30:00Z", [rnd(971, "hit", 8, 2, [slot(1928, 2513, "hit")])])
                           + "\n").encode(),
             shadow_rel: pick_json("2026-08-10"), manual_rel: pick_json("2026-08-10", dd={})}
    routed = {rel: route(rel) for rel in files if route(rel)}
    parsed = _parse(files, routed)
    good = account(files, routed, parsed)
    check_sources(files, routed, good)
    with pytest.raises(InvariantError, match="exclusion_over_records"):
        check_sources(files, routed, account(files, routed, {**parsed, dec_rel: Parsed()}))
    relabelled = [dict(a, state="declared_missing", reason="declared_missing")
                  if (a["source_path"], a["locator"]) == (contest_rel, "line=1") else a for a in good]
    with pytest.raises(InvariantError, match="declared_missing_over_records"):
        check_sources(files, routed, relabelled)
    shadow = next(a for a in good if a["source_path"] == shadow_rel)
    with pytest.raises(InvariantError, match="file_accounting_shape"):
        check_sources(files, routed, good + [dict(shadow, locator="slot=phantom")])
    primary_id = next(a["obs_id"] for a in good if (a["source_path"], a["locator"]) == (manual_rel, "slot=primary"))
    forged = [dict(a, obs_id=primary_id) if (a["source_path"], a["locator"]) == (manual_rel, "slot=double_down")
              else a for a in good]
    with pytest.raises(InvariantError, match="identity does not match"):
        check_sources(files, routed, forged)


def test_check_sources_hashes_each_file_once(monkeypatch):
    # Final review #2: the identity check must not re-hash a file per occurrence — the real contest ledger holds
    # tens of thousands of occurrences in one multi-megabyte file.
    from scripts.audit.season_ledger import reconcile
    rel = "picks/account_state/contest_ledger.jsonl"
    lines = [contest_line(f"2026-08-2{i}T14:30:00Z", [rnd(971 + i, "hit", i, 1, [slot(1928 + i, 2513, "hit")])])
             for i in range(3)]
    files, routed = {rel: ("\n".join(lines) + "\n").encode()}, {rel: "contest_ledger"}
    acc = account(files, routed, _parse(files, routed))
    calls, real = [], reconcile.sha256_hex
    monkeypatch.setattr(reconcile, "sha256_hex", lambda data: calls.append(1) or real(data))
    check_sources(files, routed, acc)
    assert len(acc) == 9 and len(calls) == len(files)


def test_only_a_readable_empty_container_is_a_file_without_records():
    empty, unreadable = "static/units/20260927T230002Z.json.gz", "static/units/20260926T230002Z.json"
    files = {empty: gz(dumps({"units": []})), unreadable: dumps({"error": "x"})}
    routed = {empty: "units", unreadable: "units"}
    parsed = _parse(files, routed)
    acc = account(files, routed, parsed)
    assert {(a["source_path"], a["state"], a["reason"]) for a in acc} == {
        (empty, "excluded", "no_records"), (unreadable, "quarantined", "missing_units_list")}
    check_sources(files, routed, acc)
    with pytest.raises(InvariantError, match="exclusion_over_records"):
        check_sources(files, routed, account(files, routed, {**parsed, unreadable: Parsed()}))


def _link(membership, accounting, ledger_rows):
    """The join the compiler performs (Task 10), written out plainly for these tests."""
    by_loc = {(a["source_path"], a["locator"]): a for a in accounting}
    canonical = {r["pick_obs_id"]: r for r in ledger_rows if r.get("pick_obs_id")}
    out = []
    for m in membership:
        src = by_loc.get((m["source_path"], f"slot={m['slot']}")) or by_loc[(m["source_path"], "file")]
        link = canonical.get(src["obs_id"])
        out.append(dict(m, source_obs_id=src["obs_id"], source_state=src["state"], source_reason=src["reason"],
                        occurrence_disposition=src["disposition"], selection_id=link["selection_id"] if link else None,
                        canonical_bts_outcome=link.get("bts_outcome") if link else None))
    return out


def test_membership_checks_prove_the_join_the_cardinality_and_the_totals():
    # Codex plan r3 #3: a real occurrence id is not enough — it must be the right one, every rule must list every
    # universe slot exactly once, and the included rows must add up to the reported totals.
    prod, shadow = "picks/2026-08-20.json", "picks/2026-08-20.shadow.json"
    files = {prod: pick_json("2026-08-20", result="hit"), shadow: pick_json("2026-08-20", result="hit")}
    routed = {prod: "pick_file"}
    parsed = _parse(files, routed)
    for row in parsed[prod].rows:
        row["file_date"] = "2026-08-20"              # as the compiler adds it; no season dates here, so out of window
    acc = account(files, routed, parsed)
    assign_dispositions(acc, {a["obs_id"]: "outside_season_window" for a in acc if a["kind"] == "pick_file"})
    summary, membership = evaluate_rules(files, RULES, {}, frozen_at="t")
    linked = _link(membership, acc, [])
    check_invariants(files, acc, [], [], linked, summary, [])
    shadow_file = next(a for a in acc if a["source_path"] == shadow)
    wrong = [dict(m, source_obs_id=shadow_file["obs_id"], source_state="excluded", source_reason=shadow_file["reason"],
                  occurrence_disposition=None) if m["source_path"] == prod else m for m in linked]
    with pytest.raises(InvariantError, match="not linked to the occurrence"):
        check_invariants(files, acc, [], [], wrong, summary, [])
    with pytest.raises(InvariantError, match="exactly once"):
        check_invariants(files, acc, [], [], linked[1:], summary, [])
    with pytest.raises(InvariantError, match="exactly once"):
        check_invariants(files, acc, [], [], linked + linked[:1], summary, [])
    inflated = [dict(s, primaries=s["primaries"] + 1) if s["rule_id"] == "S1" else s for s in summary]
    with pytest.raises(InvariantError, match="add up"):
        check_invariants(files, acc, [], [], linked, inflated, [])


def test_output_checks_catch_bad_dispositions_references_joins_links_and_days():
    days, ok, summary = ["2026-05-01"], [_day("2026-05-01")], EMPTY_SUMMARY
    acc = _acc(("a", "file")) + _acc(("d", "file"), kind="decision", disposition="canonical_decision",
                                     fields={"file_date": "2026-05-01", "action": "skip"})
    check_invariants({}, acc, [], [_day("2026-05-01", decision_obs_id="o:d:file")], [], summary, days)
    with pytest.raises(InvariantError, match="without a disposition"):
        check_invariants({}, [dict(acc[0], disposition=None)], [], ok, [], summary, days)
    with pytest.raises(InvariantError, match="unknown disposition"):
        check_invariants({}, [dict(acc[0], disposition="whatever")], [], ok, [], summary, days)
    with pytest.raises(InvariantError, match="is not an emitted decision"):
        check_invariants({}, acc, [], [_day("2026-05-01", decision_obs_id="o:a:file")], [], summary, days)   # wrong kind
    with pytest.raises(InvariantError, match="referenced by no ledger row"):
        check_invariants({}, acc, [], ok, [], summary, days)
    canonical = _acc(("p", "slot=primary"), disposition="canonical_selection")
    with pytest.raises(InvariantError, match="exactly one ledger row"):
        check_invariants({}, canonical, [], ok, [], summary, days)
    two = [{"selection_id": "s1", "round_id": 1, "unit_id": 1, "player_id": 1, "match": "inferred"},
           {"selection_id": "s1", "round_id": 1, "unit_id": 2, "player_id": 1, "match": "inferred"}]
    with pytest.raises(InvariantError, match="more than one contest slot"):
        check_invariants({}, _acc(("a", "file")), two, ok, [], summary, days)
    with pytest.raises(InvariantError, match="no ledger row"):
        check_invariants({}, _acc(("a", "file")), [], [], [], summary, days)
    sel = {"row_id": "2026-05-01|primary|1|2", "row_kind": "selection", "date": "2026-05-01", "slot": "primary"}
    with pytest.raises(InvariantError, match="mixes"):
        check_invariants({}, _acc(("a", "file")), [], [_day("2026-05-01"), sel], [], summary, days)


def _rule(files, grading, published):
    return {"recipe": "tally_0914", "files": files, "primary": grading, "legs": grading,
            "window": ("2026-03-29", "2026-09-13"), "published": published, "recipe_date": "2026-09-14"}


def test_two_rules_with_equal_totals_and_different_members_are_both_reported():
    # Codex plan r1 #10: the alternatives must include different records, not only an ungraded extra file.
    files = {"picks/2026-05-01.json": pick_json("2026-05-01", result="hit"),
             "picks/2026-05-02.json": pick_json("2026-05-02", dd={}, result="miss", slot_results={"double_down": "hit"}),
             "picks/2026-05-03.json": pick_json("2026-05-03", slot_results={"pick": "hit"})}
    summary, membership = evaluate_rules(files, {"A": _rule("F1", "G1", (2, 1)), "B": _rule("F1", "G2", (2, 1))}, {},
                                         frozen_at="2026-09-28T16:00:00.000000Z")
    assert {s["rule_id"]: s["fit"] for s in summary} == {"A": "both", "B": "both"}
    members = {r: {m["recipe_slot_key"] for m in membership if m["rule_id"] == r and m["included"]} for r in "AB"}
    assert members["A"] != members["B"] and len(members["A"]) == len(members["B"]) == 3


def test_recipe_labels_follow_the_spec():
    summary = [{"recipe": "scorecard_0911", "fit": "none"}, {"recipe": "tally_0914", "fit": "primaries_only"},
               {"recipe": "tally_0914", "fit": "none"}]
    assert recipe_labels(summary) == {"scorecard_0911": "hypothesis", "tally_0914": "unrecoverable"}
    assert recipe_labels(summary + [{"recipe": "tally_0914", "fit": "both"}])["tally_0914"] == "hypothesis"


def test_candidate_rules_and_their_evaluator_are_fingerprinted():
    # Codex plan r2 #6 and r3 #4: the digest binds the rule table and everything the evaluator reads — the source of
    # every repo-local function reachable from the recipe roots and every configuration value, regex flags included.
    assert rules_fingerprint() == RULES_SHA256
    assert sorted(recipe_closure()) == RECIPE_CLOSURE
    assert sorted(RULES, key=lambda k: (k[0], int(k[1:]))) == [f"S{i}" for i in range(1, 9)] + [
        f"T{i}" for i in range(1, 25)]


def test_the_fingerprint_binds_regex_flags_and_fails_closed_on_unknown_configuration(monkeypatch):
    import re

    from scripts.audit.season_ledger import reconcile
    monkeypatch.setitem(reconcile.FILE_SETS, "F1", re.compile(reconcile.FILE_SETS["F1"].pattern, re.IGNORECASE))
    assert rules_fingerprint() != RULES_SHA256
    monkeypatch.setattr(reconcile, "GRADED", object())
    with pytest.raises(TypeError, match="cannot bind"):
        rules_fingerprint()


def test_a_recipe_slot_is_present_when_its_value_is_not_null():
    # Codex code r1 #6, pinned before any real count: the frozen rules read a slot as present when its value is not
    # null — an object in every production file. A malformed non-null value (false, a number, a list) is quarantined
    # as an occurrence, but the recipes still count it as a present slot, exactly as the fingerprinted code reads it.
    for value in (False, 0, [], {"batter_id": 2}):
        doc = {"pick": {"batter_id": 1}, "double_down": value, "result": "hit"}
        assert set(slot_inclusion(doc, "G4", "G4")) == {"primary", "double_down"}, value
        assert slot_inclusion(doc, "G1", "G1")["double_down"] == ("hit", True), value
    assert set(slot_inclusion({"pick": {"batter_id": 1}, "double_down": None, "result": "hit"}, "G4", "G4")) == {
        "primary"}


def test_grading_truth_table_covers_every_label_shape():
    labels = ("hit", "miss", "void", "suspended", "unresolved", None, {"odd": 1})
    single = {label if isinstance(label, (str, type(None))) else "object": {g: slot_inclusion(
        {"pick": {}, "double_down": None, "result": label}, g, g)["primary"][1] for g in ("G1", "G2", "G3", "G4")}
        for label in labels}
    assert single == {
        "hit": {"G1": True, "G2": True, "G3": True, "G4": True}, "miss": {"G1": True, "G2": True, "G3": True, "G4": True},
        "void": {"G1": False, "G2": False, "G3": True, "G4": True},
        "suspended": {"G1": False, "G2": False, "G3": True, "G4": True},
        "unresolved": {"G1": False, "G2": False, "G3": True, "G4": True},
        None: {"G1": False, "G2": False, "G3": False, "G4": True},
        "object": {"G1": False, "G2": False, "G3": True, "G4": True}}
    dd_slots = {"pick": {}, "double_down": {}, "result": "miss", "slot_results": {"pick": "void", "double_down": "hit"}}
    dd_bare = {"pick": {}, "double_down": {}, "result": "miss"}
    assert {g: {s: c for s, (_, c) in slot_inclusion(dd_slots, g, g).items()} for g in ("G1", "G2", "G3", "G4")} == {
        "G1": {"primary": True, "double_down": True}, "G2": {"primary": False, "double_down": True},
        "G3": {"primary": True, "double_down": True}, "G4": {"primary": True, "double_down": True}}
    assert {g: {s: c for s, (_, c) in slot_inclusion(dd_bare, g, g).items()} for g in ("G1", "G2", "G3", "G4")} == {
        "G1": {"primary": True, "double_down": True}, "G2": {"primary": False, "double_down": False},
        "G3": {"primary": True, "double_down": True}, "G4": {"primary": True, "double_down": True}}


def test_membership_universe_records_exclusions_and_the_evidence_interval():
    files = {f"picks/2026-05-0{i}.json": pick_json(f"2026-05-0{i}", result="hit") for i in (1, 2, 3)}
    files.update({"picks/2026-09-12.json": pick_json("2026-09-12", result="hit"),
                  "picks/2026-05-04.shadow.json": pick_json("2026-05-04", result="hit"),
                  "picks/2026-05-05.json": pick_json("2026-05-05", result="void"),
                  "picks/2026-05-06.json": pick_json("2026-05-06", result={"odd": 1})})   # never crashes a recipe
    mtimes = {"picks/2026-05-01.json": "2026-05-02T03:00:00.000000Z",    # 5/01 ET: before the 9/11 recipe
              "picks/2026-05-02.json": "2026-09-11T16:00:00.000000Z",    # same ET day: order unknown
              "picks/2026-05-03.json": "2026-09-12T16:00:00.000000Z"}    # after
    rule = {"R": {"recipe": "scorecard_0911", "files": "F1", "primary": "G1", "legs": "G1",
                  "window": ("2026-03-29", "2026-09-10"), "published": (3, 0), "recipe_date": "2026-09-11"}}
    summary, membership = evaluate_rules(files, rule, mtimes, frozen_at="2026-09-28T16:00:00.000000Z")
    by_path = {m["source_path"]: m for m in membership}
    assert {p: m["exclusion_reason"] for p, m in by_path.items()} == {
        "picks/2026-05-01.json": None, "picks/2026-05-02.json": None, "picks/2026-05-03.json": None,
        "picks/2026-09-12.json": "outside_window", "picks/2026-05-04.shadow.json": "not_in_file_set",
        "picks/2026-05-05.json": "value_not_counted", "picks/2026-05-06.json": "value_not_counted"}
    assert [by_path[f"picks/2026-05-0{i}.json"]["mtime_after_recipe_date"] for i in (1, 2, 3)] == [False, None, True]
    assert {(m["historical_membership"], m["frozen_at_utc"]) for m in membership} == {
        ("unknown", "2026-09-28T16:00:00.000000Z")}
    assert summary[0]["fit"] == "both"
