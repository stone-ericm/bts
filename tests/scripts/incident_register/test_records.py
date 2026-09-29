"""Register record validation (design v3.1 §2, §3, §7; plan rev 2 Task 8 amendment).

Runs only where ``jsonschema`` is importable (``uv run --with jsonschema==4.23.0 pytest ...``); the
package is deliberately not added to the project lock, which the box syncs on deploy.
"""
import copy

import pytest

pytest.importorskip("jsonschema")

from scripts.audit.incident_register.records import validate  # noqa: E402

ORACLE = "tests/test_incident_register_2026.py::test_e77_singleton_slate_moved_up_is_delivered_before_cutoff"


def base():
    return {
        "id": "I-001", "working_id": "E77", "title": "7/16 singleton-slate pick never delivered",
        "disposition": "observed_incident", "counted": True,
        "classes": {"primary": "D", "secondary": ["A"]},
        "tier": "A", "plan_named": True,
        "delivery_mode": "dm", "policy_objective": "reach57", "stream": "production", "authority": "local",
        "contract": {"text": "an enterable pick is delivered before first pitch minus 5 minutes",
                     "source": "design §10.3; picks.SUBMISSION_CUTOFF_MIN"},
        "mechanism": [{"n": 1, "text": "check times are computed once, at the morning fetch",
                       "code": [{"path": "src/bts/scheduler.py", "ref": "7b70da7", "ref_basis": "candidate"}]}],
        "fix": [{"link": 1, "implemented": "unfixed", "deployed": "not_applicable", "mitigated": "none",
                 "verified_recovered": "not_applicable"}],
        "fixtures": {"historical_replay": [], "current_defence": [],
                     "expected_failure": [{"nodes": [ORACLE], "exception": "SingletonSlateUndelivered",
                                           "acceptance": "evidence/expected_failure/acceptance.json"}],
                     "characterization": []},
        "residual": [{"text": "the singleton-slate gap is unfixed", "reaches_production": True}],
        "watchdog": {"boundary": "singleton_slate", "trigger": "no delivery by first pitch minus 5",
                     "recovery_assertion": "a pick file with delivered_at before the cutoff"},
        "evidence": [{"id": "ev1", "kind": "contemporaneous_operator_report", "strength": "primary",
                      "locator": "commit 7b70da7 body", "written_at": "2026-07-17T10:00:00-04:00",
                      "what": "operator write-up of the undelivered singleton-slate day"}],
        "occurrences": [{
            "onset": {"at": "2026-07-16", "evidence": ["ev1"]},
            "first_detectable": "unknown", "first_machine_detection": "unknown",
            "alert": {"attempted": "unknown", "confirmed": "unknown", "failed": "unknown"},
            "operator_awareness": {"not_after": "2026-07-17T10:00:00-04:00", "evidence": ["ev1"]},
            "mitigation": "none", "restored_verification": "not_applicable",
            "latencies": {"detection": "unknown", "notification": "unknown", "recovery": "unknown"}}],
        "routes": {"H": ["7b70da7"], "R": "pending_phase2", "X": []},
    }


def fixed_tier_a():
    r = base()
    r.update(id="I-002", working_id="E85", plan_named=False)
    r["fix"] = [{"link": 1, "implemented": {"commits": ["0abf503"]},
                 "deployed": {"basis": "log", "sha": "0abf503", "live_by": "2026-09-03T20:00:00Z", "run_id": 1,
                              "evidence": ["ev1"]},
                 "mitigated": "none", "verified_recovered": "unknown"}]
    r["fixtures"] = {"historical_replay": [{"status": "unavailable", "reason": "new-API-only red nodes"}],
                     "current_defence": [{"link": 1, "status": "survivor", "patch": "mutants/x.patch",
                                          "survivor_class": "coverage_gap", "reason": "no node kills it"}],
                     "expected_failure": [], "characterization": []}
    r["residual"] = []
    return r


def errors_for(*records):
    return validate(list(records))


def test_valid_records_pass():
    assert errors_for(base(), fixed_tier_a()) == []


@pytest.mark.parametrize("mutate, needle", [
    (lambda r: r.update(extra=1), "Additional properties"),
    (lambda r: r.update(disposition="incident"), "disposition"),
    (lambda r: r["occurrences"][0].update(onset={"at": "7/16", "evidence": ["ev1"]}), "onset"),
    (lambda r: r["evidence"][0].update(kind="memo"), "kind"),
])
def test_schema_violations_are_reported(mutate, needle):
    r = base()
    mutate(r)
    errs = errors_for(r)
    assert errs and any(needle in e for e in errs), errs


def test_duplicate_ids():
    assert any("duplicate id" in e for e in errors_for(base(), base()))


def test_dangling_evidence_reference():
    r = base()
    r["occurrences"][0]["onset"]["evidence"] = ["ev9"]
    assert any("unknown evidence id ev9" in e for e in errors_for(r))


def test_dangling_related_record():
    r = base()
    r["related"] = ["I-099"]
    assert any("unknown related record I-099" in e for e in errors_for(r))


def test_observed_incident_needs_primary_observation_or_operator_report():
    r = base()
    r["evidence"][0].update(strength="reported")
    assert any("observed_incident needs" in e for e in errors_for(r))
    r["evidence"][0].update(strength="primary", kind="inference")
    assert any("observed_incident needs" in e for e in errors_for(r))


def test_operator_report_must_be_contemporaneous():
    r = base()
    del r["evidence"][0]["written_at"]
    assert any("written_at" in e for e in errors_for(r))
    r = base()
    r["evidence"][0]["written_at"] = "2026-07-20T10:00:00-04:00"   # > 48 h after the 7/16 onset
    assert any("not contemporaneous" in e for e in errors_for(r))


def test_unresolved_candidate_names_missing_evidence():
    r = base()
    r.update(disposition="unresolved_candidate")
    assert any("missing_evidence" in e for e in errors_for(r))


def test_counted_flag_follows_disposition():
    r = base()
    r.update(disposition="pre_ship_exclusion")
    assert any("counted" in e for e in errors_for(r))
    r = base()
    r.update(counted=False)
    assert any("counted" in e for e in errors_for(r))


def test_fixed_tier_a_incident_needs_replay_and_per_link_defence():
    r = fixed_tier_a()
    r["fixtures"]["historical_replay"] = []
    assert any("historical_replay" in e for e in errors_for(r))
    r = fixed_tier_a()
    r["mechanism"].append({"n": 2, "text": "second causal link", "code": []})
    r["fix"].append({"link": 2, "implemented": {"commits": ["eb010fd"]}, "deployed": "unknown",
                     "mitigated": "none", "verified_recovered": "unknown"})
    assert any("current_defence has no entry for link 2" in e for e in errors_for(r))


def test_fix_links_must_name_mechanism_links():
    r = fixed_tier_a()
    r["fix"][0]["link"] = 3
    assert any("fix names link 3" in e for e in errors_for(r))


def test_plan_named_needs_a_fixture_or_a_deferral():
    r = base()
    r["fixtures"]["expected_failure"] = []
    errs = errors_for(r)
    assert any("plan-named" in e for e in errs)
    r["fixtures"]["deferred"] = "no faithful scenario can be built without box data"
    assert not any("plan-named" in e for e in errors_for(r))


def test_unfixed_defect_with_a_fixed_contract_needs_an_expected_failure():
    r = base()
    r.update(plan_named=False)
    r["fixtures"]["expected_failure"] = []
    assert any("expected-failure" in e for e in errors_for(r))


def test_tier_b_with_a_residual_reaching_production_is_tier_a():
    r = base()
    r.update(tier="B")
    assert any("B → A" in e for e in errors_for(r))


def test_tier_pending_needs_a_reason():
    r = base()
    r.update(tier="tier_pending")
    r["residual"] = []
    assert any("tier_reason" in e for e in errors_for(r))


def test_bounds_must_be_ordered():
    r = base()
    r["occurrences"][0]["onset"] = {"not_before": "2026-07-17", "not_after": "2026-07-16", "evidence": ["ev1"]}
    assert any("not_before after not_after" in e for e in errors_for(r))


def test_certified_defence_requires_an_accepting_reviewer():
    r = fixed_tier_a()
    r["fixtures"]["current_defence"] = [{
        "link": 1, "status": "certified", "level": "component", "patch": "mutants/x.patch",
        "patch_sha256": "0" * 64, "killing_nodes": ["tests/x.py::test_y"], "acceptance": "a.json",
        "acceptance_sha256": "1" * 64, "reviewer_decision": "reject"}]
    assert errors_for(r)
    r["fixtures"]["current_defence"][0]["reviewer_decision"] = "accept"
    assert errors_for(r) == []


def test_numeric_latency_needs_bounded_endpoints():
    r = base()
    r["occurrences"][0]["latencies"]["detection"] = {"min_minutes": 0, "max_minutes": 60}
    assert any("detection latency" in e for e in errors_for(r))


def test_codex_r3_invalid_records_are_rejected():
    """r3 #6 measured: four invalid records validated clean."""
    r = fixed_tier_a()
    r["fixtures"]["historical_replay"] = [{"status": "certified", "label": "semantic_regression_replay",
                                           "fix_set": [], "symptom_nodes": ["anything"], "acceptance": ""}]
    assert errors_for(r)
    r = fixed_tier_a()
    r["fix"][0]["deployed"] = {"basis": "log"}
    assert errors_for(r)
    r = base()
    o = r["occurrences"][0]
    o["alert"]["confirmed"] = {"at": "2026-07-16T19:00:00-04:00", "evidence": ["ev1"]}
    o["latencies"]["notification"] = {"min_minutes": 50, "max_minutes": 2}
    errs = errors_for(r)
    assert any("min > max" in e for e in errs) and any("endpoint times bounded" in e for e in errs)
    r = base()
    r["evidence"][0]["written_at"] = "2026-07-01T10:00:00-04:00"          # before the 7/16 occurrence
    assert any("not contemporaneous" in e for e in errors_for(r))


def test_latency_tighter_than_its_bounds_is_rejected():
    r = base()
    o = r["occurrences"][0]
    o["onset"] = {"at": "2026-07-16T18:36:16-04:00", "evidence": ["ev1"]}
    o["first_machine_detection"] = {"detector": "hb", "at": {"not_before": "2026-07-16T18:39:00-04:00",
                                                             "not_after": "2026-07-16T18:41:00-04:00", "evidence": ["ev1"]}}
    o["latencies"]["detection"] = {"min_minutes": 3, "max_minutes": 5}
    assert any("tighter than its endpoint bounds" in e for e in errors_for(r))
    o["latencies"]["detection"] = {"min_minutes": 2, "max_minutes": 5}
    assert not any("detection" in e for e in errors_for(r))


def test_a_report_written_during_a_continuing_condition_is_contemporaneous():
    r = base()
    r["occurrences"][0]["onset"] = {"at": "2026-07-01", "evidence": ["ev1"]}   # began two weeks before the report
    assert any("not contemporaneous" in e for e in errors_for(r))             # as a one-shot event: too late
    r["occurrences"][0]["continuing"] = True
    assert not any("contemporaneous" in e for e in errors_for(r))             # a condition still present
    r["occurrences"][0]["mitigation"] = {"at": "2026-07-02", "evidence": ["ev1"]}
    assert any("not contemporaneous" in e for e in errors_for(r))             # written > 48 h after it ended


def test_an_uncited_operator_report_is_rejected():
    r = base()
    r["evidence"].append({"id": "ev2", "kind": "contemporaneous_operator_report", "strength": "primary",
                          "locator": "commit abc body", "written_at": "2026-07-16T20:00:00-04:00", "what": "extra"})
    assert any("ev2 operator report is not cited" in e for e in errors_for(r))


def test_publication_binds_certified_entries_to_their_acceptance(tmp_path):
    import hashlib
    import json as _json
    acc = {"verdict": "accepted", "patch_sha256": "0" * 64, "certificates": {"tests/x.py::test_y": {"ok": True}}}
    path = tmp_path / "acc.json"
    path.write_text(_json.dumps(acc))
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    r = fixed_tier_a()
    r["fixtures"]["current_defence"] = [{
        "link": 1, "status": "certified", "level": "component", "patch": "p.patch", "patch_sha256": "0" * 64,
        "killing_nodes": ["tests/x.py::test_y"], "acceptance": "acc.json", "acceptance_sha256": good,
        "reviewer_decision": "accept"}]
    assert validate([r], evidence_root=tmp_path) == []
    r["fixtures"]["current_defence"][0]["acceptance_sha256"] = "2" * 64
    assert any("hash mismatch" in e for e in validate([r], evidence_root=tmp_path))
    path.write_text(_json.dumps(dict(acc, verdict="rejected")))
    r["fixtures"]["current_defence"][0]["acceptance_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert any("runner verdict is 'rejected'" in e for e in validate([r], evidence_root=tmp_path))


def test_validation_does_not_mutate_input():
    r = base()
    before = copy.deepcopy(r)
    errors_for(r)
    assert r == before


def test_draft_mode_skips_only_the_fixture_completeness_rules():
    r = fixed_tier_a()
    r["fixtures"]["historical_replay"] = []
    r["fixtures"]["current_defence"] = []
    assert validate([r], publish=False) == []
    assert validate([r])                                   # publishing still refuses it
    r["evidence"][0]["strength"] = "reported"
    assert any("observed_incident needs" in e for e in validate([r], publish=False))


def test_a_config_fixture_satisfies_the_plan_named_rule():
    r = base()
    r["fixtures"]["expected_failure"] = []
    r["fix"][0]["implemented"] = "config_only"
    r["contract"]["source"] = "season-wrap plan"
    assert any("plan-named" in e for e in errors_for(r))
    r["fixtures"]["config"] = [{"nodes": ["tests/x.py::test_y"], "pins": "private mode alone still nags"}]
    assert not any("plan-named" in e for e in errors_for(r))


def test_a_date_only_report_covers_its_whole_day():
    """A note dated only by day may have been written at any time that day: not 'before' a same-day
    onset, and its 48 h window starts at the day's first instant."""
    r = base()
    r["occurrences"][0]["onset"] = {"at": "2026-07-16T15:45:00-04:00", "evidence": ["ev1"]}
    r["evidence"][0]["written_at"] = "2026-07-16"
    assert not any("contemporaneous" in e for e in errors_for(r))
    r["evidence"][0]["written_at"] = "2026-07-15"                       # the whole day precedes the onset
    assert any("not contemporaneous" in e for e in errors_for(r))


def latent_with_a_verification_report(written_at):
    """A latent defect (no occurrence) whose fix was verified live by an operator write-up."""
    r = fixed_tier_a()
    r.update(disposition="deployed_latent_defect", occurrences=[])
    r["evidence"] = [{"id": "ev1", "kind": "machine_observation", "strength": "primary",
                      "locator": "deploy run 1", "what": "deploy log"}]
    r["evidence"].append({"id": "ev2", "kind": "contemporaneous_operator_report", "strength": "primary",
                          "locator": "docs/audit/x.md", "written_at": written_at,
                          "what": "verified live on the box: the fixed path ran clean"})
    r["fix"][0]["verified_recovered"] = {"how": "operator verification on the box",
                                         "at": {"at": "2026-09-04T12:00:00-04:00", "evidence": ["ev2"]}}
    return r


def test_a_report_may_describe_the_fix_step_that_cites_it():
    """A report can describe a mitigation, install or verification rather than an occurrence; it is then
    judged against that step (written after it began, within 48 h of it), and still never uncited."""
    assert errors_for(latent_with_a_verification_report("2026-09-04T12:30:00-04:00")) == []
    late = errors_for(latent_with_a_verification_report("2026-09-08T12:30:00-04:00"))
    assert any("ev2 is not contemporaneous with the fix step it describes" in e for e in late), late
    early = errors_for(latent_with_a_verification_report("2026-09-04T11:00:00-04:00"))
    assert any("ev2 is not contemporaneous with the fix step it describes" in e for e in early), early
    uncited = latent_with_a_verification_report("2026-09-04T12:30:00-04:00")
    uncited["fix"][0]["verified_recovered"] = "unknown"
    assert any("ev2 operator report is not cited by any occurrence or fix step it describes" in e
               for e in errors_for(uncited))


def test_an_operator_report_can_date_an_install():
    r = latent_with_a_verification_report("2026-09-04T12:30:00-04:00")
    r["fix"][0]["verified_recovered"] = "unknown"
    r["fix"][0]["deployed"] = {"basis": "operator_report", "live_by": "2026-09-04T12:30:00-04:00", "evidence": ["ev2"]}
    assert errors_for(r) == []
    r["evidence"][1]["written_at"] = "2026-09-07T12:30:00-04:00"         # the report dates it three days late
    assert any("ev2 is not contemporaneous with the fix step it describes" in e for e in errors_for(r))


def two_link_continuing(report_at, links=None, mitigation=None):
    """Link 1 fixed early (7/02), link 2 fixed late (7/20); one continuing occurrence from 7/01."""
    r = fixed_tier_a()
    r["mechanism"] = [{"n": 1, "text": "first defect", "code": [{"path": "src/bts/a.py", "ref": "7b70da7", "ref_basis": "candidate"}]},
                      {"n": 2, "text": "second defect", "code": [{"path": "src/bts/b.py", "ref": "7b70da7", "ref_basis": "candidate"}]}]
    run = {"basis": "log", "sha": "0abf503", "run_id": 1, "evidence": ["ev1"]}
    r["fix"] = [dict(r["fix"][0], link=1, deployed=dict(run, live_by="2026-07-02T12:00:00Z")),
                dict(r["fix"][0], link=2, deployed=dict(run, live_by="2026-07-20T12:00:00Z"))]
    r["fixtures"]["current_defence"] = [dict(r["fixtures"]["current_defence"][0], link=n) for n in (1, 2)]
    r["evidence"] = [{"id": "ev1", "kind": "machine_observation", "strength": "primary", "locator": "run 1", "what": "deploy log"},
                     {"id": "ev2", "kind": "contemporaneous_operator_report", "strength": "primary",
                      "locator": "commit x body", "written_at": report_at, "what": "the stall is still there"}]
    o = r["occurrences"][0]
    o.update(onset={"at": "2026-07-01", "evidence": ["ev2"]}, continuing=True,
             operator_awareness={"not_after": report_at, "evidence": ["ev2"]})
    if links is not None:
        o["links"] = links
    if mitigation is not None:
        o["mitigation"] = {"at": mitigation, "evidence": ["ev2"]}
    return r


def test_a_continuing_condition_ends_at_its_earliest_end_event():
    """Self-review 2026-09-29: the end was the LATEST of mitigation / recovery / every fix install, so a
    report written long after a mitigation ended the condition still counted while a later fix existed."""
    late = errors_for(two_link_continuing("2026-07-10T12:00:00-04:00", links=[2], mitigation="2026-07-05T12:00:00-04:00"))
    assert any("ev2 is not contemporaneous with the occurrence" in e for e in late), late
    ok = errors_for(two_link_continuing("2026-07-06T12:00:00-04:00", links=[2], mitigation="2026-07-05T12:00:00-04:00"))
    assert not any("contemporaneous" in e for e in ok), ok


def test_an_occurrence_ends_with_the_fixes_of_its_own_links():
    """A multi-link record: the occurrence of link 2 ends with link 2's fix, not link 1's earlier one;
    naming no links means every link's fix, whose earliest then ends it."""
    assert not any("contemporaneous" in e for e in errors_for(two_link_continuing("2026-07-10T12:00:00-04:00", links=[2])))
    unnamed = errors_for(two_link_continuing("2026-07-10T12:00:00-04:00"))
    assert any("ev2 is not contemporaneous with the occurrence" in e for e in unnamed), unnamed
    assert any("names link 3, which the mechanism does not have" in e
               for e in errors_for(two_link_continuing("2026-07-06T12:00:00-04:00", links=[3])))
