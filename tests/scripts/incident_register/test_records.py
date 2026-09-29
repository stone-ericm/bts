"""Register record validation (design v3.1 §2, §3, §7; plan rev 2 Task 8 amendment).

Runs only where ``jsonschema`` is importable (``uv run --with jsonschema==4.23.0 pytest ...``); the
package is deliberately not added to the project lock, which the box syncs on deploy.
"""
import copy
import hashlib
import json
import tempfile

import pytest

pytest.importorskip("jsonschema")

from scripts.audit.incident_register.records import REGISTRY_PATH, validate  # noqa: E402

ORACLE = "tests/test_incident_register_2026.py::test_e77_singleton_slate_moved_up_is_delivered_before_cutoff"

# the accepted pair behind base()'s expected-failure fixture, and the registry that pair accepted
EF_REGISTRY = [{"incident": "E77", "node": ORACLE,
                "exception": "tests.test_incident_register_2026.SingletonSlateUndelivered",
                "connection": {"kind": "derived", "review": "fixture reviewed"}}]
EF_ACCEPTANCE = json.dumps({"verdict": "accepted", "accepted_nodes": [ORACLE],
                            "connections": {ORACLE: "exception_shape"},
                            "registry_sha256": hashlib.sha256(json.dumps(EF_REGISTRY, sort_keys=True).encode()).hexdigest()})
EF_ACCEPTANCE_SHA = hashlib.sha256(EF_ACCEPTANCE.encode()).hexdigest()


def ef_root(root) -> "Path":
    """An evidence root holding base()'s expected-failure acceptance artifact and its registry."""
    from pathlib import Path
    root = Path(root)
    (root / REGISTRY_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REGISTRY_PATH).write_text(json.dumps({"entries": EF_REGISTRY}))
    (root / "evidence/expected_failure").mkdir(parents=True, exist_ok=True)
    (root / "evidence/expected_failure/acceptance.json").write_text(EF_ACCEPTANCE)
    return root


_ROOT = ef_root(tempfile.mkdtemp(prefix="w15-records-"))


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
                                           "acceptance": "evidence/expected_failure/acceptance.json",
                                           "acceptance_sha256": EF_ACCEPTANCE_SHA}],
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
    return validate(list(records), evidence_root=_ROOT)


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
    assert validate([r], publish=False)                 # the artifact binding is exercised in publication tests
    r["fixtures"]["current_defence"][0]["reviewer_decision"] = "accept"
    assert validate([r], publish=False) == []


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
    assert any("ev2 is not contemporaneous with the claim at /fix[0]/verified_recovered/at" in e for e in late), late
    early = errors_for(latent_with_a_verification_report("2026-09-04T11:00:00-04:00"))
    assert any("ev2 is not contemporaneous with the claim at /fix[0]/verified_recovered/at" in e for e in early), early
    uncited = latent_with_a_verification_report("2026-09-04T12:30:00-04:00")
    uncited["fix"][0]["verified_recovered"] = "unknown"
    assert any("ev2 operator report is not cited by any dated claim it describes" in e for e in errors_for(uncited))


def test_an_operator_report_can_date_an_install():
    r = latent_with_a_verification_report("2026-09-04T12:30:00-04:00")
    r["fix"][0]["verified_recovered"] = "unknown"
    r["fix"][0]["deployed"] = {"basis": "operator_report", "live_by": "2026-09-04T12:30:00-04:00", "evidence": ["ev2"]}
    assert errors_for(r) == []
    r["evidence"][1]["written_at"] = "2026-09-07T12:30:00-04:00"         # the report dates it three days late
    assert any("ev2 is not contemporaneous with the claim at /fix[0]/deployed" in e for e in errors_for(r))


def test_a_continuing_condition_is_witnessed_only_at_its_observed_times():
    """Ruling 6 as replaced (Codex phase-1 r4): absence of a known end is not evidence of continuity; a
    report describing a still-present condition is qualified against an OBSERVED time, never an onset
    long past."""
    r = base()
    o = r["occurrences"][0]
    o.update(onset={"at": "2026-07-01", "evidence": ["ev1"]}, continuing=True)
    errs = errors_for(r)
    assert any("ev1 is not contemporaneous with the claim at /occurrences[0]/onset" in e for e in errs), errs
    r["evidence"].append({"id": "ev2", "kind": "inference", "strength": "primary", "locator": "src/bts/scheduler.py",
                          "what": "the defective plan dates from the 7/01 deploy"})
    o["onset"] = {"not_before": "2026-07-01", "not_after": "2026-07-16", "evidence": ["ev2"]}   # inferred onset
    o["observed"] = [{"at": "2026-07-16T09:00:00-04:00", "evidence": ["ev1"]}]                # witnessed state
    assert errors_for(r) == []


def test_an_occurrence_names_only_existing_links():
    r = base()
    r["occurrences"][0]["links"] = [3]
    assert any("an occurrence names link 3, which the mechanism does not have" in e for e in errors_for(r))


def test_a_repair_only_report_does_not_make_an_observed_incident():
    """Codex phase-1 r4 #5.1: a report cited only by a verification step promoted the record."""
    r = base()
    r["occurrences"] = []
    r["evidence"][0]["what"] = "verified the repair installed; no failure occurrence witnessed"
    r["fix"][0]["verified_recovered"] = {"how": "verified code live",
                                         "at": {"at": "2026-07-17T09:00:00-04:00", "evidence": ["ev1"]}}
    assert any("observed_incident needs a qualified primary witness" in e for e in errors_for(r))


def test_each_citation_of_a_report_is_qualified_on_its_own():
    """Codex phase-1 r4 #5.2/#5.3: one timely citation qualified an April occurrence and an April step."""
    r = base()
    old = copy.deepcopy(r["occurrences"][0])
    old["onset"] = {"at": "2026-04-01", "evidence": ["ev1"]}
    r["occurrences"].append(old)
    assert any("ev1 is not contemporaneous with the claim at /occurrences[1]/onset" in e for e in errors_for(r))
    r = base()
    r["fix"][0]["mitigated"] = {"action": "first action", "at": {"at": "2026-04-01", "evidence": ["ev1"]}}
    r["fix"][0]["verified_recovered"] = {"how": "verified live", "at": {"at": "2026-07-17T09:00:00-04:00", "evidence": ["ev1"]}}
    errs = errors_for(r)
    assert any("ev1 is not contemporaneous with the claim at /fix[0]/mitigated/at" in e for e in errs), errs
    assert not any("verified_recovered" in e for e in errs), errs


def test_an_impossible_chronology_is_rejected():
    """Codex phase-1 r4 #6.2: detection an hour BEFORE the onset validated as a zero latency."""
    r = base()
    o = r["occurrences"][0]
    o["onset"] = {"at": "2026-07-16T18:00:00-04:00", "evidence": ["ev1"]}
    o["first_machine_detection"] = {"detector": "test", "at": {"at": "2026-07-16T17:00:00-04:00", "evidence": ["ev1"]}}
    o["latencies"]["detection"] = {"min_minutes": 0, "max_minutes": 0}
    assert any("impossible chronology: the detection interval ends before it starts" in e for e in errors_for(r))


def test_publication_binds_expected_failure_fixtures(tmp_path):
    """Codex phase-1 r4 #6.1: a path to a missing acceptance file satisfied publication."""
    assert errors_for(base()) == []                                     # bound to the fixture root's pair
    bare = tmp_path / "bare"
    bare.mkdir()
    assert any("expected-failure acceptance file evidence/expected_failure/acceptance.json not found" in e
               for e in validate([base()], evidence_root=bare))
    r = base()
    r["fixtures"]["expected_failure"][0]["acceptance_sha256"] = "2" * 64
    assert any("expected-failure acceptance file hash mismatch" in e for e in errors_for(r))
    r = base()
    r["fixtures"]["expected_failure"][0]["exception"] = "PassGradedAsMiss"
    assert any("is not registered with the record's exception PassGradedAsMiss" in e for e in errors_for(r))
    root = ef_root(tmp_path / "other")
    (root / REGISTRY_PATH).write_text(json.dumps({"entries": EF_REGISTRY + [dict(EF_REGISTRY[0], node="x")]}))
    assert any("the registry differs from the one the pair accepted" in e for e in validate([base()], evidence_root=root))


def test_publication_without_an_evidence_root_is_refused_for_bound_claims():
    assert any("publication needs an evidence root" in e for e in validate([base()]))
    assert validate([base()], publish=False) == []                      # draft mode binds nothing


def test_an_open_ended_claim_anchors_no_report():
    """A claim with no latest instant cannot place a report within 48 h of anything."""
    r = base()
    r["occurrences"][0]["onset"] = {"not_before": "2026-07-16", "evidence": ["ev1"]}
    assert any("ev1 is not contemporaneous with the claim at /occurrences[0]/onset" in e for e in errors_for(r))


def test_operator_awareness_alone_does_not_witness_the_deviation():
    """Witness roles are the onset, observed times and machine detection; awareness is not a witness."""
    r = base()
    r["evidence"].append({"id": "ev2", "kind": "inference", "strength": "primary", "locator": "src/bts/scheduler.py",
                          "what": "the stale plan"})
    r["occurrences"][0]["onset"] = {"at": "2026-07-16", "evidence": ["ev2"]}
    assert any("observed_incident needs a qualified primary witness" in e for e in errors_for(r))


def test_a_report_written_at_the_claims_own_instant_qualifies():
    """Found on the real drafts: 'operator acted at T' cited by the report written at T was refused."""
    r = base()
    r["occurrences"][0]["operator_action"] = {"at": "2026-07-17T10:00:00-04:00", "evidence": ["ev1"]}
    assert errors_for(r) == []
    r["evidence"][0]["written_at"] = "2026-07-17T09:59:59-04:00"          # one second before the action
    assert any("ev1 is not contemporaneous with the claim at /occurrences[0]/operator_action" in e for e in errors_for(r))
