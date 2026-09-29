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
                 "deployed": {"basis": "log", "sha": "0abf503", "live_by": "2026-09-03T20:00:00Z", "evidence": ["ev1"]},
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
        "reviewer_decision": "reject"}]
    assert errors_for(r)
    r["fixtures"]["current_defence"][0]["reviewer_decision"] = "accept"
    assert errors_for(r) == []


def test_numeric_latency_needs_bounded_endpoints():
    r = base()
    r["occurrences"][0]["latencies"]["detection"] = {"min_minutes": 0, "max_minutes": 60}
    assert any("detection latency" in e for e in errors_for(r))


def test_validation_does_not_mutate_input():
    r = base()
    before = copy.deepcopy(r)
    errors_for(r)
    assert r == before
