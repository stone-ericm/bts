"""Typed deploy-run extraction (design §4 R4; plan rev 2 Task 7 amendment)."""
import json
import subprocess
from pathlib import Path

import pytest

from scripts.audit.incident_register.deploy_runs import (
    extract, fetch_log, installed_timeline, live_at, first_live,
)

TS = "2026-09-22T16:28:{:02d}.5977631Z"


def line(sec, text, step="UNKNOWN STEP"):
    return f"deploy\t{step}\t{TS.format(sec)} {text}"


def echoed_script():
    """The ssh action prints the script source first; none of it may count."""
    return "\n".join([
        'deploy\tUNKNOWN STEP\techo "Pre-deploy SHA: $(echo $OLD_SHA | cut -c1-7)"',
        'deploy\tUNKNOWN STEP\techo "Deployed $NEW_SHA"',
        'deploy\tUNKNOWN STEP\techo "CANARY PASSED — $NEW_SHA is live and healthy"',
        'deploy\tUNKNOWN STEP\t  echo "CANARY FAILED — rolling back to $(echo $OLD_SHA | cut -c1-7)"',
        # the echoed source also appears with a timestamp in some action versions
        line(1, '  echo "Rolled back cleanly to $(echo $OLD_SHA | cut -c1-7)"'),
    ])


PASSED = "\n".join([
    echoed_script(),
    line(2, "==== 2895 passed, 1 skipped, 7 deselected, 2 warnings in 261.27s (0:04:21) ====="),
    line(38, "Pre-deploy SHA: d15d382"),
    line(50, "Deployed 4490aee"),
    line(51, "Waiting 30s for services to stabilize..."),
    line(59, "================================"),
    line(59, "CANARY PASSED — 4490aee is live and healthy"),
])

ROLLED_BACK = "\n".join([
    echoed_script(),
    line(38, "out: Pre-deploy SHA: 1111111"),
    line(40, "out: Deployed 2222222"),
    line(44, "out: CANARY FAILED — rolling back to 1111111"),
    line(45, "out: Reasons:"),
    line(46, "out:   - bts-scheduler.service is not healthy (active=failed ...)"),
    line(58, "out: Rolled back cleanly to 1111111"),
])


def test_script_echo_never_counts():
    got = extract(echoed_script())
    assert got["deployed_sha"] is None and got["pre_sha"] is None
    assert got["canary"] == "absent" and got["rollback"] == "none"
    assert got["template_hits"] == {}


def test_a_template_inside_other_output_never_counts():
    """The test gate's own output (e.g. a failing assertion) can contain template text mid-line."""
    got = extract("\n".join([line(3, "E       AssertionError: Deployed 1234567"),
                             line(4, "tests/x.py::test_y - CANARY PASSED — 1234567 is live and healthy")]))
    assert got["deployed_sha"] is None and got["canary"] == "absent" and got["template_hits"] == {}


def test_passed_canary():
    got = extract(PASSED)
    assert got["pre_sha"] == "d15d382" and got["deployed_sha"] == "4490aee"
    assert got["canary"] == "passed" and got["rollback"] == "none"
    assert got["deployed_at"] == "2026-09-22T16:28:50Z"
    assert got["test_gate"] == {"passed": 2895, "skipped": 1, "deselected": 7, "warnings": 2}
    assert got["anomalies"] == []


def test_failed_canary_with_clean_rollback():
    got = extract(ROLLED_BACK)
    assert (got["pre_sha"], got["deployed_sha"], got["canary"], got["rollback"]) == (
        "1111111", "2222222", "failed", "clean")
    assert got["rolled_back_at"] == "2026-09-22T16:28:58Z"
    assert got["test_gate"] is None


def test_failed_test_gate_counts():
    got = extract(line(3, "==== 2 failed, 2890 passed, 1 skipped in 200.01s (0:03:20) ===="))
    assert got["test_gate"] == {"failed": 2, "passed": 2890, "skipped": 1}
    assert got["deployed_sha"] is None


def test_duplicate_templates_are_ambiguous_not_guessed():
    got = extract("\n".join([line(1, "Deployed aaaaaaa"), line(2, "Deployed bbbbbbb")]))
    assert got["deployed_sha"] is None and "multiple_deployed" in got["anomalies"]


def test_canary_sha_mismatch_is_flagged():
    got = extract("\n".join([line(1, "Deployed aaaaaaa"), line(2, "CANARY PASSED — bbbbbbb is live and healthy")]))
    assert "canary_sha_mismatch" in got["anomalies"]


def test_unmatched_text_never_reaches_the_output():
    """Metamorphic: vary every non-template line; the typed output is identical."""
    base = extract(PASSED)
    noisy = PASSED.replace("Waiting 30s for services to stabilize...", "pick: Some Player p=0.81 streak=17")
    noisy += "\n" + line(59, "Traceback (most recent call last): secret-ish text")
    assert extract(noisy) == base
    assert "Some Player" not in json.dumps(extract(noisy))


class _Proc:
    def __init__(self, rc, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_fetch_classifies_expired_and_errors_without_echoing_stderr():
    expired = fetch_log(1, runner=lambda *a, **k: _Proc(1, err="failed to get run log: HTTP 410: Server Error (x)"))
    assert expired == ("unavailable_expired", None)
    other = fetch_log(2, runner=lambda *a, **k: _Proc(1, err="boom: token abc123"))
    assert other == ("unavailable_error", None)
    ok = fetch_log(3, runner=lambda *a, **k: _Proc(0, out=PASSED))
    assert ok == ("retained", PASSED)


def rec(day, *, log="retained", pre=None, dep=None, canary="passed", rollback="none", conclusion="success"):
    """A typed run on 2026-07-<day>: pre-deploy line at :00, deployed line at :02, rollback at :30."""
    t = lambda sec: f"2026-07-{day:02d}T00:00:{sec:02d}Z"   # noqa: E731
    return {"run_id": day, "created_at": t(0), "conclusion": conclusion, "log": log,
            "pre_sha": pre, "pre_at": t(0) if pre else None, "deployed_sha": dep, "deployed_at": t(2) if dep else None,
            "canary": canary, "canary_at": t(10) if dep else None, "rollback": rollback,
            "rolled_back_at": t(30) if rollback == "clean" else None,
            # a clean rollback's own log line names the SHA it reinstalled (here the pre-deploy SHA)
            "rolled_back_sha": pre if rollback == "clean" else None, "anomalies": []}


def at(sha, basis="assumed_continuous", candidate=None):
    return {"sha": sha, "basis": basis, "candidate": candidate}


def test_timeline_separates_observations_transitions_and_assumptions():
    runs = [rec(1, pre="aaaaaaa", dep="bbbbbbb"), rec(2, log="unavailable_expired", canary="absent"),
            rec(3, pre="ccccccc", dep="ddddddd"),
            rec(4, pre="ddddddd", dep="eeeeeee", canary="failed", rollback="clean")]
    tl = installed_timeline(runs)
    assert live_at(tl, "2026-07-01T00:00:00Z") == at("aaaaaaa", "observed")        # the pre-deploy line
    assert live_at(tl, "2026-07-01T00:00:01Z") == at(None, "transition")          # checkout/restart in flight
    assert live_at(tl, "2026-07-01T12:00:00Z") == at("bbbbbbb")                     # assumed until the next run
    assert live_at(tl, "2026-07-02T12:00:00Z") == at(None, "unknown", "ccccccc")    # expired log in between
    assert live_at(tl, "2026-07-03T12:00:00Z") == at("ddddddd")
    assert live_at(tl, "2026-07-04T00:00:10Z") == at(None, "transition")          # deployed e, then rolled back
    assert live_at(tl, "2026-07-04T12:00:00Z") == at("ddddddd")
    assert live_at(tl, "2026-06-30T12:00:00Z") == at(None, "unknown", "aaaaaaa")


def test_a_run_without_a_deploy_line_leaves_the_assumption_standing():
    runs = [rec(1, pre="aaaaaaa", dep="bbbbbbb"), rec(2, canary="absent", conclusion="failure")]
    assert live_at(installed_timeline(runs), "2026-07-02T12:00:00Z") == at("bbbbbbb")


def test_disagreeing_endpoints_are_drift_not_continuity():
    runs = [rec(1, pre="aaaaaaa", dep="bbbbbbb"), rec(2, pre="fffffff", dep="ccccccc")]
    tl = installed_timeline(runs)
    assert live_at(tl, "2026-07-01T12:00:00Z") == at(None, "drift")
    assert live_at(tl, "2026-07-02T12:00:00Z") == at("ccccccc")


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def chain(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@t")
    git(repo, "config", "user.name", "t")
    shas = []
    for i in range(3):
        (repo / "f").write_text(str(i))
        git(repo, "add", "f")
        git(repo, "commit", "-qm", f"c{i}")
        shas.append(git(repo, "rev-parse", "--short=7", "HEAD"))
    return repo, shas


def test_first_live_is_bounded_by_the_runs_own_log_lines(chain):
    """r3 #5 measured: (t, t] was reported as an exact install time."""
    repo, (c0, c1, c2) = chain
    got = first_live([rec(1, pre=c0, dep=c0), rec(3, pre=c0, dep=c2)], c1, repo=repo)
    assert (got["not_live_before"], got["live_by"]) == ("2026-07-03T00:00:00Z", "2026-07-03T00:00:02Z")
    assert got["live_by_kind"] == "deployed" and got["sha"] == c2
    assert first_live([rec(1, pre=c0, dep=c0)], "0000000", repo=repo) is None


def test_first_live_is_bounded_across_an_expired_run(chain):
    repo, (c0, c1, c2) = chain
    got = first_live([rec(1, pre=c0, dep=c0), rec(2, log="unavailable_expired", canary="absent"),
                      rec(3, pre=c2, dep=c2)], c1, repo=repo)
    assert (got["not_live_before"], got["live_by"], got["live_by_kind"]) == (
        "2026-07-01T00:00:02Z", "2026-07-03T00:00:00Z", "pre_deploy")


def _typed(num, created, pre_at, post_at, pre, post):
    return dict(run_id=num, created_at=created, log="retained", pre_sha=pre, pre_at=pre_at,
                deployed_sha=post, deployed_at=post_at, canary="passed", rollback="none", anomalies=[])


def test_observation_points_follow_their_own_timestamps(monkeypatch):
    """Codex phase-1 r4 #9: an older-created run that EXECUTED later was ordered by creation time."""
    from scripts.audit.incident_register import deploy_runs as d
    runs = [_typed(1, "2026-01-01T10:00:00Z", "2026-01-01T14:00:00Z", "2026-01-01T14:01:00Z", "bbbbbbb", "ccccccc"),
            _typed(2, "2026-01-01T11:00:00Z", "2026-01-01T12:00:00Z", "2026-01-01T12:01:00Z", "aaaaaaa", "bbbbbbb")]
    monkeypatch.setattr(d, "_is_ancestor", lambda fix, sha, repo: sha in ("bbbbbbb", "ccccccc"))
    got = d.first_live(runs, "bbbbbbb", repo=".")
    assert (got["not_live_before"], got["live_by"]) == ("2026-01-01T12:00:00Z", "2026-01-01T12:01:00Z")


def test_disagreeing_observations_at_the_same_instant_are_refused():
    from scripts.audit.incident_register import deploy_runs as d
    runs = [_typed(1, "2026-01-01T10:00:00Z", "2026-01-01T12:00:00Z", "2026-01-01T12:01:00Z", "aaaaaaa", "bbbbbbb"),
            _typed(2, "2026-01-01T11:00:00Z", "2026-01-01T12:01:00Z", "2026-01-01T12:02:00Z", "ccccccc", "ddddddd")]
    with pytest.raises(ValueError, match="disagree"):
        d.observations(runs)


def test_the_rollback_observation_is_the_sha_the_log_names():
    """Codex phase-1 r4 #9: 'rolled back cleanly to C' after 'rolling back to A' produced an invented A point."""
    from scripts.audit.incident_register import deploy_runs as d
    got = extract("\n".join([line(1, "Pre-deploy SHA: aaaaaaa"), line(2, "Deployed bbbbbbb"),
                             line(3, "CANARY FAILED — rolling back to aaaaaaa"), line(4, "Rolled back cleanly to ccccccc")]))
    assert got["rolled_back_sha"] == "ccccccc" and "rolled_back_sha_mismatch" in got["anomalies"]
    rec = dict(run_id=1, created_at="2026-09-22T16:28:00Z", log="retained", **got)
    assert d.observations([rec])[-1]["sha"] == "ccccccc"
