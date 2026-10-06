"""The shared C1 admission gate (used by rank 3 and later candidates; the deferred 4b review's B1/B2 lessons)."""
import hashlib
import json
import subprocess

import pytest

from scripts.audit.c1 import admission as A


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, files, msg):
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


ROW = "| X-34 | the rank-3 count build | ... |\n"


def review(sha, verdict="**SIGN.**"):
    return f"# Review\n\n## Verdict\n\n{verdict} Reviewed `{sha}`.\n\n## Findings\n"


@pytest.fixture
def repo(tmp_path):
    """R reviewed; S records the review report (SIGN naming R); X publishes X-34 and the admission's first fields;
    Y sets the exposure commit."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    R = commit(r, {"pkg/a.py": "x = 1\n", "pkg/admission.json": "{}", "reg.md": "| X-33 | y |\n"}, "reviewed")
    commit(r, {"docs/review-r1.md": review(R)}, "archive the review")
    adm = {"reviewed_commit": R, "review_report": "docs/review-r1.md"}
    X = commit(r, {"reg.md": "| X-33 | y |\n" + ROW, "pkg/admission.json": json.dumps(adm)}, "publish X-34")
    adm = {**adm, "exposure_commit": X}
    commit(r, {"pkg/admission.json": json.dumps(adm)}, "admission")
    return r, adm


def check(r, adm):
    return A.admission_check(r, adm, closure=("pkg",), admission_rel="pkg/admission.json", register_rel="reg.md",
                             exposure_row="X-34")[1]


def test_admission_passes_with_a_recorded_sign_and_metadata_only_edits(repo):
    r, adm = repo
    assert check(r, adm) == []


def test_admission_refuses_unset_fields(repo):
    r, adm = repo
    for k in ("reviewed_commit", "exposure_commit", "review_report"):
        assert check(r, {**adm, k: None}), k


def test_the_review_report_must_record_a_sign_naming_the_reviewed_commit(repo):
    """4b r3 B2: a self-declared reviewed commit is not a review anchor."""
    r, adm = repo
    sha = adm["reviewed_commit"]
    assert A._review_signs(review(sha), sha) == []
    assert A._review_signs(review(sha, "**SIGN WITH EDITS.**"), sha) == []
    assert any("not a SIGN" in x for x in A._review_signs(review(sha, "**BLOCK.**"), sha))
    assert any("not a SIGN" in x for x in A._review_signs(review(sha, "**SIGNATURE PENDING**"), sha))
    assert any("does not name" in x for x in A._review_signs(review("0" * 40), sha))
    assert any("Verdict" in x for x in A._review_signs("no verdict here " + sha, sha))
    late = commit(r, {"docs/late.md": review(sha)}, "a report archived after the exposure commit")
    assert any("does not exist at the exposure commit" in x for x in check(r, {**adm, "review_report": "docs/late.md"}))
    assert late


def test_a_commit_that_changes_code_cannot_declare_itself_reviewed(repo):
    """4b r3 B2's counterexample: X publishes the row and changes code, then names itself reviewed."""
    r, adm = repo
    X2 = commit(r, {"reg.md": "| X-33 | y |\n" + ROW.replace("count build", "count build v2"), "pkg/a.py": "x = 2\n"},
                "row edit and a code change")
    assert check(r, {**adm, "reviewed_commit": X2, "exposure_commit": X2})


def test_executable_changes_untracked_sources_and_row_edits_refuse(repo):
    r, adm = repo
    (r / "pkg" / "shadow.py").write_text("y = 1\n")
    assert any("untracked" in x for x in check(r, adm))
    (r / "pkg" / "shadow.py").unlink()
    commit(r, {"reg.md": "| X-33 | y |\n| X-34 | edited | ... |\n"}, "edit")
    assert any("changed after" in x for x in check(r, adm))


def test_foreign_imports(monkeypatch, tmp_path):
    import sys
    import types
    m = types.ModuleType("bts.shadowed")
    m.__file__ = "/elsewhere/bts/shadowed.py"
    monkeypatch.setitem(sys.modules, "bts.shadowed", m)
    assert "bts.shadowed: /elsewhere/bts/shadowed.py" in A.foreign_imports(A.REPO)


def inv_row(run, claim_sha, correction, *, verb="INVALIDATE", source="Eric 2026-10-06"):
    return (f"| C1-invalidate-{run} | x | **RULED 2026-10-06: {verb} `{run}` claim `{claim_sha[:16]}`; "
            f"correction `{correction}`** | {source} |\n")


def test_a_claimed_run_is_released_only_by_eric_s_exact_invalidation(tmp_path):
    """4b r3 B1: an unrelated RULED row or a self-declared approver released a claim; now the ruling must name
    this run, this claim and a correction commit in HEAD's history, from Eric."""
    root = tmp_path / "runs"
    d = root / "abc1234-20261006T000000Z"
    d.mkdir(parents=True)
    A.write_claim(d, "f" * 40)
    claim = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
    head = git(A.REPO, "rev-parse", "HEAD")
    (root / f"INVALIDATION_{d.name}.json").write_text(json.dumps({"run": d.name, "claim_sha256": claim}))
    good = inv_row(d.name, claim, head)
    assert A.claimed_runs(root, good) == []
    for label, reg in {"none": "", "denial": inv_row(d.name, claim, head, verb="DO NOT INVALIDATE"),
                       "wrong source": inv_row(d.name, claim, head, source="Codex"),
                       "other claim": inv_row(d.name, "e" * 64, head),
                       "correction not in history": inv_row(d.name, claim, "0" * 40),
                       "unrelated ruling": f"| C1-invalidate-{d.name} | x | **RULED 2026-10-06: A** | Eric |\n"}.items():
        assert A.claimed_runs(root, reg), label
    (root / f"INVALIDATION_{d.name}.json").write_text("{}")
    assert A.claimed_runs(root, good)


def test_the_admission_lock_refuses_a_second_holder(tmp_path):
    with A.admission_lock(tmp_path):
        with pytest.raises(SystemExit, match="lock"):
            with A.admission_lock(tmp_path):
                pass


def test_read_once_hashes_and_returns_the_same_bytes(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    b, h = A.read_pinned(f, hashlib.sha256(b"abc").hexdigest())
    assert b == b"abc"
    with pytest.raises(A.ProvenanceError):
        A.read_pinned(f, "0" * 64)
