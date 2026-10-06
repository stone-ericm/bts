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


SCOPE = "historical count build"


def review(sha, verdict="**SIGN.**"):
    return f"# Review\n\n## Verdict\n\n{verdict} Reviewed `{sha}`.\n\n## Findings\n"


def x34_row(report_path, report_text, ref, scope=SCOPE):
    sha = hashlib.sha256(report_text.encode()).hexdigest()
    return f"| X-34 | rank 3 {scope}; review `{report_path}` sha256 `{sha[:16]}`; reviewed `{ref}` | ... |\n"


@pytest.fixture
def repo(tmp_path):
    """R reviewed; S archives the SIGN review naming R; X publishes X-34 citing it; Y sets the exposure commit."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    R = commit(r, {"pkg/a.py": "x = 1\n", "pkg/admission.json": "{}", "reg.md": "| X-33 | y |\n"}, "reviewed")
    rep = review(R)
    commit(r, {"docs/review-r1.md": rep}, "archive the review")
    adm = {"reviewed_commit": R, "review_report": "docs/review-r1.md"}
    X = commit(r, {"reg.md": "| X-33 | y |\n" + x34_row("docs/review-r1.md", rep, R), "pkg/admission.json": json.dumps(adm)},
               "publish X-34")
    adm = {**adm, "exposure_commit": X}
    commit(r, {"pkg/admission.json": json.dumps(adm)}, "admission")
    return r, adm


def check(r, adm):
    return A.admission_check(r, adm, closure=("pkg",), admission_rel="pkg/admission.json", register_rel="reg.md",
                             exposure_row="X-34", scope_phrase=SCOPE)[1]


def test_admission_passes_with_a_recorded_sign_and_metadata_only_edits(repo):
    r, adm = repo
    assert check(r, adm) == []


def test_admission_refuses_unset_fields(repo):
    r, adm = repo
    for k in ("reviewed_commit", "exposure_commit", "review_report"):
        assert check(r, {**adm, k: None}), k


def test_only_an_exact_sign_verdict_counts():
    """r3b F2: 'SIGN OFF REFUSED' and 'SIGN WITH EDITS pending' were accepted by a prefix match."""
    assert A.review_verdict(review("a" * 40)) == "SIGN"
    assert A.review_verdict(review("a" * 40, "**SIGN WITH EDITS.**")) == "SIGN WITH EDITS"
    for v in ("**SIGN OFF REFUSED.**", "**SIGN WITH EDITS pending application.**", "**SIGNATURE PENDING**",
              "**BLOCK.**", "SIGN."):
        assert A.review_verdict(review("a" * 40, v)) is None, v
    assert A.review_verdict("no verdict section") is None
    sha = "a" * 40
    assert any("does not name" in x for x in A._review_signs(review("0" * 40), sha))


def test_the_review_report_must_exist_at_the_exposure_commit(repo):
    r, adm = repo
    late = commit(r, {"docs/late.md": review(adm["reviewed_commit"])}, "a report archived after the exposure commit")
    assert any("does not exist at the exposure commit" in x for x in check(r, {**adm, "review_report": "docs/late.md"}))
    assert late


def test_the_exposure_row_must_bind_the_review_and_its_scope(tmp_path):
    """r3b F2: an unrelated X-34 row ('no historical fitting approved') was admitted."""
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    R = commit(r, {"pkg/a.py": "x = 1\n", "pkg/admission.json": "{}", "reg.md": "| X-33 | y |\n"}, "reviewed")
    rep = review(R)
    commit(r, {"docs/review-r1.md": rep}, "archive")
    for row in ("| X-34 | unrelated diagnostic; no historical fitting approved | withheld | Codex |\n",
                x34_row("docs/review-r1.md", rep, R, scope="diagnostic only"),
                x34_row("docs/review-r1.md", rep + "tampered", R)):
        X = commit(r, {"reg.md": "| X-33 | y |\n" + row}, "row")
        adm = {"reviewed_commit": R, "review_report": "docs/review-r1.md", "exposure_commit": X}
        assert any("does not cite" in x for x in check(r, adm)), row
        commit(r, {"reg.md": "| X-33 | y |\n"}, "reset")


def test_a_replacement_commit_with_a_new_report_is_refused(repo):
    """r3b F2: a new commit that changes code and a fresh report naming it, published before X, are refused unless
    the row cites that exact report and commit; here the old X-34 still cites R and the old report."""
    r, adm = repo
    R2 = commit(r, {"pkg/a.py": "x = 2\n"}, "replacement code")
    commit(r, {"docs/review-r2.md": review(R2)}, "a new report")
    assert check(r, {**adm, "reviewed_commit": R2, "review_report": "docs/review-r2.md"})


def test_sign_with_edits_needs_the_edits_commit(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    R = commit(r, {"pkg/a.py": "x = 1\n", "pkg/admission.json": "{}", "reg.md": "| X-33 | y |\n"}, "reviewed")
    rep = review(R, "**SIGN WITH EDITS.**")
    commit(r, {"docs/review-r1.md": rep}, "archive")
    E = commit(r, {"pkg/a.py": "x = 1  # edit applied verbatim\n"}, "apply the edits")
    X = commit(r, {"reg.md": "| X-33 | y |\n" + x34_row("docs/review-r1.md", rep, E)}, "publish")
    adm = {"reviewed_commit": R, "review_report": "docs/review-r1.md", "exposure_commit": X}
    assert any("edits_commit" in x for x in check(r, adm))
    assert check(r, {**adm, "edits_commit": E}) == []                 # the closure is compared with E, not R


def test_a_commit_that_changes_code_cannot_declare_itself_reviewed(repo):
    r, adm = repo
    X2 = commit(r, {"reg.md": "| X-33 | y |\n| X-34 | edited | ... |\n", "pkg/a.py": "x = 2\n"}, "row edit and code")
    assert check(r, {**adm, "reviewed_commit": X2, "exposure_commit": X2})


def test_executable_changes_untracked_sources_and_row_edits_refuse(repo):
    r, adm = repo
    (r / "pkg" / "shadow.py").write_text("y = 1\n")
    assert any("untracked" in x for x in check(r, adm))
    (r / "pkg" / "shadow.py").unlink()
    commit(r, {"pkg/a.py": "x = 3\n"}, "a later code change")
    assert any("changed since the reviewed commit" in x for x in check(r, adm))


def test_foreign_imports(monkeypatch, tmp_path):
    import sys
    import types
    m = types.ModuleType("bts.shadowed")
    m.__file__ = "/elsewhere/bts/shadowed.py"
    monkeypatch.setitem(sys.modules, "bts.shadowed", m)
    assert "bts.shadowed: /elsewhere/bts/shadowed.py" in A.foreign_imports(A.REPO)


def inv_row(run, claim_sha, correction, report="docs/fix-review.md", report_sha="0" * 16, *, verb="INVALIDATE",
            source="Eric 2026-10-06"):
    return (f"| C1-invalidate-{run} | x | **RULED 2026-10-06: {verb} `{run}` claim `{claim_sha[:16]}`; "
            f"correction `{correction}` reviewed `{report}` `{report_sha[:16]}`** | {source} |\n")


def test_a_claimed_run_is_released_only_by_eric_s_exact_invalidation_with_a_reviewed_correction(tmp_path):
    """4b r3 B1 / r3b F3: the ruling names this run and claim, a correction in HEAD's history, and a SIGN review of
    that correction (at HEAD) bound by its sha prefix; the source token is exactly Eric."""
    root = tmp_path / "runs"
    d = A.make_run_dir(root, "abc1234-20261006T000000Z")
    A.write_claim(d, "f" * 40)
    claim = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
    head = git(A.REPO, "rev-parse", "HEAD")
    report = "docs/audit/2026-10-05-c1-r4b-code-codex-r3.md"          # a real report at HEAD: BLOCK, not a SIGN
    rep_sha = hashlib.sha256((A.REPO / report).read_bytes()).hexdigest()   # committed and unchanged at HEAD
    (root / f"INVALIDATION_{d.name}.json").write_text(json.dumps({"run": d.name, "claim_sha256": claim}))
    assert any("does not SIGN" in x for x in A.claimed_runs(root, inv_row(d.name, claim, head, report, rep_sha)))
    for label, reg in {"none": "",
                       "denial": inv_row(d.name, claim, head, report, rep_sha, verb="DO NOT INVALIDATE"),
                       "Erica": inv_row(d.name, claim, head, report, rep_sha, source="Erica 2026-10-06"),
                       "other claim": inv_row(d.name, "e" * 64, head, report, rep_sha),
                       "correction not in history": inv_row(d.name, claim, "0" * 40, report, rep_sha),
                       "report sha mismatch": inv_row(d.name, claim, head, report, "0" * 16),
                       "unrelated ruling": f"| C1-invalidate-{d.name} | x | **RULED 2026-10-06: A** | Eric |\n"}.items():
        assert A.claimed_runs(root, reg), label
    (root / f"INVALIDATION_{d.name}.json").write_text("{}")
    assert A.claimed_runs(root, inv_row(d.name, claim, head, report, rep_sha))


def test_make_run_dir_creates_and_refuses_reuse(tmp_path):
    d = A.make_run_dir(tmp_path / "runs", "r1")
    assert d.is_dir()
    with pytest.raises(FileExistsError):
        A.make_run_dir(tmp_path / "runs", "r1")


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
