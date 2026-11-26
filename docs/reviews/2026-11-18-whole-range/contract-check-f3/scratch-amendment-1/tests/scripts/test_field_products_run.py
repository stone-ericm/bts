"""W2.1/W2.2 driver (code review r1 F1, F6): the publication-bound X-24/X-25 gate, the pre-outcome source freeze,
fail-closed prerequisites, and end-to-end runs over synthetic inputs in the real on-disk formats."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime

import pandas as pd
import pytest

from scripts.audit.field_products import run
from tests.scripts import test_field_products_builders as B

REG = run.REGISTER_REL


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _repo(tmp_path):
    """A throwaway local repo: register without rows, then commit A publishes X-24, commit B publishes X-25."""
    repo = tmp_path / "repo"
    (repo / "docs/audit").mkdir(parents=True)
    _git(tmp_path, "init", "-q", str(repo))
    (repo / REG).write_text("| X-23 | old |\n")
    (repo / "src.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / REG).write_text("| X-23 | old |\n| X-24 | board |\n")
    _git(repo, "commit", "-q", "-am", "X-24")
    a = _git(repo, "rev-parse", "HEAD")
    (repo / REG).write_text("| X-23 | old |\n| X-24 | board |\n| X-25 | cohort |\n")
    _git(repo, "commit", "-q", "-am", "X-25")
    return repo, a, _git(repo, "rev-parse", "HEAD")


def _set(monkeypatch, x24, x25):
    monkeypatch.setattr(run, "X24_COMMIT", x24)
    monkeypatch.setattr(run, "X25_COMMIT", x25)


def test_gate_passes_only_for_rows_published_by_their_named_commits_with_clean_sources(tmp_path, monkeypatch):
    repo, a, b = _repo(tmp_path)
    _set(monkeypatch, a, b)
    g = run.gate(repo=repo, sources=["src.py"])
    assert g["head"] == b and g["rows"]["X-24"]["commit"] == a and g["rows"]["X-25"]["commit"] == b


def test_gate_refuses_while_the_commits_are_unset(tmp_path, monkeypatch):
    repo, a, b = _repo(tmp_path)
    _set(monkeypatch, None, b)
    with pytest.raises(SystemExit, match="X-24 gate: .*unset"):
        run.gate(repo=repo, sources=["src.py"])


def test_gate_refuses_a_row_absent_at_its_named_commit_or_not_published_by_it(tmp_path, monkeypatch):
    repo, a, b = _repo(tmp_path)
    _set(monkeypatch, a, a)                                          # X-25 is not in commit A
    with pytest.raises(SystemExit, match="X-25 gate: .*no X-25 row"):
        run.gate(repo=repo, sources=["src.py"])
    _set(monkeypatch, b, b)                                          # X-24 already existed in B's parent
    with pytest.raises(SystemExit, match="X-24 gate: .*did not publish"):
        run.gate(repo=repo, sources=["src.py"])


def test_review_probe_uncommitted_register_rows_do_not_pass_the_gate(tmp_path, monkeypatch):
    repo, a, b = _repo(tmp_path)
    (repo / REG).write_text("| X-23 | old |\n| X-24 | board |\n| X-25 | cohort |\n| X-99 | uncommitted |\n")
    _set(monkeypatch, a, b)
    with pytest.raises(SystemExit, match="working register differs"):
        run.gate(repo=repo, sources=["src.py"])


def test_gate_refuses_a_row_changed_since_publication_and_a_non_ancestor(tmp_path, monkeypatch):
    repo, a, b = _repo(tmp_path)
    _set(monkeypatch, a, "0" * 40)
    with pytest.raises(SystemExit, match="X-25 gate: .*not an ancestor"):
        run.gate(repo=repo, sources=["src.py"])
    (repo / REG).write_text("| X-23 | old |\n| X-24 | board, amended |\n| X-25 | cohort |\n")
    _git(repo, "commit", "-q", "-am", "amend")
    _set(monkeypatch, a, b)
    with pytest.raises(SystemExit, match="X-24 gate: .*changed since"):
        run.gate(repo=repo, sources=["src.py"])


def test_gate_refuses_dirty_or_untracked_executing_sources(tmp_path, monkeypatch):
    repo, a, b = _repo(tmp_path)
    _set(monkeypatch, a, b)
    (repo / "src.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="executing sources"):
        run.gate(repo=repo, sources=["src.py"])
    _git(repo, "checkout", "--", "src.py")
    (repo / "new.py").write_text("y = 1\n")
    with pytest.raises(SystemExit, match="executing sources"):
        run.gate(repo=repo, sources=["src.py", "new.py"])


def test_executing_sources_cover_the_package_and_its_repo_dependencies():
    srcs = run.executing_sources()
    assert "scripts/audit/field_products/run.py" in srcs and "scripts/final_leaderboard_grab.py" in srcs
    assert "scripts/audit/mlb_benchmark/metrics.py" in srcs and "src/bts/leaderboard/scraper.py" in srcs
    assert REG in srcs and not any(s.startswith("tests/") for s in srcs)


def test_main_refuses_before_reading_anything_when_the_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "X24_COMMIT", None)
    with pytest.raises(SystemExit):
        run.main(["--data-root", str(tmp_path / "nope"), "--ledger-dir", str(tmp_path / "nope"),
                  "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


EARLY = [(1001, "alpha"), (2001, "b one"), (2002, "jordan"), (2003, "jordan"), (2004, "dee"), (2005, "eve")]
CAP1, CAP2 = datetime(2026, 5, 2, 12), datetime(2026, 7, 4, 12)


def _inputs(tmp_path):
    hits = B.profile([B.pred(1000, 1, [(1, 5, 3, "hit")]), B.pred(1001, 2, [(1, 5, 4, "hit"), (2, 5, 6, "hit")]),
                      B.pred(1002, 0, [(1, 5, 7, "not_hit")])])
    grab = B.make_grab(tmp_path, board_pages=B.pages(B.board_rows(420)), early=EARLY, cohort_a=2, cohort_b=3,
                       profiles={1000: (200, hits), 1001: (200, hits), 2001: (200, hits),
                                 2002: (200, B.profile([]))})
    up = tmp_path / "data" / "leaderboard" / "user_picks"
    up.mkdir(parents=True)
    d1, d2, d3 = date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)
    p = B.daily_pick
    B.write_daily(up, "alpha", [[p(10, 1, cap=CAP1, pick_date=d1, streak=1)],
                                [p(10, 1, cap=CAP2, pick_date=d1, streak=1),
                                 p(11, 1, cap=CAP2, pick_date=d2, streak=3, player=21),
                                 p(11, 2, cap=CAP2, pick_date=d2, streak=3, player=22)]])
    B.write_daily(up, "b one", [[p(10, 1, cap=CAP1, pick_date=d1, result="not_hit", streak=0)]])
    B.write_daily(up, "b_one", [[p(10, 1, cap=CAP2, pick_date=d1, result="not_hit", streak=0),
                                 p(12, 1, cap=CAP2, pick_date=d3, streak=1)]])
    B.write_daily(up, "jordan", [[p(10, 1, cap=CAP2, pick_date=d1)]])
    B.write_daily(up, "dee", [[p(10, 1, cap=CAP1, pick_date=d1, player=20)], [p(10, 1, cap=CAP2, pick_date=d1,
                                                                                player=30)]])
    B.write_daily(up, "stranger", [[p(10, 1, cap=CAP2, pick_date=d1)]])
    sel = B.ledger_selection
    rows = [sel("s1", "2026-05-01", "primary", "hit", round_id=1), sel("s2", "2026-05-02", "primary", "hit", round_id=2),
            sel("s3", "2026-05-02", "double_down", "not_hit", round_id=2, unit=2),
            sel("s4", "2026-05-04", "primary", "hit", round_id=4)]
    contest = [{"selection_id": r["selection_id"], "match": "evidenced", "slot_result": r["contest_slot_grade_raw"],
                "round_id": r["round_id"], "unit_id": r["unit_id"]} for r in rows]
    led = B.make_ledger(tmp_path / "validation", rows, contest)
    return grab, led


def _argv(tmp_path, led, *extra):
    return ["--data-root", str(tmp_path / "data"), "--ledger-dir", str(led), "--out", str(tmp_path / "out"),
            "--early-manifest", str(tmp_path / "early.json"), *extra]


def _patch_gate(monkeypatch, head="f" * 40):
    monkeypatch.setattr(run, "gate", lambda: {"head": head, "rows": {}, "sources": []})


def test_main_runs_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch):
    _patch_gate(monkeypatch)
    grab, led = _inputs(tmp_path)
    assert run.main(_argv(tmp_path, led, "--our-user-id", "1150", "--n-resamples", "60")) == 0
    out = next((tmp_path / "out").glob("fffffff-*"))
    res = json.loads((out / "results.json").read_text())
    assert res["w21"]["census_gate"]["census"] is True
    sb = res["w21"]["season_best"]
    assert sb["basis"] == "census" and sb["own"]["stored_rank"] == 151
    assert sb["thresholds"]["ge_20"] == {"known": 110, "upper_bound": 110}
    assert res["w21"]["case_series_A"]["n_A"] == 2 and res["w21"]["case_series_A"]["survivor_selected"] is True
    assert res["w21"]["case_series_A"]["raw_witness"] == {"verified": 2}
    w22 = res["w22"]
    assert w22["manifest_E"]["n"] == 6 and w22["manifest_E"]["checks"]["manifest_sha_matches_grab"] is True
    assert w22["identity"]["members"] == {"bound": 3, "quarantined_manifest_collision": 2, "no_daily_file": 1}
    assert w22["identity"]["ownership_quarantined"] == [2004]
    assert w22["identity"]["attribution_basis"] == "stable_5_01_username_unwitnessed"
    assert w22["identity"]["per_batch_identity_established"] is False
    assert w22["identity"]["per_batch_identity"] == 'Batch-specific identity for all May 1-July 3 daily observations has not been established. Daily pick parquet omits user ID; the normal daily writer provides no general durable batch-ID receipt. Particular retained backfill logs or later ID-bearing snapshots/stats may supply partial evidence, but coverage has not been verified. Primary attribution uses the unwitnessed stable May 1 name assumption.'
    assert res["limits"][0].startswith(w22["identity"]["per_batch_identity"])
    assert w22["availability"]["daily_history"] == {"available": 2, "quarantined_manifest_collision": 2,
                                                    "quarantined_ownership_conflict": 1, "no_daily_file": 1}
    prim = w22["primary"]
    a, s = prim["tables"]["all_observed_dates"], prim["tables"]["shared_dates"]
    # E: alpha 5/01 hit + 5/02 DD hit,hit ; b one 5/01 miss + 5/03 hit -> 5 slots, 4 hits; ours 4 slots, 3 hits
    assert a["E"] == {"dates_with_slots": 3, "users": 2, "rounds": 4, "round_id_missing": 0,
                      "user_dates_with_multiple_rounds": 0, "attribution_basis": "stable_5_01_username_unwitnessed",
                      "slots": 5, "hits": 4, "ratio": 0.8}
    assert a["ours"]["slots"] == 4 and a["ours"]["rounds"] == 3 and a["dates"] == 4
    assert s["date_list"] == ["2026-05-01", "2026-05-02"] and s["E"]["slots"] == 4 and s["ours"]["slots"] == 3
    assert s["intervals"]["diff_ours_minus_E"]["n_resamples"] == 60
    assert prim["ours"]["included_by_slot"] == {"primary": 3, "double_down": 1}
    assert prim["ledger"]["receipt"]["run"] == run.LG.ACCEPTED_RUN
    assert prim["ledger"]["acceptance_byte_identity_established"] is False
    assert prim["E_exclusions"] == {"window_slots": 5, "usable_graded": 5, "excluded_by_label": {},
                                    "excluded_by_status": {}}
    assert a["coverage"]["E_members"] == 6 and a["coverage"]["window_calendar_dates"] == 64
    av = pd.read_parquet(out / "w22_availability.parquet").set_index("user_id")
    assert av.loc[1001, "allocation"] == "E_in_A" and av.loc[1001, "window_graded_slots"] == 3
    # R2-1: daily rounds have no completeness witness, so no positive streak bound is claimed for E
    assert av.loc[1001, "window_streak_status"] == "no_complete_rounds"
    assert pd.isna(av.loc[1001, "window_longest_lower_bound"]) and av.loc[1001, "window_longest_exact"] is None
    ws = prim["window_streaks"]
    assert ws["status"] == {"no_complete_rounds": 2}
    assert ws["rule"].startswith("qualified complete all-hit rounds only; daily positive streak bounds unavailable "
                                 "without a completeness witness")
    ext = w22["extension"]
    assert ext["coverage"]["usable"] == 2 and ext["coverage"]["budget_omissions"] == 2
    assert ext["pooled"] == {"users": 2, "dates": 3, "rounds": 6, "slots": 8, "hits": 6, "ratio": 0.75}
    man = json.loads((out / "manifest.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    assert man["sources"] == freeze["sources"] and man["agrees_with_freeze"] is True
    files = man["sources"]["files"]
    assert files["daily/alpha.parquet"]["sha256"] == hashlib.sha256((tmp_path / "data/leaderboard/user_picks/"
                                                                     "alpha.parquet").read_bytes()).hexdigest()
    assert "ledger/ACCEPTED.json" in files and not any("jordan" in k for k in files)   # quarantined: never read
    assert man["code"]["head"] == "f" * 40 and "scripts/audit/field_products/census.py" in man["code"]["files"]
    assert (out / "w22_daily_slots.parquet").exists() and (out / "w21_distribution.parquet").exists()


def test_review_probe_a_source_changed_after_the_freeze_refuses_the_run(tmp_path, monkeypatch):
    """F1 probe: after the freeze, the first outcome step rewrites alpha.parquet's results to not_hit. The analysis
    reads the frozen bytes, the end-of-run verification sees the change, and nothing is published."""
    _patch_gate(monkeypatch)
    grab, led = _inputs(tmp_path)
    alpha = tmp_path / "data/leaderboard/user_picks/alpha.parquet"
    real = run.C.load_board_receipts

    def tamper(*a, **k):
        df = pd.read_parquet(alpha)
        df["result"] = "not_hit"
        df.to_parquet(alpha)
        return real(*a, **k)
    monkeypatch.setattr(run.C, "load_board_receipts", tamper)
    with pytest.raises(SystemExit, match="daily/alpha.parquet: content changed"):
        run.main(_argv(tmp_path, led, "--n-resamples", "20"))
    out = next((tmp_path / "out").glob("fffffff-*"))
    assert (out / "freeze.json").exists() and not (out / "results.json").exists()


def test_a_listing_change_during_the_run_refuses_it(tmp_path, monkeypatch):
    _patch_gate(monkeypatch)
    grab, led = _inputs(tmp_path)
    real = run.C.load_board_receipts

    def add_file(*a, **k):
        (tmp_path / "data/leaderboard/user_picks/late.parquet").write_bytes(b"x")
        return real(*a, **k)
    monkeypatch.setattr(run.C, "load_board_receipts", add_file)
    with pytest.raises(SystemExit, match="daily_listing: listing changed"):
        run.main(_argv(tmp_path, led, "--n-resamples", "20"))


@pytest.mark.parametrize("breakage,message", [
    ("zero_sha", "manifest_sha_matches_grab"), ("move_b", "recomputed_allocation_matches"),
    ("copy", "grab_input_copy_matches"), ("identity_artifact", "identity_artifact_hash"),
    ("receipt", "ACCEPTED.json")])
def test_review_probe_failed_prerequisites_refuse_before_any_outcome(tmp_path, monkeypatch, breakage, message):
    """F6 probe: a zeroed manifest hash, a moved B allocation, a mismatched manifest copy, an identity.json not
    matching its artifact hash, or an arbitrary ACCEPTED.json each stop the run before any outcome is parsed."""
    _patch_gate(monkeypatch)
    grab, led = _inputs(tmp_path)
    cj = json.loads((grab / "cohort.json").read_text())
    if breakage == "zero_sha":
        cj["early_cohort_sha256"] = "0" * 64
    elif breakage == "move_b":
        cj["B"] = [2001, 2003, 2005]
    elif breakage == "copy":
        (grab / "inputs" / "early_cohort.json").write_text("{}")
    elif breakage == "identity_artifact":
        (grab / "identity.json").write_text((grab / "identity.json").read_text() + " ")
    elif breakage == "receipt":
        (led / "ACCEPTED.json").write_text("this is not an acceptance receipt")
    if breakage in ("zero_sha", "move_b"):
        (grab / "cohort.json").write_text(json.dumps(cj))
        status = json.loads((grab / "status.json").read_text())
        status["artifacts"]["cohort"]["sha256"] = hashlib.sha256((grab / "cohort.json").read_bytes()).hexdigest()
        (grab / "status.json").write_text(json.dumps(status))
    monkeypatch.setattr(run.C, "load_board_receipts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read")))
    with pytest.raises(SystemExit, match=message):
        run.main(_argv(tmp_path, led, "--n-resamples", "20"))
    assert not list((tmp_path / "out").glob("*/results.json"))


def test_freeze_lists_every_source_hash_before_any_outcome_bearing_read(tmp_path, monkeypatch):
    _patch_gate(monkeypatch, head="e" * 40)
    grab, led = _inputs(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("outcome read attempted")
    monkeypatch.setattr(run.C, "load_board_receipts", boom)
    monkeypatch.setattr(run.P, "read_observations", boom)
    with pytest.raises(RuntimeError, match="outcome read attempted"):
        run.main(_argv(tmp_path, led))
    out = next((tmp_path / "out").glob("eeeeeee-*"))
    freeze = json.loads((out / "freeze.json").read_text())
    files = freeze["sources"]["files"]
    assert {"daily/alpha.parquet", "final_grab/status.json", "final_grab/identity.json", "ledger/ACCEPTED.json",
            "final_grab/raw/board/page_001.json.gz", "final_grab/user_picks/2001.parquet",
            "final_grab/raw/profiles/2001.json.gz"} <= set(files)
    assert freeze["binding_counts"]["members"]["bound"] == 3 and not (out / "results.json").exists()


def test_nested_daily_files_are_counted_and_a_pre_sanitizer_semicolon_name_is_read_whole(tmp_path, monkeypatch):
    _patch_gate(monkeypatch, head="d" * 40)
    grab, led = _inputs(tmp_path)
    early = json.loads((tmp_path / "early.json").read_text())
    up = tmp_path / "data" / "leaderboard" / "user_picks"
    nested = up / "a"
    nested.mkdir()
    B.write_daily(nested, "b", [[B.daily_pick(10, 1, cap=CAP1, pick_date=date(2026, 5, 1))]])
    B.write_daily(up, "e;ve", [[B.daily_pick(10, 1, cap=CAP1, pick_date=date(2026, 5, 1))]])
    early["users"][5]["usernames_2026_05_01"] = ["e;ve"]                       # member 2005
    (tmp_path / "early.json").write_text(json.dumps(early))
    for f in (grab / "inputs" / "early_cohort.json",):
        f.write_bytes((tmp_path / "early.json").read_bytes())
    cj = json.loads((grab / "cohort.json").read_text())
    cj["early_cohort_sha256"] = hashlib.sha256((tmp_path / "early.json").read_bytes()).hexdigest()
    (grab / "cohort.json").write_text(json.dumps(cj))
    status = json.loads((grab / "status.json").read_text())
    status["artifacts"]["cohort"]["sha256"] = hashlib.sha256((grab / "cohort.json").read_bytes()).hexdigest()
    (grab / "status.json").write_text(json.dumps(status))
    assert run.main(_argv(tmp_path, led, "--n-resamples", "20")) == 0
    out = next((tmp_path / "out").glob("ddddddd-*"))
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["daily_nested_files_ignored"] == ["a/b.parquet"]
    av = pd.read_parquet(out / "w22_availability.parquet").set_index("user_id")
    assert av.loc[2005, "daily_history"] == "available" and av.loc[2005, "window_graded_slots"] == 1
    res = json.loads((out / "results.json").read_text())
    assert res["w21"]["season_best"]["own"]["matched_by_user_id"] is False      # no --our-user-id given


def test_outcomes_are_parsed_from_the_frozen_bytes_not_the_disk(tmp_path, monkeypatch):
    """With the end-of-run check disabled, a post-freeze rewrite of alpha.parquet (all not_hit) still cannot reach
    the analysis: E's ratio stays the frozen 0.8 (the rewrite would make it 0.2)."""
    _patch_gate(monkeypatch)
    grab, led = _inputs(tmp_path)
    alpha = tmp_path / "data/leaderboard/user_picks/alpha.parquet"
    real = run.C.load_board_receipts

    def tamper(*a, **k):
        df = pd.read_parquet(alpha)
        df["result"] = "not_hit"
        df.to_parquet(alpha)
        return real(*a, **k)
    monkeypatch.setattr(run.C, "load_board_receipts", tamper)
    monkeypatch.setattr(run.Stage, "verify", lambda self: [])
    assert run.main(_argv(tmp_path, led, "--n-resamples", "20")) == 0
    res = json.loads(next((tmp_path / "out").glob("fffffff-*/results.json")).read_text())
    assert res["w22"]["primary"]["tables"]["all_observed_dates"]["E"]["ratio"] == 0.8


def test_review_probe_cohort_a_board_fields_come_from_qualified_raw_rows(tmp_path, monkeypatch):
    """R2-3: only the output parquet is altered (id 1000: rank 99, best 40). The gate fails its integrity checks,
    and Cohort A still reports id 1000's qualified raw row: rank 1, best 30 — never 99/40."""
    _patch_gate(monkeypatch)
    grab, led = _inputs(tmp_path)
    status = json.loads((grab / "status.json").read_text())
    path = grab / status["artifacts"]["leaderboard_snapshot"]["path"]
    df = pd.read_parquet(path)
    df.loc[df["user_id"] == 1000, ["rank", "season_best_streak"]] = [99, 40]
    df.to_parquet(path)
    assert run.main(_argv(tmp_path, led, "--n-resamples", "20")) == 0
    out = next((tmp_path / "out").glob("fffffff-*"))
    res = json.loads((out / "results.json").read_text())
    assert {"board_output_hash", "board_rows_equal_raw"} <= set(res["w21"]["census_gate"]["failures"])
    a = pd.read_parquet(out / "w21_case_series_A.parquet").set_index("user_id")
    assert a.loc[1000, "board_rank"] == 1 and a.loc[1000, "board_season_best"] == 30
