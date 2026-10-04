"""The accepted W1.1 ledger reader (code review r1 F6): the acceptance receipt is parsed and verified against the
predeclared fingerprint, the six named build outputs and the published build identity, and the parsed bytes are bound
to the build's own published counts — before any rate is computed."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit.field_products import ledger as LG
from tests.scripts import test_field_products_builders as B


def _build(tmp_path):
    rows = [B.ledger_selection("s1", "2026-05-01", "primary", "hit"),
            B.ledger_selection("s2", "2026-05-02", "double_down", "not_hit", match="inferred", unit=2),
            {"row_id": "day|2026-05-03", "row_kind": "skip_day", "date": "2026-05-03"}]
    contest = [{"selection_id": "s1", "match": "evidenced", "slot_result": "hit", "round_id": 1, "unit_id": 1},
               {"selection_id": "s2", "match": "inferred", "slot_result": "not_hit", "round_id": 1, "unit_id": 2}]
    return B.make_ledger(tmp_path, rows, contest)


def _load(d):
    disk = lambda p: p.read_bytes()
    info = LG.verify_receipt(d, sorted(p.name for p in d.iterdir()), disk)
    return LG.load_bound(d, disk, info)


def test_a_valid_accepted_build_is_read_and_bound(tmp_path):
    ledger, contest, info = _load(_build(tmp_path))
    assert len(ledger) == 3 and len(contest) == 2 and info["receipt"]["run"] == LG.ACCEPTED_RUN
    assert info["bound_counts"]["row_kinds"] == {"selection": 2, "skip_day": 1}


def test_review_probe_an_arbitrary_acceptance_file_is_refused(tmp_path):
    d = _build(tmp_path)
    (d / "ACCEPTED.json").write_text("this is not an acceptance receipt")
    with pytest.raises(SystemExit, match="ACCEPTED.json"):
        _load(d)


@pytest.mark.parametrize("field,value", [("rules_fingerprint", "0" * 64), ("run", "other-run"),
                                         ("files", ["season_2026_ledger.parquet"])])
def test_receipt_fields_must_match_the_published_build(tmp_path, field, value):
    d = _build(tmp_path)
    receipt = json.loads((d / "ACCEPTED.json").read_text())
    receipt[field] = value
    (d / "ACCEPTED.json").write_text(json.dumps(receipt))
    with pytest.raises(SystemExit, match=field):
        _load(d)


def test_build_identity_directory_and_listing_are_checked(tmp_path):
    d = _build(tmp_path)
    build = json.loads((d / "season_2026_ledger_build.json").read_text())
    build["code_sha"] = "f" * 40
    (d / "season_2026_ledger_build.json").write_text(json.dumps(build))
    with pytest.raises(SystemExit, match="code_sha"):
        _load(d)
    other = tmp_path / "x"
    d2 = _build(other)
    (d2 / "extra.txt").write_text("x")
    with pytest.raises(SystemExit, match="listing"):
        _load(d2)
    renamed = d2.rename(other / "some-other-build")
    (renamed / "extra.txt").unlink()
    with pytest.raises(SystemExit, match="directory"):
        _load(renamed)


def test_a_parquet_that_is_not_the_compilers_schema_is_refused(tmp_path):
    d = _build(tmp_path)
    pd.DataFrame([{"row_kind": "selection"}]).to_parquet(d / "season_2026_ledger.parquet")
    with pytest.raises(SystemExit, match="schema"):
        _load(d)


def test_parsed_rows_must_match_the_builds_published_counts(tmp_path):
    """Bytes that are not the accepted build's (here: a row dropped after acceptance) are refused before any rate."""
    import pyarrow.parquet as pq
    d = _build(tmp_path)
    t = pq.read_table(d / "season_2026_ledger.parquet")
    pq.write_table(t.slice(0, 2), d / "season_2026_ledger.parquet")
    with pytest.raises(SystemExit, match="row_kinds"):
        _load(d)
