"""C2 step 2a, design §2/§4/§5.4: the bts_slate_v3 envelope and the witness's transport to persistence."""
import json
import math

import pandas as pd
import pytest

from bts import orchestrator as O
from bts import slate as S

WITNESS = {"schema": "bts_serving_witness_v2", "model": {"source": "cache", "sha256": "a" * 64}, "errors": []}


def _preds(serving=WITNESS, projected=(False, True, False)):
    df = pd.DataFrame({"batter_id": [1, 2, 3], "game_pk": [10, 10, 11], "p_game_hit": [0.8, 0.75, 0.7],
                       "flags": ["", "PROJECTED lineup", ""], "projected": list(projected)})
    if serving is not ...:
        df.attrs["serving"] = serving
    return df


def _payload(path):
    return json.loads(path.read_text())


def test_the_envelope_is_v3_and_carries_the_witness(tmp_path):
    p = _payload(S.save_slate(_preds(), "2027-04-01", tmp_path, "local"))
    assert S.SCHEMA_VERSION == "bts_slate_v3" and p["schema_version"] == "bts_slate_v3"
    assert p["serving"] == WITNESS
    assert list(p) == ["schema_version", "date", "tier", "written_at", "n_rows", "rows", "serving"]


def test_posted_rows_persist_an_explicit_false(tmp_path):
    rows = _payload(S.save_slate(_preds(), "2027-04-01", tmp_path, "local"))["rows"]
    assert [r["projected"] for r in rows] == [False, True, False]


def test_no_witness_is_null(tmp_path):
    p = _payload(S.save_slate(_preds(serving=...), "2027-04-01", tmp_path, "mac"))
    assert p["serving"] is None and p["n_rows"] == 3


@pytest.mark.parametrize("bad", [{"x": float("nan")}, {"x": {1, 2}}, {"x": object()}, {"x": math.inf}])
def test_an_unserializable_witness_is_null_and_the_slate_still_writes(tmp_path, bad):
    p = _payload(S.save_slate(_preds(serving=bad), "2027-04-01", tmp_path, "local"))
    assert p["serving"] is None and p["n_rows"] == 3


def test_a_frame_whose_attrs_raise_is_swallowed_as_before(tmp_path):
    """pandas itself reads attrs when selecting the row columns, so a raising attrs getter fails the row extraction
    exactly as at f882411: no slate, nothing raised."""
    class NoAttrs(pd.DataFrame):
        @property
        def attrs(self):
            raise RuntimeError("synthetic attrs failure")

        @attrs.setter
        def attrs(self, value):
            pass
    frame = NoAttrs({"batter_id": [1], "game_pk": [10], "p_game_hit": [0.8], "flags": [""], "projected": [False]})
    assert S.save_slate(frame, "2027-04-01", tmp_path, "local") is None


def test_rows_are_unchanged_by_the_witness(tmp_path):
    a = _payload(S.save_slate(_preds(), "2027-04-01", tmp_path / "a", "local"))
    b = _payload(S.save_slate(_preds(serving=...), "2027-04-01", tmp_path / "b", "local"))
    assert a["rows"] == b["rows"]


# ---------------------------------------------------------------- run_and_pick's transport (§5.4)

def _config(tmp_path):
    return {"orchestrator": {"picks_dir": str(tmp_path)}, "tiers": [{"name": "local", "type": "local"}]}


@pytest.fixture
def selection(monkeypatch):
    import bts.contest_state as CS
    import bts.picks as PK
    import bts.strategy as ST
    seen = {}

    class State:
        streak, saver_available, allow_double, best_streak, best_status = 0, False, True, None, None
        source, status, contest_source_date = "test", "ok", None

    def fake_select(predictions, *a, **k):
        seen["attrs"] = dict(predictions.attrs)
        seen["frame"] = predictions.copy()

        class Sel:
            pass
        return Sel()
    monkeypatch.setattr(CS, "load_decision_streak_state", lambda *a, **k: State())
    monkeypatch.setattr(PK, "get_game_statuses_detailed", lambda date: {})
    monkeypatch.setattr(ST, "select_pick", fake_select)
    return seen


def test_the_slate_holds_the_witness_and_selection_sees_the_baseline_frame(tmp_path, monkeypatch, selection):
    preds = _preds()
    monkeypatch.setattr(O, "predict_local", lambda date: preds)
    out, sel, tier = O.run_and_pick(_config(tmp_path), "2027-04-01")
    assert _payload(tmp_path / "slates" / "2027-04-01.json")["serving"] == WITNESS
    assert selection["attrs"] == {} and out.attrs == {}
    pd.testing.assert_frame_equal(selection["frame"], _preds(serving=...))


def test_a_witness_drop_failure_never_breaks_selection(tmp_path, monkeypatch, selection):
    class Sticky(dict):
        def pop(self, *a):
            raise RuntimeError("synthetic pop failure")

    class StickyFrame(pd.DataFrame):
        sticky = Sticky(serving=WITNESS)

        @property
        def _constructor(self):
            return pd.DataFrame

        @property
        def attrs(self):
            return StickyFrame.sticky

        @attrs.setter
        def attrs(self, value):
            pass
    preds = StickyFrame(_preds(serving=...))
    monkeypatch.setattr(O, "predict_local", lambda date: preds)
    out, sel, tier = O.run_and_pick(_config(tmp_path), "2027-04-01")
    assert sel is not None and tier == "local" and out is preds
    assert _payload(tmp_path / "slates" / "2027-04-01.json")["serving"] == WITNESS


def test_run_and_pick_drops_the_witness_even_when_the_slate_writer_did_not(tmp_path, monkeypatch, selection):
    """save_slate normally takes the witness off first; run_and_pick's own drop is the backstop for selection."""
    import bts.slate as SLATE
    preds = _preds()
    monkeypatch.setattr(O, "predict_local", lambda date: preds)
    monkeypatch.setattr(SLATE, "save_slate", lambda *a, **k: None)          # a writer that leaves the attrs alone
    O.run_and_pick(_config(tmp_path), "2027-04-01")
    assert selection["attrs"] == {}
