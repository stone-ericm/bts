"""Unit tests for the pure decide_action + DecisionContext (Phase 2a).

decide_action is a behavior-neutral extraction of select_pick's action branch.
The heuristic tests pin the MDP-absent path (ctx.mdp=None); the MDP tests inject
a forced action via monkeypatching the _mdp_action_from seam (and use a weak
primary_p so the MDP path is distinguishable from the heuristic, which would skip);
the last two tests prove select_pick's executed action matches decide_action.
"""
import pandas as pd

from bts.strategy import decide_action, DecisionContext, select_pick


def _ctx(primary_p, second_p=0.78, has_diff=True, allow_double=True, mdp=None):
    return DecisionContext(primary_p=primary_p, second_p=second_p, has_diff_game=has_diff,
                           date="2026-06-17", allow_double=allow_double, mdp=mdp)


# --- Heuristic path (MDP absent: ctx.mdp is None) ---

def test_heuristic_skip_below_threshold():
    assert decide_action(_ctx(0.79), streak=8, saver=False) == "skip"


def test_heuristic_double_when_p_both_clears_threshold():
    # streak 8 -> threshold 0.55; 0.80*0.78=0.624 >= 0.55 -> double
    assert decide_action(_ctx(0.80, second_p=0.78), streak=8, saver=False) == "double"


def test_heuristic_single_when_p_both_below_threshold():
    # streak 8 -> 0.55; 0.80*0.60=0.48 < 0.55 -> single
    assert decide_action(_ctx(0.80, second_p=0.60), streak=8, saver=False) == "single"


def test_no_diff_game_forces_single():
    assert decide_action(_ctx(0.90, has_diff=False, second_p=None), streak=8, saver=False) == "single"


def test_allow_double_false_forces_single():
    assert decide_action(_ctx(0.90, second_p=0.90, allow_double=False), streak=8, saver=False) == "single"


def test_sprint_streak_never_doubles():
    # streak 56 -> _double_threshold None -> single even with great candidates
    assert decide_action(_ctx(0.95, second_p=0.95), streak=56, saver=False) == "single"


# --- MDP path (adjustment #5: heuristic-only tests don't cover the injected-policy
#     branch). primary_p=0.50 means the heuristic would SKIP, so a non-skip result
#     proves the MDP action was actually used. ---

def test_mdp_double_passes_through_when_executable(monkeypatch):
    monkeypatch.setattr("bts.strategy._mdp_action_from", lambda *a, **k: "double")
    assert decide_action(_ctx(0.50, second_p=0.50, mdp={"x": 1}), streak=12, saver=True) == "double"


def test_mdp_skip_overrides_strong_candidates(monkeypatch):
    monkeypatch.setattr("bts.strategy._mdp_action_from", lambda *a, **k: "skip")
    # MDP authority: skip even though p=0.95 would heuristically double, never skip
    assert decide_action(_ctx(0.95, second_p=0.95, mdp={"x": 1}), streak=12, saver=True) == "skip"


def test_mdp_double_downgraded_when_no_diff_game(monkeypatch):
    monkeypatch.setattr("bts.strategy._mdp_action_from", lambda *a, **k: "double")
    # post-MDP "double must be executable" guard -> single (heuristic would skip at 0.50)
    assert decide_action(_ctx(0.50, has_diff=False, second_p=None, mdp={"x": 1}), streak=12, saver=True) == "single"


def test_mdp_double_downgraded_when_allow_double_false(monkeypatch):
    monkeypatch.setattr("bts.strategy._mdp_action_from", lambda *a, **k: "double")
    # allow_double operational clamp applies to MDP doubles too (heuristic would skip at 0.50)
    assert decide_action(_ctx(0.50, second_p=0.50, allow_double=False, mdp={"x": 1}), streak=12, saver=True) == "single"


# --- Behavior preservation: select_pick's executed action matches decide_action ---

def _preds(rows):
    defaults = {
        "batter_id": 100001, "team": "NYM", "lineup": 1,
        "pitcher_name": "Test Pitcher", "pitcher_id": 200001,
        "game_time": "2026-04-01T23:10:00Z", "p_hit_pa": 0.30, "flags": "",
    }
    out = []
    for i, r in enumerate(rows):
        row = {**defaults, **r}
        row.setdefault("batter_name", f"Batter {i + 1}")
        out.append(row)
    return pd.DataFrame(out)


_AVAIL = {"abstract": "P", "detailed": "Pre-Game"}


def test_select_pick_double_matches_decide_action(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: None)  # force heuristic
    preds = _preds([
        {"batter_name": "A", "p_game_hit": 0.85, "game_pk": 111},
        {"batter_name": "B", "p_game_hit": 0.80, "game_pk": 222},
    ])
    result = select_pick(preds, "2026-04-01", tmp_path,
                         game_statuses_detailed={111: _AVAIL, 222: _AVAIL})
    sp_action = "double" if result.daily.double_down is not None else "single"
    direct = decide_action(_ctx(0.85, second_p=0.80), streak=8, saver=False)
    assert sp_action == direct == "double"   # 0.85*0.80=0.68 >= 0.55


def test_select_pick_single_game_matches_decide_action(tmp_path, monkeypatch):
    monkeypatch.setattr("bts.strategy._load_mdp", lambda: None)
    preds = _preds([{"batter_name": "Solo", "p_game_hit": 0.90, "game_pk": 111}])
    result = select_pick(preds, "2026-04-01", tmp_path,
                         game_statuses_detailed={111: _AVAIL})
    sp_action = "double" if result.daily.double_down is not None else "single"
    direct = decide_action(_ctx(0.90, has_diff=False, second_p=None), streak=8, saver=False)
    assert sp_action == direct == "single"
