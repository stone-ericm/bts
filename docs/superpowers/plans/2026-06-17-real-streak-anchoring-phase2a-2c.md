# Phase 2 (PART A: 2a + 2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Lay Phase 2's foundation — split `select_pick`'s action decision into a pure, testable `decide_action` (behavior-neutral), and replace the model-saver proxy with real ledger-based saver inference.

**Architecture:** Task 1 (2a) extracts a `DecisionContext` (the impure prep: candidates, MDP policy, `allow_double`) and a pure `decide_action(ctx, streak, saver)` from `select_pick`; `select_pick` keeps identical behavior. Tasks 2–3 (2c) add a `contest_ledger` parser and a saver inference that reads the persisted per-round ledger, replacing Phase 1's "model-saver when streaks agree" proxy. PART B (2b, the plausible-set/invariance gate) is a deferred follow-up — NOT in this plan.

**Tech Stack:** Python 3, pandas, pytest (`UV_CACHE_DIR=/tmp/uv-cache uv run pytest`). Spec: `docs/superpowers/specs/2026-06-17-real-streak-anchoring-phase2-design.md`.

**Branch:** create `phase2a-decide-action` off `main` before Task 1.

**Reference — current `select_pick` action branch** (`strategy.py:232-257`): MDP lookup (`_mdp_action`), else heuristic (`p<0.80`→skip; `_double_threshold` + a different-game second pick → double if `p_both>=threshold` else single; else single), then the `allow_double` clamp and the "double needs an executable different-game pick" guard. `decide_action` owns exactly this branch.

## Codex plan-review adjustments (apply during execution)

Codex verified Task 1 is behavior-preserving (MDP arg order matches today; the two `action=="double"` guards and the Step-5 splice are correct; leftover `_mdp_action` is harmless dead code; the ledger path is the right convention). Fixes to fold in:

1. **Task 3 — no-ledger MUST be conservative (biggest fix).** `infer_saver([])` must return `"unknown"`, NOT `"available"` — add `if not rounds: return "unknown"` at the top. So a missing/empty/bad ledger → `unknown` → `saver_available = False` (the spec's unknown→unavailable rule), never silently "available". Add a test: no `contest_ledger.jsonl` → `state.saver_available is False`.
2. **Tasks 2/3 — honor the stable-two-fetch rule** (spec §2c: latest ledger values are provisional). `parse_latest_ledger` reads the last TWO fetch rows; `LedgerRound` gains `stable: bool` (same `roundId`+`result`+`streak` present in both rows). `infer_saver` marks `consumed` ONLY on a *stable* consuming round; an unstable (single-read/provisional) consuming round → `unknown`. Add tests for both.
3. **Task 2 — don't fabricate `pre_streak`.** When a round's `post_streak` is None, set `prev_post = None` (break the chain) so the next round's `pre_streak` is correctly unrecoverable — don't carry the stale prior value.
4. **Task 1 — exact float behavior.** Today multiplies `best_row["p_game_hit"] * second["p_game_hit"]` raw; match it — store `primary_p`/`second_p` from the raw pandas values (drop the `float(...)` coercion, or coerce BOTH) so behavior is identical regardless of column dtype.
5. **Task 1 — strengthen the golden guard** (heuristic-only tests don't cover MDP). Add: monkeypatch `bts.strategy._mdp_action_from` → `"double"` and assert (a) `has_diff_game=False` → single, (b) `allow_double=False` → single; plus a behavior-preservation test that runs `select_pick` on a small fixture (diff-game / no-diff / a few streaks) and asserts its action == `decide_action(ctx, …)`.
6. **Line ref:** the saver block to replace in Task 3 Step 5 is at `contest_state.py:279` (the `if contest.saver_available is not None …` block), not `262-267`.

---

### Task 1: Extract `DecisionContext` + pure `decide_action` (2a, behavior-neutral)

**Files:**
- Modify: `src/bts/strategy.py`
- Test: `tests/test_decide_action.py` (new)

- [ ] **Step 1: Write the failing unit tests for `decide_action`**

`tests/test_decide_action.py`:
```python
from bts.strategy import decide_action, DecisionContext


def _ctx(primary_p, second_p=0.78, has_diff=True, allow_double=True, mdp=None):
    return DecisionContext(primary_p=primary_p, second_p=second_p, has_diff_game=has_diff,
                           date="2026-06-17", allow_double=allow_double, mdp=mdp)


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
```

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_decide_action.py -q`
Expected: FAIL — `decide_action` / `DecisionContext` not defined.

- [ ] **Step 3: Add `DecisionContext`, `_mdp_action_from`, and `decide_action` to `strategy.py`**

Insert after `_double_threshold` (after `strategy.py:88`):
```python
@dataclass
class DecisionContext:
    """Impure prep for one pick decision; `decide_action` is pure over (streak, saver)."""
    primary_p: float            # best candidate's p_game_hit
    second_p: float | None      # executable different-game second pick's p_game_hit (or None)
    has_diff_game: bool         # a valid different-game second pick exists
    date: str                   # YYYY-MM-DD (for MDP days_remaining)
    allow_double: bool          # global operational clamp (NOT uncertainty logic)
    mdp: dict | None            # injected MDP policy (None -> heuristic)


def _mdp_action_from(mdp: dict | None, p_game_hit: float, streak: int, date: str, saver: bool) -> str | None:
    if not mdp:
        return None
    from bts.simulate.mdp import lookup_action
    end = datetime.strptime(SEASON_END_DATE, "%Y-%m-%d")
    today = datetime.strptime(date, "%Y-%m-%d")
    days_remaining = max(0, (end - today).days)
    return lookup_action(mdp["policy_table"], mdp["boundaries"],
                         streak, days_remaining, saver, p_game_hit, mdp["season_length"])


def decide_action(ctx: DecisionContext, streak: int, saver: bool) -> str:
    """Pure skip/single/double decision given a prepared context + (streak, saver)."""
    action = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)
    if action is None:
        if ctx.primary_p < SKIP_THRESHOLD:
            action = "skip"
        elif _double_threshold(streak) is not None and ctx.has_diff_game and ctx.second_p is not None:
            p_both = ctx.primary_p * ctx.second_p
            action = "double" if p_both >= _double_threshold(streak) else "single"
        else:
            action = "single"
    if action == "double" and not ctx.allow_double:
        action = "single"
    if action == "double" and not ctx.has_diff_game:
        action = "single"
    return action
```

- [ ] **Step 4: Run — verify the unit tests pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_decide_action.py -q`
Expected: PASS.

- [ ] **Step 5: Refactor `select_pick` to use the context + `decide_action`**

In `strategy.py`, replace the action block (`strategy.py:232-260`, from `# Determine action:` through the `if action == "skip": return None`) with:
```python
    saver = load_saver_available(picks_dir) if saver_available is None else saver_available
    second_row = diff_game.iloc[0] if len(diff_game) >= 1 else None
    ctx = DecisionContext(
        primary_p=best_row["p_game_hit"],
        second_p=float(second_row["p_game_hit"]) if second_row is not None else None,
        has_diff_game=len(diff_game) >= 1,
        date=date,
        allow_double=allow_double,
        mdp=_load_mdp(),
    )
    action = decide_action(ctx, streak, saver)

    if action == "skip":
        return None
```
(The `_mdp_action` function may be left in place or removed; `decide_action` uses `_mdp_action_from`. The DailyPick construction below — `new_pick`, `double_pick`, `runner_up` — is unchanged and still uses `best_row`/`diff_game`.)

- [ ] **Step 6: Run the existing strategy + integration suites (behavior preservation)**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_strategy.py tests/test_decide_action.py tests/test_cli_integration.py -q`
Expected: PASS — `select_pick`'s behavior is unchanged (the golden guard: every existing `test_strategy.py` decision still holds).

- [ ] **Step 7: Commit**

```bash
git add src/bts/strategy.py tests/test_decide_action.py
git commit -m "refactor(strategy): extract pure decide_action + DecisionContext from select_pick (behavior-neutral)"
```

---

### Task 2: `contest_ledger` parser (2c foundation)

**Files:**
- Create: `src/bts/contest_ledger.py`
- Test: `tests/test_contest_ledger.py` (new)

- [ ] **Step 1: Write the failing test**

`tests/test_contest_ledger.py`:
```python
import json
from bts.contest_ledger import parse_latest_ledger, LedgerRound


def test_parses_per_round_with_pre_and_post_streak(tmp_path):
    led = tmp_path / "contest_ledger.jsonl"
    # two fetch rows; the LATEST row's predictions are parsed
    led.write_text("\n".join(json.dumps(r) for r in [
        {"recorded_at": "2026-06-16T17:00:00Z", "active_streak": 7, "predictions": []},
        {"recorded_at": "2026-06-17T17:00:00Z", "active_streak": 8, "predictions": [
            {"roundId": 904, "result": "hit", "streak": 6, "streakIncrease": 1,
             "roundPredictions": [{"playerId": 1, "result": "hit"}]},
            {"roundId": 905, "result": "hit", "streak": 7, "streakIncrease": 1,
             "roundPredictions": [{"playerId": 2, "result": "hit"}]},
            {"roundId": 903, "result": "hit", "streak": 5, "streakIncrease": 2,
             "roundPredictions": [{"playerId": 3, "result": "hit"}, {"playerId": 4, "result": "hit"}]},
        ]},
    ]))
    rounds = parse_latest_ledger(led)
    assert [r.round_id for r in rounds] == [903, 904, 905]   # sorted by roundId
    r905 = rounds[-1]
    assert isinstance(r905, LedgerRound)
    assert r905.post_streak == 7 and r905.pre_streak == 6      # pre = prior round's post
    assert rounds[0].is_dd is True and rounds[1].is_dd is False  # 903 had 2 slots


def test_missing_ledger_returns_empty(tmp_path):
    assert parse_latest_ledger(tmp_path / "nope.jsonl") == []
```

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_ledger.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the parser**

`src/bts/contest_ledger.py`:
```python
"""Parse the per-round MLB ledger (contest_ledger.jsonl) into a per-round series.

Each ledger line is one fetch: {recorded_at, active_streak, ..., predictions: [...]}.
A prediction row carries the POST-round `streak`; pre_round_streak is the prior
settled round's post_streak. Finality/correction state is NOT in the rows — callers
must treat the latest values as provisional (see saver inference).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LedgerRound:
    round_id: int
    result: str | None          # hit / not_hit / void / None
    pre_streak: int | None      # prior round's post_streak (None for the earliest row)
    post_streak: int | None
    streak_increase: int | None
    is_dd: bool                 # two entered slots


def parse_latest_ledger(ledger_path: Path) -> list[LedgerRound]:
    """Parse the most recent fetch's predictions into rounds sorted by roundId."""
    if not ledger_path.exists():
        return []
    lines = [ln for ln in ledger_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    try:
        latest = json.loads(lines[-1])
    except json.JSONDecodeError:
        return []
    preds = sorted(
        (p for p in latest.get("predictions", []) if p.get("roundId") is not None),
        key=lambda p: int(p["roundId"]),
    )
    rounds: list[LedgerRound] = []
    prev_post: int | None = None
    for p in preds:
        post = p.get("streak")
        rounds.append(LedgerRound(
            round_id=int(p["roundId"]),
            result=p.get("result"),
            pre_streak=prev_post,
            post_streak=post if isinstance(post, int) else None,
            streak_increase=p.get("streakIncrease"),
            is_dd=len(p.get("roundPredictions", [])) >= 2,
        ))
        if isinstance(post, int):
            prev_post = post
    return rounds
```

- [ ] **Step 4: Run — verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_ledger.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bts/contest_ledger.py tests/test_contest_ledger.py
git commit -m "feat(ledger): parse contest_ledger.jsonl into a per-round series (pre/post streak, DD)"
```

---

### Task 3: Ledger saver inference + retire the proxy (2c)

**Files:**
- Modify: `src/bts/contest_ledger.py` (add `infer_saver`)
- Modify: `src/bts/contest_state.py` (`load_decision_streak_state` uses the ledger saver)
- Test: `tests/test_contest_ledger.py`, `tests/test_decision_saver_fallback.py`

- [ ] **Step 1: Write the failing inference tests**

Append to `tests/test_contest_ledger.py`:
```python
from bts.contest_ledger import infer_saver, LedgerRound


def _r(rid, result, pre, post, is_dd=False):
    return LedgerRound(rid, result, pre, post, None, is_dd)


def test_saver_consumed_on_miss_at_10_15_that_did_not_reset():
    # not_hit at pre-streak 12 with post 12 (didn't reset) -> the mulligan absorbed it
    assert infer_saver([_r(1, "hit", 11, 12), _r(2, "not_hit", 12, 12)]) == "consumed"


def test_saver_available_when_no_consuming_round():
    assert infer_saver([_r(1, "hit", 7, 8), _r(2, "hit", 8, 9)]) == "available"


def test_saver_unknown_on_ambiguous_dd_miss_at_10_15():
    # a not_hit at 10-15 that didn't reset BUT is a DD (one slot may have missed) -> ambiguous
    assert infer_saver([_r(1, "hit", 11, 12), _r(2, "not_hit", 12, 12, is_dd=True)]) == "unknown"


def test_saver_unknown_when_pre_streak_unrecoverable():
    assert infer_saver([_r(1, "not_hit", None, 11)]) == "unknown"
```

- [ ] **Step 2: Run — verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_ledger.py -k saver -q`
Expected: FAIL — `infer_saver` not defined.

- [ ] **Step 3: Implement `infer_saver`**

Append to `src/bts/contest_ledger.py`:
```python
def infer_saver(rounds: list[LedgerRound]) -> str:
    """Return 'consumed' | 'available' | 'unknown' for the one-time 10-15 saver.

    Consumed: a settled not_hit whose pre-streak was 10-15 and whose streak did NOT
    reset to 0 (the mulligan absorbed it). Ambiguous (-> unknown, never 'consumed'):
    a DD (one slot may have missed), or an unrecoverable pre-streak. Clear consumption
    anywhere wins; else an ambiguous candidate -> unknown; else available.
    """
    ambiguous = False
    for r in rounds:
        if r.result != "not_hit":
            continue
        if r.pre_streak is None or r.post_streak is None:
            ambiguous = True
            continue
        if 10 <= r.pre_streak <= 15 and r.post_streak != 0:
            if r.is_dd:                 # one-slot DD miss can look the same
                ambiguous = True
                continue
            return "consumed"
    return "unknown" if ambiguous else "available"
```

- [ ] **Step 4: Run — verify the inference tests pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_ledger.py -q`
Expected: PASS.

- [ ] **Step 5: Wire the ledger saver into `load_decision_streak_state`, retiring the proxy**

In `contest_state.py` `load_decision_streak_state`, replace the contest-saver block (`contest_state.py:262-267` — the `if contest.saver_available is not None: … elif status == "fresh" and contest.streak == model_streak: contest_saver = model_saver … else: contest_saver = False`) with:
```python
    # Saver: prefer an explicit contest value; else infer from the per-round ledger
    # (Phase 2 — replaces the model-saver-when-streaks-agree proxy). 'unknown' is
    # conservatively unavailable until Phase 2b carries it as set uncertainty.
    if contest.saver_available is not None:
        contest_saver = contest.saver_available
    else:
        from bts.contest_ledger import parse_latest_ledger, infer_saver
        led = infer_saver(parse_latest_ledger(picks_dir / "account_state" / "contest_ledger.jsonl"))
        contest_saver = (led == "available")
```
(This block still sits AFTER the `status` is computed, so `status` is available; remove the now-unused `status == "fresh"` saver path.)

- [ ] **Step 6: Update `tests/test_decision_saver_fallback.py`**

Those two tests assert the OLD proxy (fresh + agreeing streaks -> model saver). The proxy is gone. Rewrite them so saver comes from a ledger fixture: write a `contest_ledger.jsonl` with no consuming round -> `saver_available is True`; write one with a consumed-saver round -> `False`. (Mirror the `infer_saver` fixtures; put the ledger at `tmp_path/"account_state"/"contest_ledger.jsonl"`.)

- [ ] **Step 7: Run the contest-state + saver suites**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_contest_state.py tests/test_decision_saver_fallback.py tests/test_contest_ledger.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/bts/contest_ledger.py src/bts/contest_state.py tests/test_contest_ledger.py tests/test_decision_saver_fallback.py
git commit -m "feat(saver): infer the 10-15 saver from the per-round ledger; retire the model-saver proxy"
```

---

### Task 4: Full suite + ship

- [ ] **Step 1:** `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -k "strategy or decide or contest or ledger or saver or cli or scheduler or health" -q` → PASS.
- [ ] **Step 2:** read-only live sanity: `ssh bts-hetzner 'cd ~/projects/bts && .venv/bin/python -c "from pathlib import Path; from bts.contest_state import load_decision_streak_state as L; print(L(Path(\"data/picks\")).saver_available)"'` (sanity that the ledger-saver path runs against the real ledger).
- [ ] **Step 3:** PR + `git push origin main:deploy` (canary). 2a is behavior-neutral; 2c only changes the saver value at 10-15 (you're at 8, so no live behavior change yet — safe).

## Self-Review

**Spec coverage:** 2a `decide_action`/`DecisionContext` split + `allow_double` kept + golden (Task 1); observation model — 2c reads the ledger (account evidence), never local picks (Tasks 2-3); 2c parser + saver inference + ambiguity→unknown→conservative + proxy retired (Tasks 2-3). PART B (2b plausible-set/gate) intentionally NOT planned. **Type consistency:** `DecisionContext(primary_p, second_p, has_diff_game, date, allow_double, mdp)` and `LedgerRound(round_id, result, pre_streak, post_streak, streak_increase, is_dd)` are used identically across tasks; `infer_saver -> "available"|"consumed"|"unknown"`. **No placeholders.** **Note:** confirm `lookup_action`'s exact arg order against `bts/simulate/mdp.py` during Task 1 (kept identical to today's `_mdp_action`).
