# Early-Lock Timing and MDP Aggressiveness Audit (2026-05-28)

**Status:** evidence-only. No production behavior change, no policy artifact
write, and no deploy claim.

## Question

The original production concern was early lock timing: the live scheduler may
commit to an early-game pick before later confirmed lineups can improve the
candidate pool. During that audit, a second mechanism appeared: the saved MDP
policy cannot distinguish much of the current live probability range, so weak
double-down pairs can look over-aggressive at low streaks.

This memo keeps those mechanisms separate:

1. **Scheduler lock timing:** did fallback force an early pick even when the
   lock decision said to wait?
2. **MDP action aggressiveness:** after the primary exists, does the MDP choose
   `double` in a live probability regime below its lowest saved boundary?

## Key Finding

The early-lock concern was real historically. The clearest counterexample is
2026-05-22, when the system delivered Nico Hoerner before his 14:20 ET game
even though the fallback refresh still had `should_lock=False` and most later
lineup windows were still pending.

Current code has a direct guard for this case: if fallback refresh returns
`should_lock=False` and future scheduled checks still have pending lineup data,
the scheduler archives the unsafe fallback candidate and continues. A focused
test covers that path, and 2026-05-28 production naturally exercised it.

The MDP/bin-collapse finding is real too, but it is secondary to the original
timing question and should not be implemented in the early-lock branch.

## Historical Counterexample: 2026-05-22

Observed scheduler behavior:

- `13:20 ET`: lineup check ran; only two new confirmed lineups were available.
  The prediction run still noted `28` teams using projected lineups.
- `13:28 ET`: candidate was `Nico Hoerner` at `0.700`, with
  `gap=1.5%` versus best projected, so `should_lock=False`.
- The selected primary pick's game started at `14:20 ET`; there was no
  scheduled lineup check before the fallback deadline.
- `13:53 ET`: fallback refresh kept `Nico Hoerner` at `0.700`; the broader
  slate still had `28` projected teams.
- The scheduler force-delivered before first pitch.
- A restart soon after found the pick already locked, so later lineup checks
  could not replace it.

The root pick JSON confirms the shape: Hoerner was the primary in game
`824679` at `14:20 ET` with `projected_lineup=false`; the double-down was
Ronald Acuna Jr. in game `824922` at `19:15 ET` with `projected_lineup=true`.
The post-restart scheduler state had only game `824679` confirmed and all later
game windows still unconfirmed. Raw game data for both games ended at
`detailedState=Final`, so this was not the postponed-game status artifact.

The pick missed. The miss is not the proof by itself; the proof is that the
scheduler committed even though the lock decision said the confirmed-lineup
pool was not strong enough and future lineup windows remained.

That is the exact shape Eric was reacting to: not just "a weak pick", but a
weak early-game pick that cut off the chance to wait for later confirmed
lineups.

## Current Code and Test Surface

The current scheduler computes whether later checks can still add lineup data:

```text
_has_pending_future_confirmation_window(future_runs, confirmed_sides)
```

In the fallback path, current behavior is:

```text
if refresh.should_post is False and has_pending_future_window:
    archive unsafe candidate
    continue scheduler loop
```

Existing focused coverage:

```text
tests/test_scheduler.py::TestRunDay::test_fallback_defers_when_should_lock_false_and_future_checks_remain
tests/test_scheduler.py::TestRunDay::test_fallback_defers_when_double_down_game_creates_early_deadline
```

Those tests cover both early-deadline shapes: the original primary's own game
is early, and an early double-down game pulls the fallback deadline ahead of a
later primary. In both cases they return `should_post=False` and verify:

- no Bluesky/DM delivery;
- no live-forward capture;
- no root pick JSON remains;
- one `deferred_fallback_*.json` archive is written; and
- the archive reason is `should_lock_false_future_checks_remain`.

There is also a complementary test that fallback still delivers when no future
pending lineup window remains. That preserves the original safety requirement:
do not drift past first pitch when there is no real information left to wait
for.

## Accepted Tradeoff

The current defer behavior intentionally abandons a confirmed early-game primary
when it is below the lock threshold and later lineup windows still have pending
information. That is the cost of de-aggressing early locks: on some
early-game-heavy days, waiting may lead to a later pick, a weaker pick, or a
skip if no later candidate clears the delivery rules.

That tradeoff is explicit. It is different from missing a submission window:
when no future pending lineup window remains, the complementary fallback path
still delivers before first pitch.

## Current Production Evidence: 2026-05-28

The 2026-05-28 production lifecycle exercised the newer fallback/defer path:

- `12:10 ET`: candidate `Nathan Lukes` at `0.6986`, not locked.
- `13:10 ET`: same candidate, still not locked.
- `13:43 ET`: fallback refresh changed primary to `Ozzie Albies` at `0.7230`;
  `should_lock=False`, so fallback was deferred because future lineup checks
  with pending lineup data remained.
- `15:10 ET`: a later lineup check selected `Ozzie Albies` at `0.7369` with
  `gap=3.8%` versus best projected contender, `should_lock=True`, and the pick
  was delivered by DM.

That is the desired response to the 2026-05-22 failure mode: wait when the
fallback candidate is not lock-worthy and later lineup windows still matter,
then deliver only after the confirmed-lineup pool improves enough.

## Secondary MDP Finding

The deployed policy artifact remains:

```text
data/models/mdp_policy.npz
boundaries = [0.7959788446, 0.8114907020, 0.8252474190, 0.8407401967]
season_length = 180
```

All root-level production primary picks through 2026-05-28 mapped below the
lowest saved boundary:

| Window | Picks | Primary p range | Q-bin counts |
|---|---:|---:|---|
| Full production window | `61` | `0.690137` to `0.791806` | `{Q0: 61}` |
| Recent 21 picks | `21` | `0.690137` to `0.787996` | `{Q0: 21}` |

At low streaks with saver available, Q0 maps to `double` for every audited
production pick. That creates weak-looking double-down examples:

| Date | Primary | p1 | Double-down | p2 | independent p_both |
|---|---|---:|---|---:|---:|
| 2026-05-21 | Jose Ramirez | `0.690` | Carlos Cortes | `0.687` | `0.474` |
| 2026-05-22 | Nico Hoerner | `0.700` | Ronald Acuna Jr. | `0.686` | `0.480` |
| 2026-05-23 | Bo Bichette | `0.723` | Julio Rodriguez | `0.718` | `0.519` |
| 2026-05-28 | Ozzie Albies | `0.737` | Nathan Lukes | `0.699` | `0.515` |

This is evidence for a separate double-down guardrail or policy-calibration
workstream. It does not prove that primary selection should lock earlier or
later, and it should not be allowed to blur the early-lock conclusion.

## Recommended Next Step

For the early-lock topic:

- treat 2026-05-22 as the historical failure shape;
- treat the current fallback-defer path as the intended fix;
- keep the existing regression tests as the code-level guardrail;
- continue live validation on the next naturally deferred fallback day; and
- do not add a new live behavior change unless a fresh example shows the
  current guard misses a May-22-shaped case.

For the MDP/DD topic:

- keep `docs/sota_audit/2026-05-29-mdp-dd-guardrail-prereg.md` as a separate
  pre-registration artifact;
- do not re-solve or swap `data/models/mdp_policy.npz` from this evidence;
- do not ship a Q0/pair-floor guardrail without a separate backtest and review
  gate.

## Verification

Focused existing scheduler test:

```bash
PYTHONPATH=src /Users/stone/projects/bts/.venv/bin/python \
  -m pytest \
    tests/test_scheduler.py::TestRunDay::test_fallback_defers_when_should_lock_false_and_future_checks_remain \
    tests/test_scheduler.py::TestRunDay::test_fallback_defers_when_double_down_game_creates_early_deadline \
    -q
```

Audit script test:

```bash
PYTHONPATH=src /Users/stone/projects/bts/.venv/bin/python \
  -m pytest tests/scripts/test_audit_mdp_live_aggressiveness.py -q
```
