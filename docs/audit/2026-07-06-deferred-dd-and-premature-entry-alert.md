# 2026-07-06 — Deferred double-down (early DD game, late projected primary) and a premature "pick not entered" alert

## Summary

A `check-pick-entered` DM ("⚠️ BTS pick NOT entered in MLB app … Fix it now!")
fired at 13:30 ET for a pick that had **never been delivered or locked**, and
was in fact deferred and deleted by the scheduler 16 minutes later. Eric flagged
it as firing early.

Investigation confirmed one real, unambiguous bug (the alert), **falsified** a
suspected second bug (a "silent forfeited day"), and surfaced one open strategy
question (whether to lock a strong double-down early on a projected lineup).

This was the trigger case:

| Slot | Player | Game | First pitch (ET) | Lineup | p_game_hit |
|------|--------|------|------------------|--------|------------|
| Primary (top pick) | Luis Arraez (SF) | 823205 | **9:45 PM** | **projected** | 87.8% |
| Double-down | Lane Thomas (KC) | 824089 | **2:10 PM** | confirmed | 74.6% |

The MDP genuinely selected a **double-down** (`double_down` is only populated
when `action == "double"` — `strategy.py:370-383`). The two games are ~7.5 hours
apart, and the DD's game locks first.

### Relevant BTS rules (confirmed by Eric)

- When **two** picks are entered, **both lock at the earlier game's first
  pitch** — the later pick locks early. So this DD would lock entirely at 2:10 PM.
- Any pick must be submitted **≥ 5 minutes before first pitch**. The true
  submission deadline for this DD was therefore **2:05 PM**, not 2:10.
- A **single** pick locks at its own game's first pitch (so a lone Arraez would
  lock at 9:45 PM, leaving time for his real lineup).

## Timeline (2026-07-06, ET)

- **13:10–13:21** — Lineup check. MDP selects Arraez (projected, 9:45) + Thomas
  (confirmed, 2:10) as a double-down. Log: `should_lock=False`. Writes
  `data/picks/2026-07-06.json`.
- **13:30:01** — `check-pick-entered` cron runs. `{date}.json` still exists →
  `load_pick` returns the pick → 40 min to the earliest game (2:10), inside the
  75-min window → entry not found → **DM sent**. Marker written:
  `{reason: no_pick, status: alerted}`.
- **13:35 / 13:46:58** — Fallback deadline for the earliest game. `should_lock`
  still False; future checks with pending lineups remain →
  **FALLBACK DEFERRED**. `_defer_pick_at_fallback` archives the pick and
  **deletes `data/picks/2026-07-06.json`**. Scheduler sleeps until 17:40 ET.
- **17:40 ET** — (predicted) scheduler wakes, `load_pick` returns None →
  re-selects fresh from the evening slate. Thomas's early game has started and is
  filtered out; the day gets a new evening single/DD.

## Q1 — Confirmed: the locker will not lock this DD, and the early half's odds are irrelevant

Three **delivery** paths exist, all gated on `should_post` ← `should_lock`:

1. **Lineup-check delivery** (`scheduler.py:2054`) needs `should_post=True`.
   `should_lock` (`strategy.py:211-237`) returns `False` immediately when the
   **top/primary** pick's lineup is projected (`strategy.py:223`). Arraez's 9:45
   PM lineup does not confirm until ~7:45 PM — hours after the 2:10 lock — so this
   never fires in time.
2. **In-loop fallback** (`scheduler.py:2155`) delivers *unless*
   `should_post is False AND has_pending_future_window`. Both were true, so it
   **deferred**. It would only force-commit a projected pick here if no later game
   still had a pending lineup (`_has_pending_future_confirmation_window`,
   `scheduler.py:911-923`) — never the case on a normal evening slate.
3. **Final/post-loop fallback** (`scheduler.py:2244-2250`, "delivering on
   projected data") runs only after the loop already deferred past 2:10 — too late
   for the early half.

**Key point for the strategy question below:** `should_lock` never inspects the
**early DD half's** odds. It keys entirely on whether the *primary* pick is
projected, plus a gap check against *other projected* games. Thomas at 74.6% vs
99% would change nothing. The code does the opposite of "lock early if the early
half is strong enough" — that logic does not exist.

**Precision correction (Codex review):** "only three lock paths" was imprecise.
There is a fourth path that sets `state.pick_locked` **without delivering** — the
classification lock: `run_single_check` (`scheduler.py:1054-1068`) returns a
locked result for an existing pick whose committed game has started, and
`run_day` marks `state.pick_locked` (`scheduler.py:2032-2052`). It is not a
*delivery* path and is not gated on `should_post`, but it is a lock path. It
requires `{date}.json` to still exist — see the crash caveat under Q2.

## Q2 — The suspected freeze mechanism does not apply in the deferral flow; "no forfeit" is NOT fully proven

Initial static reading suggested the undelivered pick would freeze: once Thomas's
2:10 game starts, `_classify_unposted_game_status` (`picks.py:626`) marks a
committed game `locked` (abstract != "P"), `_committed_pick_game_pks`
(`picks.py:743`) includes the DD game, and `classify_pick_lock_state`
(`picks.py:748-792`) would then return `locked=True` for an *undelivered* pick —
which `select_pick` (`strategy.py:288-293`) would return without re-selecting.

**That trace was wrong, because it assumed `{date}.json` persists after
deferral. It does not.** `_defer_pick_at_fallback` (`scheduler.py:1556-1571`)
**unlinks** `data/picks/{date}.json` (line 1569-1570) — docstring: *"Archive and
remove an unsafe fallback candidate so later checks refresh."*

Empirical confirmation on the production box (bts-hetzner), 2026-07-06:

```
$ .venv/bin/python -c "... load_pick('2026-07-06', Path('data/picks')) ..."
load_pick(2026-07-06) -> None
```

With `load_pick` returning None, the classifier is never reached: `select_pick`
sees `current=None`, skips reuse, and re-selects fresh from the evening slate. So
the **specific freeze-via-persisted-file mechanism does not apply** to today's
deferred pick.

**But do not overstate this as "the day is not forfeited" (Codex review).** Two
things remain unproven / at risk:

1. **Evening delivery not observed.** As of the prod check (~14:42 ET) the
   scheduler was still `sleeping until 17:40 ET`; no evening pick had been
   selected or delivered. The proven facts are only: stale DD archived, live
   `{date}.json` deleted, `load_pick → None`, scheduler asleep. That the 17:40+
   runs actually re-pick *and deliver* a pick is expected from the mechanism but
   was **not** verified end-to-end. (Follow-up: confirm from the evening logs.)
2. **Crash-before-defer residual forfeit risk.** The unlink only protects us
   because the defer branch actually ran. If the daemon crashed *after* writing
   the preview `{date}.json` (~13:21) but *before* the defer/unlink (~13:46), the
   preview would persist, and on the next wake the classification-lock path
   (Q1 precision note) would mark the undelivered preview `pick_locked` once
   Thomas's early game started — a genuine no-pick day (health checks might catch
   it late). Narrow window, but real, and it is the general form of the "freeze"
   I first suspected.

So Q1 is confirmed; Q2's original "silent forfeit" is **not** the normal-flow
outcome, but "no forfeit, ever" is stronger than the evidence supports.

## Bug (real, unambiguous): `check-pick-entered` alarms on undelivered picks

`check-pick-entered` (`cli.py:1530-1650`) fires whenever a `{date}.json` exists
and the earliest game is within the window. But `{date}.json` is rewritten all
day with previews/projections and does **not** imply the pick was delivered to a
human. It never calls `pick_was_delivered` (`picks.py:299-307`) — the exact gate
that `check-results` already uses (GH #144: "a stale preview/undelivered
`<date>.json` … no longer corrupts the streak").

Result: it nagged Eric to enter a pick the system never handed him, and then
deferred/deleted 16 minutes later.

**Fix — gate on COMMIT/LOCK state, not `pick_was_delivered` alone (Codex review).**
A naive `pick_was_delivered(daily)` gate would *suppress legitimate alerts*,
because two committed-but-not-"delivered" states exist in `_deliver_and_lock_pick`:

- `private_locked` (`scheduler.py:634-646`) — private mode: `pick_locked=True`,
  never posted/DM'd, so `pick_was_delivered` is False. Eric must still enter it
  manually. (Prod is currently `pick_delivery = "dm"`, so not live today — but the
  mode is configurable and this would silently disable the alert in private mode.)
- `locked_unconfirmed` (`scheduler.py:618-632`) — daemon crashed mid-send:
  `delivery_attempted=True`, `pick_locked=True`, delivery flags unset, so
  `pick_was_delivered` is False. This is precisely the "verify manually" case where
  the reminder is *most* wanted. **This one is live-relevant even in dm mode.**

The correct gate is the same commit signal `check-results` uses (GH #144):
`decision.json` commit / `scoreable`, with `pick_was_delivered` as the fallback
when no decision record exists. All three commit states
(`delivered`/`private_locked`/`locked_unconfirmed`) write a decision via
`_record_commit`, so a commit-based gate fires for all real picks and stays silent
for undelivered previews/deferred picks (today: no decision.json → correctly
silent). Tests must cover: deferred/undelivered → no DM; `private_locked` → DM;
`locked_unconfirmed` → DM; normal `delivered` → DM.

### Minor correctness nit

The alert's countdown measures minutes to **first pitch** (2:10), not to the true
**submission cutoff** of first pitch − 5 min (2:05). The forced *delivery* already
targets ~35 min early, so a delivered pick clears the cutoff comfortably; but the
alert copy/threshold should use first-pitch − 5 to match the real deadline.

## Open strategy question: lock a strong DD early on a projected lineup?

Today the system passed up an 87.8% / 74.6% double-down because it would not
commit Arraez's projected lineup at the 2:05 lock, and re-picked from the evening
slate instead. That is defensible-conservative (never commit a projected top pick
blind). Eric's instinct is that a strong-enough early opportunity should be worth
locking on projection.

Framing correction: when you lock this DD early, the risk you accept is on the
**late projected half** (Arraez, committed blind at 2:05), **not** the early
confirmed half (Thomas). The early half's odds tell you whether the DD is *worth*
spending the early lock; the late half's projection reliability is what you are
gambling on.

Prior work is the seed of the safety threshold: `should_lock` + `early_lock_gap`
(`473494b`), empirically derived (`f7e1ffe`), and the earliest-game fallback
deadline (`6fd61f9`). What is missing is a path that force-delivers on projection
at the earliest-game deadline **when the early game belongs to the DD and the
primary is a later projected game**, gated by a reliability threshold on the
projected member (deliver-as-is if dominant/high-confidence; otherwise drop to a
single — e.g. make the projected player a single that locks at its own later
game).

This is a strategy change to the live-streak scheduler. It should get its own
spec/plan, TDD, and a Codex adversarial pass before shipping — not be bundled with
the alert fix.

## Recommendations

1. **Ship now — DONE (branch `fix/check-pick-entered-commit-gate`):** gate
   `check-pick-entered` on the **commit/lock state** via `is_scoreable_commit`
   (decision.json `scoreable`, fallback `pick_was_delivered`) before the window
   check and any contest fetch; the firing window now excludes the un-submittable
   final 5 min; the DM countdown reports minutes to the first-pitch − 5 cutoff.
   Tests: undelivered preview → silent; decision-record commit → DM; skip-day
   decision → silent; inside-cutoff → silent; countdown-to-cutoff wording.
2. **Mostly no change, but verify + harden:** the deferral's delete-and-re-pick is
   correct for the normal flow. Follow-ups: (a) confirm from the evening logs that
   2026-07-06 actually re-picked and delivered; (b) consider the crash-in-the-
   `write→defer` window (Q2 residual) — decide whether the classification-lock
   path should refuse to lock an *undelivered* preview whose committed game
   started (i.e. treat it as stale → re-pick) rather than freeze it.
3. **Decide separately:** whether to add "deliver early on projection with a
   safety threshold" for early-game double-downs (Eric's belief). Own spec + TDD +
   Codex if pursued.

## Codex adversarial review (2026-07-06, gpt-5.5, repo + live-box access)

An independent Codex pass verified the claims against the code and prod. Outcome —
three valid corrections, folded in above; no claim was fabricated or reversed:

- **Q1** — correct but imprecise; added the non-`should_post` classification-lock
  path.
- **Q2** — deletion confirmed on prod (`source_exists False`), but flagged
  "day is not forfeited" as **overstated**: evening delivery unverified + the
  crash-before-defer residual. Softened accordingly. This was the highest-signal
  pushback.
- **Alert fix** — flagged `pick_was_delivered`-only as **incomplete** (suppresses
  `private_locked` / `locked_unconfirmed`). Fix revised to a commit/lock gate.
- **T-5 nit** and **strategy framing** — confirmed correct.

### Codex pre-merge review of the fix (round 2, gpt-5.5, repo access)

Reviewed `fix/check-pick-entered-commit-gate` adversarially. Five findings; two
fixed in the patch, three triaged as deferred/low-risk:

**Fixed:**
- **Window vs cutoff (HIGH)** — the DM used the T-5 cutoff but the firing window
  still ran to first pitch, so it could nag inside the un-submittable final 5 min
  and print a negative countdown (`(-1 min to submit)`). Window lower bound raised
  to `submit_cutoff_min` (5). Regression test `test_no_alert_inside_submission_cutoff`.
- **Test gap** — the positive tests only exercised the `pick_was_delivered`
  fallback, not the `decision.json` gate. Added `test_committed_via_decision_record_alerts`
  (scoreable commit, no delivery flags → DM) and `test_skip_day_decision_no_dm`
  (action=skip / scoreable=false → silent).

**Deferred (surfaced for decision, not slipped into this patch):**
- **`private_locked`/`locked_unconfirmed` with a missing/failed decision.json
  (HIGH)** — `is_scoreable_commit` falls back only to `pick_was_delivered`, which
  is False for those not-delivered lock states, so if the *best-effort* decision
  write failed the alert would be suppressed. Narrow (double-fault; private mode
  isn't used in prod's `dm` config). The robust fix is to gate on the scheduler's
  authoritative `pick_locked` instead of `is_scoreable_commit` — a deliberate
  change to the commit signal that also affects `check-results`, so it wants its
  own change, not this one.
- **`load_decision` laxity (MEDIUM)** — it doesn't assert the record's internal
  `date` equals the requested date, and `is_scoreable_commit` ignores `action`
  (an inconsistent `action=skip, scoreable=true` record would alert). Both are
  corruption-class and live in code shared with `check-results`; changing shared
  validation is out of scope for the alert fix.
- **Alert reads `{date}.json`, not the committed decision payload (MEDIUM)** — a
  post-lock divergence between `{date}.json` and the decision record would make
  the alert check the wrong player. Low real risk: the scheduler stops rewriting
  `{date}.json` once `pick_locked` (breaks the check loop), so they stay aligned
  post-commit. Reworking the checker to read the decision payload is a larger
  change; noted as a follow-up.

## Code references

- `strategy.py:211-237` — `should_lock` (line 223: projected top pick → False)
- `strategy.py:288-293` — `select_pick` reuse/locked branch
- `strategy.py:370-383` — `double_down` set only when `action == "double"`
- `scheduler.py:2054` — lineup-check delivery gate
- `scheduler.py:2155-2172` — in-loop fallback defer/deliver
- `scheduler.py:2244-2250` — final fallback "delivering on projected data"
- `scheduler.py:1556-1571` — `_defer_pick_at_fallback` (unlinks `{date}.json`)
- `scheduler.py:911-923` — `_has_pending_future_confirmation_window`
- `picks.py:299-307` — `pick_was_delivered`
- `picks.py:600-641` — `_classify_unposted_game_status`
- `picks.py:741-745` — `_committed_pick_game_pks`
- `picks.py:748-792` — `classify_pick_lock_state`
- `cli.py:1530-1650` — `check-pick-entered`
