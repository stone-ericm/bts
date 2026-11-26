# Tail policy (E[season-best]) — mechanism audit 9/03 → 9/27 (P-05)

**Date computed:** 2026-09-28 · **Plan item:** W1.4b · **Register:** P-05 → X-16 (`docs/audit/2026-09-22-exposure-register.md`) · **Data:** frozen W0.7 snapshot `final-20260928/` (manifest sha256 `e3dce024…`) · **Evidence:** `docs/audit/2026-09-28-p05-tail-policy-audit.json` · **Reader:** `scripts/audit/p05_tail_policy_audit.py` (rules pinned by `tests/scripts/test_p05_tail_policy_audit.py`)

## Protocol (quoted, unchanged)
- Design (`docs/audit/2026-09-03-emax-tail-policy.md` §3): "Stop rule, explicit: skip iff `min(57, s + 2d) <= m`"; §1: the regime is resolved from state alone — `reach57` iff `streak + 2*d_eff >= 57`, else `emax_season_best`; first-day acceptance = decision.json `objective=emax_season_best`, `best_status=trusted`, `effective_best=18`, `tail_policy_sha256 == dc5d0c99…`, `degraded_reason=null`.
- W1.4b obligation: "audit objective/state/provenance per date 9/03→9/27, stop behaviour on 9/19, delivered/entered/private separation. A few dates validate the mechanism, not the rates." This read reports no hit rates and makes no E[best]-optimality claim.

## Sources joined per date
`<date>/decision.json` (schema v3) · the pick file, when one exists · every scheduler-journal `Policy:` line (122 lines in the window) · the contest ledger (the contest's own record of entered, graded rounds) · the frozen rounds calendar (days left = rounds from the date through 9/27) · the frozen policy artifacts.

## Result
**0 violations across 25 dates × 191 checks.**

| Check | Result |
|---|---|
| Artifact binding | `mdp_tail_policy.npz` sha256 `dc5d0c99…` (`bts_tail_policy_v1`, objective `emax_season_best`) declares base `66d15471…` == sha256 of `mdp_policy.npz` |
| Regime from state (design §1) | `emax_season_best` on all 25 dates, as the rule requires (the largest `s + 2d` in the window is 50 on 9/03) |
| Days left | every journal line's `days` equals the rounds calendar (25 on 9/03 → 1 on 9/27) |
| Best-streak trust | `best_streak = 18`, `best_status = trusted`, `effective_best = 18` on all 25 dates, including 9/14–9/27 while the contest profile sat at source date 9/13 (stale — no entries after 9/13) |
| Provenance | `tail_policy_sha256` = the artifact on every decision.json and every pick file; `degraded_reason = null` throughout; pick files carry `policy_decision.objective = emax_season_best` |
| Journal vs decision.json | all 122 `Policy:` lines agree with that day's decision (objective, action, streak, effective best, tail) |
| **Stop behaviour** | double on 9/03–9/18 (on 9/18: 0 + 2·10 = 20 > 18 → play); **skip from 9/19** (0 + 2·9 = 18 ≤ 18 → stop) and on every day through 9/27 — the stop rule matches the recorded action on all 25 dates |

## Delivered / entered / private separation
| Period | Days | Delivery (`decision.json`) | Contest (ledger) | Policy state |
|---|---|---|---|---|
| 9/03 → 9/13 | 11 | `delivered`, `scoreable` | entered: every round has graded slots | contest profile, `fresh` (source date = previous day) |
| 9/14 → 9/18 | 5 | `private_locked` (computed, not delivered) | not entered | contest profile frozen at 9/13 (`fresh` → `lagged` → `stale`); streak 0 every day |
| 9/19 → 9/27 | 9 | `not_applicable` (skip), not scoreable; no pick file | not entered | same; stop rule |

- **Private results never reached the policy or the contest.** The private picks were still graded locally (`scoreable = true`, as intended while the box was silent), but the policy's streak came from the contest profile. After both 9/15 private legs hit, the 9/16 decision still ran at streak 0.
- **"void" in the contest ledger** is the contest's label for a miss at streak 0 (nothing to lose), not a missing entry. The 9/03, 9/07 and 9/13 rounds are void, yet each has two graded, entered slots (26 such rounds this season). The first version of this reader treated void as "not entered"; the evidence JSON uses the corrected rule (entered = graded slots present).
- **First live day.** The tail policy went live mid-afternoon on **9/03**, not 9/04 as the design's rollout planned: the fix deployed at 15:20 ET (reflog `eb010fd`), the first `Policy:` line is at 15:24, and the 9/03 double went out at 18:18. That day already meets the first-day acceptance fields.

## Disposition
**The mechanism behaved as designed on every date.** It chose the right goal from state alone. The only best streak it could stop on was trusted. It stopped exactly when 18 became unbeatable (9/19) and not a day earlier. Provenance is complete and consistent from the artifact through decision.json, the pick files and the journal. The delivered, entered and private streams stayed separate. What this does not establish, per the protocol: whether E[season-best] was the right objective, whether doubling at low streaks was optimal, or any rate. Register row X-16 records this read as *completed — mechanism validated, no rates*.

## Reproduce
On the box, from `~/projects/bts`: `.venv/bin/python scripts/audit/p05_tail_policy_audit.py --out /tmp/p05.json` (defaults to the frozen snapshot; read-only).
