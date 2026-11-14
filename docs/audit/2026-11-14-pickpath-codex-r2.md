# Verdict

**BLOCK** — final round, tier A. Reviewed HEAD `b7e83ef935fd8d806ca46432d5c2e9e5870a8233`.
The original counterexamples are repaired. B2 still loses delivery on the explicitly requested crash/restart interleaving; see C1.

# B1

**Closed for the reported defect.** The original probe and adopted regression pass: same-day replay preserves the applied increment, and the scorer's retry saves the terminal result without applying it twice.

Additional executed probes covered hits and saver-absorbed misses across 23:59, midnight, 02:00 and 07:40, both with and without a retry. Streak and saver remain correct. Without a retry the unresolved file still prevents reconstruction; the applied state is preserved, not silently repaired or declared terminal. This follows the existing incomplete-history rule and R12's stated limit.

I also forced both lock orders between a concurrent real daemon retry and real reconcile on the half-applied day. Both finish with one applied hit and a terminal pick; no stuck lock or rollback observed.

# B2

**Original delivery-loss case closed; crash recovery incomplete.** The original day-level probe now delivers the healthy primary as a single. Adoption strengthened it with single-file and decision-action assertions.

Executed additional checks for either leg surviving: the DM names only the survivor; the pick and decision contain one slot; the state locks; the live-forward trigger sees the saved single; the actual production snapshot loader reads one slot; cached shadow evaluation evaluates one slot. Both real scorers (`check-results` and daemon polling) add exactly one, and subsequent reconcile replay retains that increment. The focused kept-leg-cutoff regression also passes.

# N1 and N2

**N1 closed:** both the original delayed-request probe and adopted test now refuse the private lock after cutoff, without a scoreable decision.

**N2's original malformed wording is fixed:** postponed-game refusals no longer render "None min past the None cutoff"; legacy late refusals retain their wording. The new dropped-leg success wording has a separate NOTE below.

# New findings

## C1 — BLOCK: deleting the cached double before saving the single reintroduces a lost day on restart

**Location:** `src/bts/scheduler.py:2301–2307`, calling the unlink at `2210–2211`.

**Concrete state:** two distinct 19:10 games; cached double; DD game postponed; the primary remains playable until its 19:05 cutoff. At the 18:35 final fallback, `_keep_playable_leg` archives and unlinks the double. The process dies before `save_pick(single, ...)` publishes the replacement. Restart at 18:36; all prediction cascades return no predictions, as in the original B2 fallback scenario.

**Wrong production outcome:** no live pick file remains, so the restarted real `run_day` cannot attempt its cached fallback. It reaches tomorrow's wakeup with no pick DM despite the healthy primary still being enterable. Neither restart selection nor final fallback recovers the archived candidate. The new single conversion has no atomic replacement across this gap; R11 does not name losing the survivor on a conversion crash.

**Verified:** fault injected at the single-save boundary, followed by real restarted `run_day`, `run_single_check` and fallback control flow. Observed no pick file and no DM. A matched control retaining the original double at its live path until replacement delivers the healthy single at 18:36 under the same prediction failures.

**Smallest fix:** archive the double without unlinking its live file, then let `save_pick` atomically replace it with the single. Until that replacement succeeds, keep the old cached candidate available for the same reduction after restart. Preserve the helper's existing removal behavior for actual refusals/defer paths.

# Notes

- **N2 follow-up, verified production diagnostic defect:** `src/bts/health/late_delivery.py:57–60` claims the survivor "was delivered alone" solely from a record written before the send. With the single's DM raising `OSError`, delivery returns false and the saved single is undelivered, yet the real health check says "Kwan was delivered alone". Record reduction as reduction; claim delivery only from a confirmed delivery record. This is a NOTE, not an additional BLOCK. The new health-wording test plants a reduction record and asserts this premature success wording without establishing a send.
- **Shadow comparison, reading:** `_keep_playable_leg` returns a new DailyPick locally; `run_day` still holds the original double. When the original primary is dropped, the inline shadow trigger receives that removed primary's name (`scheduler.py:2935–2936`, `3130`, `3231`). `_run_shadow_prediction` uses it for the agreement log. Persisted pick readers, including the out-of-process shadow entry point, see the correct single. Reload the committed pick before supplying the inline comparison name.
- **WARN dispatch, reading:** the new WARN at `scheduler.py:2314–2318` is supplied in the dispatcher's `alerts` argument. That argument retains only CRITICAL alerts; without `warn_attention` or a prior failed-alert resurface there is no warning DM. Calling a mocked dispatcher in the test does not establish a real WARN send. If that DM is intended, supply it as WARN attention.
- Compared the adopted probes with `/tmp/test_pickpath_pp1_probes.py`: no behavioral assertion was weakened. B1 now checks preservation immediately after replay and the final hit; N1 checks no decision; B2 checks the actual single and decision action. All original counterexamples still execute their original paths.

# What I ran

- **145 passed:** `tests/test_pickpath_review_pp1.py`, `tests/test_postponed_delivery_guard.py`, `tests/test_streak_applied_dates.py`, `tests/test_incident_register_2026.py`, `tests/test_thin_slate_checks.py`, `tests/test_late_delivery_guard.py`, `tests/test_reconcile_receipt.py`.
- **All 4 original temporary probes passed** at this HEAD, including the guard-bypass control.
- **Additional probes: 11 passed, 2 failed as expected.** `/tmp/test_pickpath_pp2.py` covers midnight/retry/saver states, both concurrent lock orders, both surviving legs through both scorers and replay, production snapshot/shadow readers, the conversion-crash restart and its preserving-file control, and failed-send health wording. The two failures demonstrate C1 and the N2 follow-up.
- Reproduce the additional probes: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest /tmp/test_pickpath_pp2.py -q -p no:cacheprovider --tb=short`.
- Read the full top-commit diff, R11/R12, adopted tests, archive/save mechanics, score/replay locks, delivery/commit paths, health dispatch, and shadow/live-forward pick readers. Tests used synthetic feeds/files, controlled clocks and mocked request/DM leaves. No full suite, model training, account access, deployment, commit or push. No prohibited results directories read; source/tests unchanged.
