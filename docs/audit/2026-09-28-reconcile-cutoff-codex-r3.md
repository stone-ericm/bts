## Round

r3 — reviewed HEAD 134eaf0 + the revised working tree, including the new cutoff test file and the documentation edits.

Verification: `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest tests/test_reconcile_cutoff.py tests/test_picks.py tests/test_scoring_lock.py tests/test_streak_replay.py -q` — **106 passed**, including all 18 cutoff cases. `git diff --check` passed. No network, production inspection, extra test files, or mutation runs. The additional edge-case conclusions below are from source inspection.

## Findings

1. **SHOULD — qualify the new incident entry's unconditional “no decision impact” claim.** `docs/audit/2026-season-wrap-index.md:22`.

   The entry says the same-day replay defect has “no decision impact” because policy state comes from the contest. That protection applies when contest state supplies the decision, or when its absence blocks the caller. It is not a property of every supported invocation.

   **Failure scenario:** in a picks directory without a contest observation, a daytime reconcile removes today's already-settled hit from the local streak. `load_decision_streak_state` defaults to `require_contest_state=False` and then returns that altered model streak (`src/bts/contest_state.py:282`, `src/bts/contest_state.py:298-315`). The preview command uses this default and passes the resulting streak to selection (`src/bts/cli.py:1318-1327`). Selection can therefore receive the wrong streak; whether it changes the chosen action depends on the policy state. This is the previously identified fallback path, not a new runtime defect or an observed production incident.

   **Suggested fix:** replace the parenthetical with: “Contest-backed decisions do not use the local replay streak; callers allowing model-state fallback can be affected.” Keep the local streak/saver repair tracked. No change to the cutoff guard is required for this documentation correction.

**Disposition and focused checks**

- **r2 #2 fixed.** `src/bts/picks.py:1180-1181` reads the replay date while holding `scoring_lock` and derives both the season and exclusive date boundary from that same reading. The new August midnight-lock test passes; reverting to the entry-time boundary would exclude its planted hit and fail the streak assertion.
- **Dec 31 → Jan 1:** the replay arguments become the new year and January 1 of that year together. With no earlier picks in the new year, the existing replay helper returns local streak 0 and saver available (`src/bts/picks.py:598-620`). This matches a fresh January 1 invocation under the existing calendar-year replay convention; it does not mix the old season with the new boundary. No separate New Year fixture was executed. The “include the day that just ended” comment has the natural qualification that it belongs to the selected replay season.
- **Fresh reading past 08:00:** the added clock read occurs after the proposal loop. It cannot revive a discarded late answer or cause another MLB fetch. A correction admitted and applied before cutoff remains available to the local replay. The existing response-time and locked mutation checks remain intact (`src/bts/picks.py:1139`, `src/bts/picks.py:1150`). No new cutoff bypass found.
- **r2 #1 substantially closed.** `ARCHITECTURE.md:271` and C-03 (`docs/audit/2026-09-corrections-index.md:9`) now state the two scheduled attempts, best-effort observation, unobserved late-morning corrections, ambiguous empty correction lists, and exclusion of nonterminal picks. C-03 records observation receipts, missed-check visibility, authoritative entered-day finalization, and cron installation as 2027 follow-ups. These are accurately presented as outstanding work. The same-day replay omission is now recorded in W1.5; the qualification in finding 1 is still needed.
- **Remaining scope:** the signature continues to cover prevention of late MLB overwrites, not complete observation through 08:00 or authoritative contest finalization. The pre-existing same-day replay omission remains. No other new runtime defect was found in the final diff.

## Verdict

SIGN — the replay-boundary repair is correct and the residual contract is now documented; only the nonblocking decision-impact wording needs qualification.
