# Design pass, part 1: which streak and which saver the decision uses at a season boundary

You are the designer. No code is to be written or changed in this pass. Ruling R23 (roadmap ledger) stopped the previous loop after the lead wrote three rules that each failed review; this pass exists so that the fourth design does not come from the lead. A second agent will attack your design before the lead decides.

## What the system does
Beat the Streak picks are made by a scheduler on a production box. The pick decision takes a streak, a "saver" state and the days left in the season, and looks up an action (skip, single, double) in a policy table. The streak comes from `contest_state.load_decision_streak_state` (`src/bts/contest_state.py`), which prefers an unexpired manual override file, then the automatic observation written by the fetch (`src/bts/contest_fetch.py` writes `data/picks/account_state/contest_streak.json` from the real contest account's profile and rounds endpoints), then local state (`data/picks/streak.json`). The saver is in `src/bts/saver_state.py`. Health checks for this state are in `src/bts/health/contest_state.py`.

The system first ran in April 2026. **Nobody on this project has observed what the profile and rounds endpoints return across a season boundary**, and nobody will before March 2027: the account is real, requests to it are the owner's to authorise, and the box is switched off until 21 days before a confirmed opener. Opening Night is 2027-03-24 (one game), the traditional opener 03-25.

## The problem
On Opening Night the files on the restored box are last season's. Nothing ties a saved streak or saver to a season. A wrong streak changes the action (the policy differs by streak, and the tracked policy differs on the saver at streaks 7 to 9). The failure directions, in the owner's order of cost: refusing every pick with no way to recover; picking on a stale non-zero streak; treating a saver as spent or as in reserve wrongly.

## Read first, in this order
1. `docs/reviews/2026-11-14-season-rollover/FRESH-START.md`: what landed, what did not, why the three rules failed, the open findings, what must change, what is not in dispute.
2. The two round-2 reports that broke the third rule: `docs/audit/2026-11-14-season-rollover-codex-r2.md`, `docs/audit/2026-11-14-season-rollover-carbon-r2.md`; and the audit `docs/audit/2026-11-14-season-rollover-audit-carbon.md` (F1 to F4 concern this design) with `docs/audit/2026-11-14-season-bound-table.md`.
3. The blocked attempt, for what not to repeat: branch `sr-state` (`git diff main...sr-state -- src/`), and its contract `docs/reviews/2026-11-14-season-rollover/briefs/kickoff-a4.md`.
4. The code as it is on `main`: the three modules above, `src/bts/strategy.py` (`SEASON_END_DATES`), the callers of `load_decision_streak_state`, and how `streak.json` and `applied_dates` are updated (`update_streak`, the 02:00 reconcile).
5. `docs/ops/2027-season-start.md` rows B4 to B8 (what the operator does before the first pick) and `CLAUDE.md` section "2027 season".

## What to deliver
Write `design-carbon.md` in the repository root of this copy. Plain prose and tables; no code beyond a signature or a file format where the design fixes one.

1. **State space.** Every combination that can exist on a day the decision runs, built from what is actually observable: which files exist, what each says, when it was written, what the fetch saw on its last successful and last attempted run, whether an operator override exists, the calendar date against the season's first game. Include the states the three failed rules mishandled (a restored box with no local history mid-season; a profile that has not reset by day two; a first fetch of the season that lists only last season's rounds; a failed rounds mapping; clock skew at the year boundary) and any others you find. Say which states you believe are reachable and why.
2. **Evidence.** For each decision the design makes, the evidence it rests on, and whether that evidence is recorded at its source or inferred. FRESH-START names the common mistake: inferring what the contest's account state is from the presence or absence of local files. Do not repeat it. If the fetch should record more of what it saw, specify the fields and their meaning.
3. **The decision table.** For every state: the streak used, the saver passed to the policy, whether a pick is made, what alert fires and when, and how the operator recovers. No state may end in "refuse forever". Mark each row with what happens if the row's assumption about the contest's behaviour is wrong.
4. **What stays unknowable until March,** and how the design behaves under each possibility (for example: the profile resets before Opening Night, on it, some days after, or keeps a carried streak field with a separate season marker).
5. **The lead's leading option, to be refuted or improved:** an operator override to streak 0 for the opening days (the unexpired manual override already wins over the automatic file), fetch-side provenance for everything after, and code that only refuses a file that is plainly last season's. State the strongest case against it: what it costs if the operator forgets it, sets it late, or leaves it past the first hit; what it does to the saver; whether it can be made fail-safe without becoming "refuse forever".
6. **At least one alternative** that differs in kind, with the same table, and your recommendation between them with the cost of being wrong for each.
7. **Tests the design implies:** the RED tests an implementer would write first, as one line each (state, expected outcome), drawn from the state space and from both reviewers' probes in `docs/reviews/2026-11-14-season-rollover/probes*/`.
8. **Split into change sets** that can be reviewed separately, smallest first, and which of them must be live before the first pick of 2027.

## Rules
- Read the real code for every claim about current behaviour, and say where you ran something. You may write and run probes in a scratch folder inside this copy; do not change tracked files.
- Do not contact the contest, the box or any network service. Everything you need is in this copy.
- Where you are unsure, say "unsure" and what would settle it. A design with honest unknowns is worth more than a complete-looking one.
- Keep it under 400 lines.

When the file is complete, print `SRD1-<first 8 hex characters of the sha256 of design-carbon.md>` and then `DONE`.
