# Season rollover: where it stopped on 2026-11-14, and the brief for a fresh start

Written by the lead session after two review rounds ended in BLOCK from both reviewers (Codex at xhigh, Gemini 4 Carbon at high). Under the work policies that ends this loop: the obligations are recorded here and the next attempt starts fresh with something material changed.

## What landed
The data-path hardenings only (branch `sr-data`, files identical to the reviewed ones), after their own gate run:
- **S3** `features/compute.py`: the probable-pitcher scan skips non-numeric file stems (a weather sidecar raised).
- **S4** `data/pull.py::download_game_feed`: the body must parse as a JSON object; written through a temporary file and a rename.
- **S5** `data/build.py::build_season`: the pre-season no-op only while the season file does not exist; a season path that cannot be listed raises.

Both reviewers passed these in round 1 and raised nothing against them in round 2.

## What did not land, and where it is
| Branch | Content | State |
|---|---|---|
| `sr-state` (`8557acc`) | S1 (which streak the decision uses at a season boundary), S2 (its health check), S7 (the saver the policy receives), the malformed-pick guard | **blocked by both reviewers in round 2**; do not merge |
| `sr-lastcheck` (`d20f691`) | S6: extra attempts and a same-day alert for a day with no selected pick | Codex passed it in round 2; Carbon has one open finding (below); two existing tests needed updates that the gate caught |

Reports: `docs/audit/2026-11-14-season-rollover-{codex,carbon}-r{1,2}.md`. Briefs, handovers and both reviewers' probes: this folder (`briefs/`, `handovers/`, `probes/`, `probes-audit/`, `probes-r2/`). The probes are the raw material for the next attempt's RED tests.

## Why the season rule failed three times
The lead wrote three definitions of "which season does this saved contest observation belong to", and each fixed one state and exposed another:
1. **By the date of the latest settled round.** Refused a legitimate first fetch of a new season whose profile still listed last season's rounds, with no way to recover (found by the lead before review).
2. **By the fetch time.** Accepted last season's streak from an un-reset profile fetched this year (found by Carbon and by its audit).
3. **A seven-row table over fetch year, round year and whether any of our picks has settled this year.** Both reviewers broke it in round 2:
   - A restored box with no local pick history, mid-season, after a fetch whose rounds mapping failed: the table calls it a season start and returns streak 0 where the true streak is 13 (Codex B1).
   - An un-reset profile on day two: once our first pick has settled locally the table trusts last season's streak again, or (with a dated old round) refuses every pick with no recovery (Codex B1, Carbon B2).
   - A missing saver flag counted as "in reserve" whenever no local pick had settled, even with a contest streak of 13 (Codex B2, Carbon B1).

The common mistake: **inferring what MLB's account state is from the presence or absence of local files.** Local history can be missing (a restore) or present (a pick that never reached the contest), and neither says whether MLB has rolled the account over. Nobody on this project has observed what the profile and rounds endpoints return across a season boundary: the system first ran in April 2026.

## Open findings, all to be closed or explicitly ruled on
**Streak state and saver (`sr-state`):**
- Codex r2 B1, Carbon r2 B2: the provenance problem above. Codex: "this needs a contract correction, not different assertions around the present table."
- Codex r2 B2, Carbon r2 B1: an `uninitialized` saver must stay conservative once the contest season has started; the new-season reserve needs positive evidence of a season start. (The `not_earned` below 10 mapping is right and both reviewers confirmed it.)
- Codex r2 B3: a pick file with invalid UTF-8 still raises out of the history scan (`UnicodeDecodeError`); skip it like invalid JSON.
- Codex r2 B4, Carbon r2 B3: the health check applies the season rejection to an unexpired manual override, and reads the file's streak where the decision uses another; a `not_earned` saver at streak 10 to 15 gets no warning.
- Gate: `tests/test_orchestrator.py::TestRunAndPick::test_uses_fresh_contest_streak_for_live_action` fails with this branch merged. Neither author nor reviewer ran it.
- Carbon r2 notes: `update_streak` still increments last season's `streak.json` on the first result until the 02:00 reconcile; `decision.json` and the skip-policy shadow now record the policy's reserve value while the dashboard shows the persisted state; a clock-skew edge at the year boundary.

**Scheduler (`sr-lastcheck`):**
- Carbon r2 B4: a pick selected in an extra attempt and then refused because its game went void ends the extra attempts for later games that are still playable, and the day-end alert fires early.
- Both reviewers, note: a PLANNED check that overruns its pick's fallback deadline still runs a second refresh cascade and can lose the pick; the "reuse this check's cascade" branch written after the 2026-08-30 incident never runs, because `run_single_check` does not return its lock decision. Pre-existing. Enabling it for planned checks changes one existing scenario from a refusal to a delivery; that needs its own ruling and review.
- Carbon r2 note: a doubleheader's placeholder second game that is already final counts as playable for an extra attempt.

**From Carbon's audit (`docs/audit/2026-11-14-season-rollover-audit-carbon.md`), not yet started:** F4 (the first-fetch health exemption ends when Opening Night's pick settles), F5 (park-drag alarms and a crash on an empty reseeded store), F6 (in research mode the restored crontab still fetches the contest streak four times a day with a DM on failure and last season's cookies; a wrong DM handle in `deploy/systemd/bts-leaderboard.service`), F7 (a nine-batter Opening Night lock wedges the live-forward capture), F2 and F8 (dashboard saver season; shadow evaluations pooling last season's files).

## What to change this time
1. **The design comes from a design pass, not from another table by the lead.** Ask Carbon (it found every hole so far) for the decision design, with Codex attacking it, before any code: the full state space, the evidence each decision rests on, and what stays unknowable until March. The lead decides between the proposals and ledgers the ruling.
2. **Evidence at the source.** The fetch should record what it actually saw, so the decision does not have to infer it: settled rounds of this season, settled rounds from another season only, no settled rounds, or a rounds mapping that failed. A fetch that cannot establish current-season provenance must not overwrite an observation that had it.
3. **An operator override for the opening days.** The unexpired manual override already wins over the automatic file. Setting it to streak 0 for the first days of the season, while the fetch output is captured and read, removes the need to guess MLB's rollover behaviour in code. The code then only has to refuse a file that is plainly last season's. This is the lead's leading option; the design pass should try to refute it.
4. **The author runs every test that touches the changed modules,** found by grep, not a list from the lead. Twice the gate caught an existing test that neither author nor reviewer had run.
5. **Separate change sets, separately reviewed:** the streak state and saver; the scheduler's extra attempts (plus the ruling on planned-check reuse); the audit's F4 to F8.

## Not in dispute, keep
- Reading the saver's season from the current year in the decision, the health check and the dashboard; promoting `not_earned` to `active` in the fetch only on a current-season settled round.
- `not_earned` below streak 10 reaches the policy as "in reserve".
- A malformed (non-object) old pick file is skipped by the history scan.
- S6's shape after round 2: extra attempts only with no pick file and no recorded skip, one per remaining distinct start with a playable game, the attempt's own cascade reused, the alert dated by the slate and worded truthfully after a refusal.
