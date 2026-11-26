# Gemini as a second reviewer: two known-answer tests (2026-11-14)

**Question (Eric, 2026-11-13):** "Do you think the most recent Gemini agents with higher effort can also provide reviews?"

**Method.** Gemini 4 Hydrogen at XHigh effort, through the Antigravity CLI (`agy` 1.5.8, `--model gemini-4-hydrogen-xhigh`), in a fresh session per test, in an isolated copy of the repository at the commit Codex had reviewed (cloned from a bundle of that commit, so no later history or review reports were reachable). Each brief was the one Codex got, word for word except paths, plus one added instruction: write and run its own adversarial probes. Telemetry and AI-credit spending were switched off in the CLI first (both were found on). Briefs and the probe scripts it wrote: `docs/reviews/2026-11-14-gemini-known-answer/`.

## Test 1: the framing test's studio runner at `d2da185` (tier C, about 270 lines)

| Codex r1 finding (`2026-11-13-c2-framing-2026-studio-codex-r1.md`) | Gemini |
|---|---|
| B1 no catcher coverage still gets a disposition | found, demonstrated end to end with real LightGBM |
| B2 a date missing from every seed is accepted | found, demonstrated |
| B3 invalid profiles accepted; summaries unreconciled | found, demonstrated (split into two findings) |
| note: CPU total | found |

3 of 3 blocking findings. One extra: the aggregate ran with the 2026 input file missing. Report: `2026-11-14-gemini-known-answer-f26-report.md`. Its fast probe script was re-run by the lead and reproduces every claim. 36 minutes; about 2% of the weekly Gemini allowance.

## Test 2: the eight pick-path commits at `46e4e9e` (tier A, about 470 lines across the scheduler, grading, streak and cron)

| Codex pp1 r1 finding (`2026-11-14-pickpath-codex-r1.md`) | Gemini |
|---|---|
| B1 a hit lost across a crash, a same-day reconcile and the applied-date marker | **found**, demonstrated |
| B2 a postponed double-down leg discards the playable primary | **missed** |
| N1 private mode can lock past the cutoff after the status read | **missed** |
| N2 a postponement refusal rendered as "None min past the None cutoff" | **found** (rated blocking; also noted it names the batter) |

Findings Codex did not report in its two rounds on this range (the three later Codex rounds had narrower scopes), each re-run by the lead from Gemini's own probe script and confirmed:

| Gemini finding | Lead's check | Disposition |
|---|---|---|
| B3 `check-pick-entered` marks the day confirmed on a batter-only match before the unit capture binds the game; later runs only write `already_confirmed`; `bts day-outcome` then reports a valid entry as `entry_unverifiable` twice a day | reproduced | real; to fix (day-outcome must not be stricter than the entry checker's own confirmation) |
| B4 the nightly cron's `bts data build --seasons <this year>` raises until the season's first regular-season game exists, aborting the chain before `bts preview` and the R2 sync, every night from the box's restore to the opener | reproduced with the real CLI on a missing and an empty season directory | real; a regression from `1137cbe`; to fix |
| B5 on a two-game slate with separated starts, an early check selects a pick on the first game, the regular check then returns nothing, and that pick's fallback deadline is skipped because later checks exist | reproduced | real; the outcome (a later game's pick is delivered) equals the pre-change scheduler's; ledgered as a limit, not fixed now |
| note: `plan_run_times` can schedule two runs at the same minute (an early check beside a regular one) | reproduced | real; to fix |
| note: `bts day-outcome` reports `no_decision` on a day when every game is postponed, and at 23:50 on a doubleheader skip day whose end-of-day record is written at 23:55 | reproduced | real; to fix |
| notes on pre-existing grading edge cases and a reconcile-receipt label | not re-run | recorded for the daily batch review |

22 minutes; about 4% of the weekly Gemini allowance.

## Reading

- On a small tool with a written specification it matched Codex completely. On the cross-module pick path it found one of Codex's two blocking findings and one of its two notes, and five real defects Codex had not reported in its two rounds on that range. Codex in turn found two things Gemini missed. Neither is a superset of the other.
- October's test (Gemini 4 Argon at High, no probe instruction) found 0 of 4. What changed: a newer model, a higher effort setting, and an explicit instruction to run adversarial probes. This test cannot separate those three.
- Gemini rated as blocking some items a Codex reviewer and the lead would call notes. Its verdict is not usable as a gate by itself; its findings are leads that must be reproduced.
- Sample: two reviews, one project, one night.

## Decision (lead, under the review-cadence rule; Eric can overrule)

Gemini 4 Hydrogen at XHigh joins tier A reviews as a **second, parallel reviewer** beside Codex. Codex's verdict remains the gate. Every Gemini finding is reproduced by the lead or the author before it counts. Its brief always carries the instruction to write and run its own probes, and it always works in an isolated copy.
