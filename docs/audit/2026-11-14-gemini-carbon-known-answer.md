# Known-answer test: Gemini 4 Carbon (High) as a tier A reviewer (2026-11-14)

Eric upgraded the Google subscription on 2026-11-14 ("I just upgraded my Gemini subscription, which I think includes access to a better model"). The Antigravity CLI (1.5.9) now reports the account as Google AI Ultra and lists Gemini 4 Carbon at low, medium and high effort. Carbon is Google's top model and is offered in the CLI only on Ultra and enterprise plans; Hydrogen, the parallel tier A reviewer since ruling R15, is a faster model distilled from it (sources at the end).

Ruling R15 rests on two known-answer tests of Hydrogen. This is the same test for Carbon before it replaces Hydrogen in that role.

## Design
- **Input:** commit `dbb78b1`, the version of the g2 fixes that both reviewers blocked in round 2, in an isolated copy cloned from a bundle of that commit: no later history and no review reports.
- **Brief:** the round-2 brief Hydrogen received (`docs/reviews/2026-11-14-g2-fix/gemini-prompt-r2.md`), adapted for a reviewer that did not take part in round 1 (the round-1 blocking findings are summarised in it, as they were for Hydrogen). Saved as `docs/reviews/2026-11-14-gemini-carbon/prompt.md`.
- **Model:** `gemini-4-carbon-high`. Settings read before the brief was sent, after the plan change: Enable Telemetry off, Use AI Credits off, Non-Workspace Access off, Sandbox Mode on.
- **Handicap:** Hydrogen came to round 2 having already traced this code in round 1; Carbon starts cold.

## Answer key, committed at 11:30 ET before Carbon's report existed
What reviewers found on this commit or on this code path, each confirmed by the lead:

| # | Finding | Found by |
|---|---|---|
| K1 | **Blocking.** The builder swallows malformed feed shapes on which the base raised (JSON list or null, `gameData: null`, `game: null`, a list or dict `type`); beside a regular-season feed it writes and publishes a partial season file | Codex xhigh round 2; Hydrogen round 2 |
| K2 | **Blocking for the deploy gate.** The parity test's fixture runs `git show main:…`: errors without a local `main` branch, compares the file with itself once merged | Hydrogen round 2 |
| K3 | Last season's `contest_streak.json` / `streak.json` supply the streak on Opening Night; a prior-season observation can read fresh | Codex round 2; Hydrogen round 2 |
| K4 | Lineups that post after the last check leave no pick on a team's first game: no fallback without a pick file, no alert | the final Codex reviewer only (one commit later; the code path is the same here) |
| K5 | A weather sidecar file breaks the feature scan (`int(f.stem)`) | Codex round 2; the final Codex reviewer |
| K6 | A legitimate first fetch of a season (`source_date` missing) raises a CRITICAL health alert | Codex round 2; Hydrogen round 2 |
| K7 | On a team's first game it has no rows until its lineup posts, and the early lock reads that as "all confirmed" | Hydrogen round 2 |
| K8 | False green: the cold-start tests in `tests/test_refresh_memo.py` stub training and prediction | Codex round 2; Hydrogen round 2; the final Codex reviewer |
| K9 | A readable but unknown game-type string returns the no-op (accepted by ruling R21) | Hydrogen round 2 |

Scoring: K1 and K2 are the blockers a reviewer had to find. K3 to K9 measure depth on the "what fails next" question. Anything outside the key is reproduced before it counts, as with every Gemini finding.

## Result
(to be filled in when the report arrives)
