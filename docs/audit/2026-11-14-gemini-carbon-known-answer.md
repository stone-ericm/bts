# Known-answer test: Gemini 4 Carbon (High) as a tier A reviewer (2026-11-14)

Eric upgraded the Google subscription on 2026-11-14 ("I just upgraded my Gemini subscription, which I think includes access to a better model"). The Antigravity CLI (1.5.9) now reports the account as Google AI Ultra and lists Gemini 4 Carbon at low, medium and high effort. Carbon is Google's top model and is offered in the CLI only on Ultra and enterprise plans; Hydrogen, the parallel tier A reviewer since ruling R15, is a faster model distilled from it (sources at the end; the distillation claim is from third-party write-ups of Google's announcement).

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
Report: `docs/audit/2026-11-14-gemini-carbon-known-answer-report.md` (verbatim; marker `AGYKA3-5047ae11` equals the file's hash). One pass, 39 minutes (11:29 to 12:08 ET). Verdict BLOCK.

| # | Carbon (High), cold | Hydrogen (XHigh), round 2, warm | Codex xhigh, round 2 | Final Codex xhigh, fresh, one commit later |
|---|---|---|---|---|
| K1 partial build beside a regular feed | **found**, with the fix that was later adopted | found | found | (already fixed) |
| K2 `git show main:` fixture | **found** | found | missed | (already fixed) |
| K3 last season's streak on Opening Night | **found** (both files) | found | found | not reported |
| K4 late lineups leave no pick | **found**, by running the real day loop | missed | missed | found |
| K5 weather sidecar breaks the feature scan | **found** | missed | found | found |
| K6 first-fetch health alarm | **found** | found | found | not reported |
| K7 first-game early lock | **found** | found | missed | not reported |
| K8 cold-start tests stub training | **found** | found | found | found |
| K9 unknown type string is a no-op | missed | found | missed | not reported |
| **Total of 9** | **8** | 7 | 5 | 3 of the 7 still open |

Both blockers found. Carbon is the only single pass that found K4, and it found in one cold pass what three reviewer passes had found between them, except the accepted design note K9.

**Outside the key, reproduced by the lead on current `main` (`ba1337a`):** an unreadable `data/raw/<season>` directory (`chmod 000`, even with regular-season feeds inside) or a regular file in its place makes `build_season` return the no-op, where the base raised. `Path.glob` swallows the `OSError`. Neither Codex (three rounds, the last by a fresh reviewer) nor Hydrogen (three rounds) reported it. It is in the landed code; the state needs a permission or mount fault, and the outcome is silent: predictions run without the current season and the nightly chain publishes. Added to the follow-up brief as part of S5 (`docs/reviews/2026-11-14-season-rollover/kickoff-draft.md`; probe `probes/lead_unreadable_season_dir.py`).

**Outside the key, not reproduced (notes):** the park-drag producer crashing on an empty new-season file; `predict.py` reading the schedule with a bare `urlopen`; a thin-slate test whose docstring claims more than it runs.

**Cost on the Ultra plan:** the 5-hour Gemini allowance went from 100% to 92% and the weekly one from 100% to 98%.

## Decision (ruling R22)
Gemini 4 Carbon at High replaces Hydrogen at XHigh as the second model family on tier A reviews (`agy --model gemini-4-carbon-high`). Everything else in R15 stands: it runs in an isolated copy, in parallel with Codex; Codex's verdict is the gate; every Gemini finding is reproduced before it is acted on; `/config` must read Enable Telemetry off and Use AI Credits off before each use. Hydrogen XHigh is the fallback when Carbon's allowance is short.

One test, on one commit, with a brief that named the area to look in. It does not show that Carbon is the better gate; it shows that it is at least as good a second reviewer as the one it replaces, at a cost that allows it on every tier A change.

## Sources
- [Gemini 4 Carbon is now available (Google)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-4-carbon-now-available/)
- [Choose a model (Antigravity docs)](https://antigravity.google/docs/models/)
- `agy models` and the CLI header on 2026-11-14 ("Google AI Ultra", "Gemini 4 Carbon (High)")
