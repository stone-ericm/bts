# Change set E (operator texts) STOPPED after round 2 (2026-11-24 21:09)

Session `bts-wr-fresh9`. Branch `fix/wr-e` at `256ac958` (pushed, NOT merged): contract `kickoff-fix-e.md` (version 2, amendments 1 and 2), contract check `contract-check-e/` (Codex xhigh, `WRCHKE-2dd1d06c`, BUILD WITH THESE EDITS), code commits `db52bd02` and `256ac958`, the document rows in the session's commits. `main`'s `src/`, `docs/ops/2027-season-start.md` and `INCIDENT.md` are unchanged. Under R51 a second-round BLOCK is not overridden and the session builds no follow-up of its own (R57): the lead rules.

## The rounds
| Round | Codex xhigh (the gate) | Carbon High |
|---|---|---|
| 1 on `db52bd02` | **BLOCK** `WREREV1-2f4ce4d0` (`review-fix-e-codex/`): 8 findings | **BLOCK** `WRECARB1-45cfd488` (`review-fix-e-carbon/`): 6 findings, 3 shared with Codex |
| 2 on `256ac958` | **BLOCK** `WREREV2-cf907e69` (`review-fix-e-codex-r2/`): 7 new findings; all 14 round-1 findings confirmed closed on their own inputs | see "Carbon, round 2" at the end |

Gate on `db52bd02`: 10,814 passed, failing set exactly the pinned 53. The gate on `256ac958` was queued and CANCELLED before it started when the BLOCK arrived (the hold was removed; no gate result exists for the round-2 commit; its focused tests: 313 passed in the reviewer's one-process run).

## What held in both rounds (three independent reviewers)
- Every rendered message matches the composed contract character for character (five skip cases on three surfaces, both stale warnings, both refusals); lengths 881 and 882 characters with the envelope.
- Words only: against the base, whether a summary exists, every other field of `run_single_check`'s return, the once-a-day notice, the missed-pick gate, the warnings' level, source, key, status file, date, dedup and interrupt boundaries, the refusals' archive, record and sources, and `effective_pick_bar` over the whole shipped table in both regimes are unchanged (380 frame and candidate combinations, 23,884 bar comparisons; Carbon 20,160 and 25,908). `strategy.py` has no diff.
- Renderers do not raise on decoded bad data; the banner is escaped; only case K2 says `below the`.
- The season walk on the commit: 4,599 runs, no wording problem, actions, bars and day counts equal to the saved records; the 841 formerly flagged notices are 229 K3 and 612 K4.
- Round 2: all 16 scratch changes are noticed by `tests/test_wr_e_texts.py`.

## Round 2's blocking findings (Codex; none reproduced by the session: the stop came first)
| # | Finding | Kind | Reviewer's fix |
|---|---|---|---|
| B1 | `INCIDENT.md` §4 (the round-1 Carbon fix): the unit installer replaces a held writer's mask and reloads at once, so the mask cannot be re-applied "before `daemon-reload`" | document sentence written in round 1 | "For an offseason snapshot, follow B8c steps 1–7. Restore only the scheduler and dashboard unit files in step 3, with all other B8c holds in place." |
| B2 | Cookie recovery (`INCIDENT.md` §3, checklist A4): `uv run python` in a synced Mac checkout has no Playwright; on the box a readable `pass` entry or `BTS_LEADERBOARD_COOKIE_FILE` takes precedence over the file the text names | document sentence written in round 1 | name the capture environment and "the cookie source the box actually reads" |
| B3 | `INCIDENT.md` §2: "Apply the current `deploy` SHA again" has no quiet-window condition (§5 has it) | document sentence written in round 1 | add the condition |
| B4 | The morning stale warning: the same command filed after the first lineup check has started can be accepted while a pick chosen on the stale state is still delivered | message text (`scheduler.py`) and B8a | "Before the first check, if prior entries are settled or absent and MLB differs …"; B8a: "If the first lineup check has started, have Eric verify any pending or delivered pick against MLB before acting on it." |
| B5 | The dashboard heading `NO PICK AT LAST CHECK` stays over an earlier check's summary when a later check produced no predictions (the stored summary is kept on purpose) | heading text (`web.py`; came from the contract check) | `LAST RECORDED NO-PICK SUMMARY` |
| B6 | `build_skip_summary`: `isinstance(candidate, Mapping)` sits outside the `try`; a candidate whose `__class__` raises escapes (Carbon's round-1 note N4, not fixed in round 2; the real selector passes a dict) | code, one line | move it inside the `try` |
| B7 | `INCIDENT.md` §5 normal ship: after a merge elsewhere, a stale local `main` pushes nothing new | pre-existing sentence, unchanged by E | "Switch to local `main`, fast-forward it from `origin/main`, and confirm it equals the approved merged SHA before pushing." |

## Assessment (the session's reading)
1. **The code rows are close:** three of the seven are in the code commit, each a one-line change with the reviewer's own text (B4's sentence, B5's heading, B6's `try`). Nothing in either round touched behaviour.
2. **The documents are where it does not converge.** Round 1 found 11 document faults, round 2 found 4 more, 3 of them in sentences written to fix round 1, and each fix reached further into procedures nobody has rehearsed (the provider rescue boot, masks, `pass` against the cookie file, dispatch after a rollback). E's contract opened two whole documents ("one consistent order that can be followed") to a per-sentence worst-state review, and every reviewer finds another real gap in a box procedure that has never been run. The findings are true; the scope has no natural end. This is the pattern of D and D2 (text-level proofs of box state).
3. **My own part:** the contract let the checklist and `INCIDENT.md` be rewritten in the same change set as five message strings; a narrower contract (the messages, plus only the checklist sentences that quote them) would probably have signed in two rounds. I also adopted two round-1 replacement sentences from Carbon (the unit installer, the cookie command) after confirming them only by reading, and round 2 blocked both.

## Ways forward (for the lead; R51, R57)
- **Ea, the messages only (recommended):** a small follow-up on `256ac958` with B4's sentence, B5's heading and B6's `try`, the checklist limited to the B8a sentences that restate those messages, and `INCIDENT.md` and the other checklist rows REVERTED to `main` in that piece; one fresh Codex verdict on the changed material. It closes SW1's text, BL-N3, C2-B1's text, BL-B4 and BL-B5.
- **Eb, the documents, as their own piece (tier D text reviewed per sentence), or deferred to January with the Debian rehearsal:** BL-B6, BL-text, BL-B3 and C2-O1/BL-N4 are about the first boot after the restore, which is the thing January's rehearsal runs for real. The branch holds the full reviewed text and both rounds' findings as the starting point. Until then the checklist on `main` keeps its known contradictions (closing-test table `closing-test/codex-blind/operator-text.md`).
- **Or Eric's acceptance** of round 2's findings as limits (not recommended for B1 and B2: the text as written cannot be followed).

## Obligations if E continues
Round 2's seven fixes as written above; a gate on the final commit (none was run on `256ac958`); the season walk re-run; `CLAUDE.md` and the `FINDINGS.md` state table updated only on a merge.

## Carbon, round 2
Still running when this record was written (21:09); its verdict is appended when it reports. It cannot change the stop: Codex's verdict is the gate.

**22:07: Carbon High, round 2 on `256ac958`: BLOCK** `WRECARB2-fab09611` (`review-fix-e-carbon-r2/`; not reproduced by the session). The runtime code matched the contract with no change beyond text; three findings, all in `INCIDENT.md` and the checklist (the unit installer and the masks, as Codex B1; the cookie instructions against a synced Mac checkout and checklist A4, as Codex B2; and one more runbook instruction). It agrees with the assessment above: the code rows held, the documents did not converge.
