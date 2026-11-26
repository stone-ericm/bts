# Verdict

**BLOCK** on `988708df`. The revised later-slate procedure can leave a known wrong streak in use for the next lineup check after settlement. Review scope was the changed text and the `1ed57c47..988708df` source delta.

## Blocking findings

- **B1 · `docs/ops/2027-season-start.md:31`, B8a “Then compare MLB with status…”**
  **Harmful outcome:** Following the revised wait and handoff sends a wrong double-down at 17:05 ET on July 17, on automatic streak 0, when settled MLB state at streak 15 calls for a skip.
  A July 15 warning is read July 17 at 09:30. Its pick settled as a hit, MLB shows `15/15/used`, and the July 16 entry is pending in a suspended game. The revised wait runs past the 16:05 first check, which produces no pick. The pending entry settles as a Pass at 16:08, so MLB still shows `15/15/used`. At 16:10 Eric has no pending or delivered pick to verify. The revised handoff files nothing, and the stale automatic state stays eligible. The original text allowed the 09:30 filing in this particular state. On two confirmed 80% candidates, the real second check and delivery sent a double-down only under the revised instructions. A real dated write at 16:10 also produced a skip, while preserving the requirement to wait for settlement. [Differential replay](probe-late-settlement.log)
  **Replacement for the compare-and-file sentence:** “Then compare MLB with status and, if they differ, run `bts set-contest-streak --date TODAY --streak N --best-streak B --saver S --picks-dir DIR` with TODAY set to the current ET date before its first lineup check. If that check has started, have Eric verify MLB and any pending or delivered pick before acting on the pick. If the state differs while the slate is unlocked and a later lineup check or fallback refresh remains, have Eric run the same command between checks.”

## Edits

- **E1 · `src/bts/scheduler.py:95` and `docs/ops/2027-season-start.md:31`.** The warning date can be the next day after a midnight send. Its current sentence does not name the delivered pick as the slate reference, and “absent” can be read before closure. B8a's later-slate paragraph resolves both in the full procedure. Following that paragraph at 00:05 with the prior entry window open prevented a write. The two shorter instructions can still be read ambiguously. [Replay](probe-absence.log)
  DM old “Write for the {D} slate only on that ET date, with prior entries settled or absent.” Replace with “Write only on the pick's ET slate date once prior entries settle or their windows close empty.” This projects to 896 characters in the longest envelope.
  B8a old “On a warning after a delivery: write that command only on D ET for the D slate, after MLB has settled all prior entries or confirmed their absence after their entry windows closed.” Replace with “On a warning after a delivery: use the warning's command only if the delivered pick's slate and today's ET date both equal D, and MLB has settled every prior entry or confirmed its absence after its entry window closed.”
- **E2 · `src/bts/scheduler.py:1767`.** `Reason: Model ID below themodel.` renders as `Reason: Model ID under themodel.` The renderer changes a prefix in a longer word, altering display text with no operator action. Replace `return " ".join(value.splitlines()).replace("below the", "under the") if isinstance(value, str) else "?"` with `return re.sub(r"\bbelow the\b", "under the", " ".join(value.splitlines())) if isinstance(value, str) else "?"` and add `import re`. [Display replay](probe-display-surfaces.log)

## Notes

- Reading the warning for the first time after the first check, with all prior entries already settled and no pick yet, could leave stale state under either revision. The blocking replay instead starts before the first check and isolates a delay introduced by the new all-prior-entries condition. [After-check control](probe-contest-replay.log)
- MLB match, pending and Pass states were synthetic operator inputs. The CLI, state loading, policy checks, and captured delivery path were real. December and January were synthetic clock controls.
- `nice` reported a priority-setting warning and the commands continued. Initial probe runs exposed fixture issues with cutoff isolation, Markdown cell parsing, a carried legacy bar and pytest assertion formatting. The retained follow-up runs supplied the evidence below.

## 1. C22 to C27

| Row | Replay and outcome |
|---|---|
| C22 | July 15 warning read July 17 at 09:30. MLB shows 13 while July 16 is pending. Waiting, then reading real status after the fetch, kept automatic 14 for selection. Filing under the original wait left dated 13 in use. The late Pass variant above exposed B1. [Replay](probe-contest-replay.log) |
| C23 | July 15 11:40 delivery DM and B8a both condition filing on prior settlement or absence. On July 14 pending, waiting left no manual row. After settlement real fetch and status showed 14. The obsolete early filing kept 13. A separate permitted correction wrote dated `14/15/active` and preserved the bytes of the sent pick and saver flag. [Replay](probe-contest-replay.log) |
| C24 | A July 14 delivery warning read July 15 at 08:30 and a July 14 morning warning read July 16 both allowed `saver-state --use` for MLB's spent 2027 saver. After the July 15 dated row expired, the July 16 automatic decision at streak 14 and 79% skipped with `used`; the `active` control chose a single. [Replay](probe-saver.log) |
| C25 | After a send crossed midnight, July 14 as `--date` exited 2; July 15 and absolute `--picks-dir` wrote only to the scheduler directory from another cwd. After 16:05, the rule directs verification, but an empty first check leaves no pick to verify. A between-check write was effective and changed double to skip. [Replay](probe-contest-replay.log) |
| C26 | `63669c42` created a real no-decision summary without `basis`; same-day restart showed the recorded suffix and retained the notice timestamp. All three current surfaces rendered the replacement text in stored batter, team and reason. B8a's opening matched the contract and cutoff still refused stale delivery. [Display](probe-display-surfaces.log) · [B8a text](probe-display-replay.log) · [Cutoff](probe-contest-replay.log) |
| C27 | All five scratch changes caused test assertion failures. See item 4. [Results](probe-mutations-replay.log) |

## 2. Each changed sentence

Visible B8a text matched the contract character for character after removing Markdown delimiters. Both warnings and the skip strings matched the contract in the rendered captures. [Text extraction](probe-display-replay.log) · [Warning comparison](probe-parity.log) · [Skip surfaces](probe-display-surfaces.log)

| Sentence | Worst state and result |
|---|---|
| Delivery “Write for the {D} slate…” | Previous entry pending at 11:40 prevented a premature dated write. A transport crossing midnight printed the next date while the saved slate stayed July 14. At 00:05 the applicable B8a sentence required the still-open prior window to close. See E1. |
| Delivery “In {Y} ET…” | One-day-late read fixed the spent flag. A synthetic December 31 send confirmed January 1 used 2028, its warning year. MLB's 2027 usage gave no instruction to write a 2028 flag. |
| Morning “In {Y} ET…” | Two-day-late read fixed the flag despite automatic fetch recovery. The December 31 warning did not authorize January 1 usage. A fresh January 1 warning printed 2028. |
| Skip “Streak recorded with this check…” | Legacy same-day restart, missing or invalid `basis`, printed the stored streak without asserting it was used. Valid decision printed `used`; `no_decision`, invalid streak and absent streak omitted the suffix. |
| Stored `below the` replacement | Real skip journal, DM and banner showed `Observations under the expected count` and rendered the stored batter and team likewise. K2 still printed its comparison. A longer-word prefix also changed. See E2. |
| B8a “After the switch, live delivery remains eligible…” | Real accepted stale state delivered with guards satisfied. At the exact cutoff, a separate replay refused with no pick transport. |
| B8a “On a warning after a delivery…” | On D with prior entries pending there was no write. Once settled, a supported differing observation left the pick and flag intact. See E1 for the D/slate wording. |
| B8a “The sent pick and saver flag stay as they were.” | Before and after exact CLI filing, both persisted files were byte-identical. |
| B8a “For any date later than the pick's slate…” | The intervening entry now counts. At 00:05 a missing entry was insufficient while its window remained open. At 08:30 confirmed absence allowed the CLI write. The in-effect dated row then governed comparison even though automatic state still differed. |
| B8a “Then compare MLB with status…” | Absolute DIR worked from another cwd before the first check. When the first check had run and a prior entry only then settled, the empty handoff let the next check send on stale state. See B1. |
| B8a “In D's ET year away from midnight…” | The same-year and cross-year outcomes agreed with both DMs. `active` and `not_earned` became `used`; `uninitialized` took explicit current-year `--init used`; `used` required no write. An initialization race was resolved by rereading and applying `--use`. |

## 3. Source delta and warning lengths

`git diff 1ed57c47 988708df -- src` touches only `src/bts/scheduler.py`. Reconstructing HEAD from the prior source via the exact rendering substitutions gave a byte match. Old and new traces matched WARN level, `contest_state` source, incident key, dedicated receipt path, ET date, failure/retry and dedup. Delivery contained the warning interrupt and morning propagated it on both. Captured pick DM and pick, decision and scheduler JSON matched. Notice timing, restart suppression and stored summary fields matched for legacy, no-decision and decision inputs. The body digest changed with the words. [Warning parity](probe-parity.log) · [Skip notice parity](probe-skip-parity.log)

| Real envelope | Old timestamp, longest | Future timestamp | Unusable timestamp |
|---|---:|---:|---:|
| Delivery | **892** | 876 | 743 |
| Morning | **889** | 873 | 740 |

The stored timestamp is capped at 128 characters and the old-age rendering has at most eight integer digits. Neither warning sends for a fresh record. The WARN-only receipt retried without a failed-CRITICAL banner. [Measurements](probe-parity.log)

## 4. Tests and scratch changes

- `tests/test_wr_e_texts.py` 63 passed. `tests/test_stale_delivery_cs4c.py` and `tests/test_morning_state_cs6b1.py` 123 passed. `tests/test_scheduler_skip_visibility.py`, `tests/test_scheduler_tail_objective.py` and `tests/scripts/season_walk/` 66 passed. [Texts](tests-texts.log) · [Warnings](tests-warnings.log) · [Skip and walk](tests-skip-walk.log)
- Restored the old delivery slate sentence, old delivery saver date, old morning saver date, legacy `used` suffix, and unchanged stored phrase one at a time in scratch `src/` trees. Targeted tests detected each with 5, 5, 3, 10 and 1 assertion failures respectively. Zero escaped. [Scratch results](probe-mutations-replay.log)

ran: C22–C27 offline replays, prior-revision and exact-text comparisons, warning length and delivery checks, six requested test targets (252 tests), five one-at-a-time scratch changes; did not run: network calls, the shared wrapper, the full fast suite, or a fresh full season-walk CLI run.
