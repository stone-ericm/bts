# Whole-range fix set B, round 2

## Verdict

**SIGN WITH EDITS** on `fe593f29` against `9e0ec28a`. Both round-1 production blockers are resolved. Add a regression assertion that exercises the automatic decision's saver read.

## Blocking findings

None in production.

## Notes

- The saver parameter of `tests/test_wr_b_edges.py:215` interrupts the first saver read, in status diagnostics. The decision can skip the saver read and still pass all 99 mutation tests. With a current switch, valid automatic and dated observations, and a 5,000-digit JSON integer in `saver_state.json`, the real CLI accepts today’s dated path and refuses the automatic fallback. The ordinary required loader raises `ValueError` with the dated file absent. The saver-skipping mutant prints automatic accepted in that same state. Add this case to the verdict parity matrix, or interrupt the saver read inside the decision after diagnostics complete. See `saver-value-*.log`.

## 1. Default pick-path loader

- The diff against base in `src/bts/contest_state.py` and `src/bts/saver_state.py` contains only `omit_selected_dated` and the `switch` pass-through. The saver loader matches base. Status needs the captured switch to keep a single open and apply the current ET year when `--season` displays a different year.
- The adapted round-1 matrix produced byte-identical results for **2,112 states** at four pinned ET instants, including return values, exception types, messages, arguments and causes. The original normalization of the B1 warning text is the only comparison mask. Both revisions reached the six FIFO reads.
- Audited every production caller in the CLI, scheduler, orchestrator, dashboard, health and saver paths. Only status supplies a switch snapshot to the required loader, and only its fallback supplies `omit_selected_dated=True`. The loader forwards those options internally. The nonblocking helper has one production caller, the B5 fetch-status decoder.

## 2. Status and the ordinary required loader

- **24 ordinary-file fixtures** matched the complete independent verdict, including the first line of each refusal and recorded-time display. Covered four dated/automatic combinations, a winning legacy override, expired dated state, corrupt manual/auto/switch, wrong activation, incomplete fetch, uninitialized saver, settlement gap, missing/naive/stale/future stamps and alternate inspection years on both sides of Dec 31 midnight ET. Removed the selected dated file only for the reference read. All 42 status fixtures preserved bytes, mtimes and file types.

| State path | FIFO with no writer | Directory |
| --- | --- | --- |
| Manual observation | Status and live block | Live raises `ContestStateError`. Status reports refusal, exit 0. |
| Automatic observation | Status and required automatic read block | Live raises `ContestStateError`. Status reports refusal, exit 0. |
| Season switch | Status and required automatic read block | Live raises `ContestStateError` for the corrupt switch. Status exits 1. |
| Saver flag | Status and required automatic read block | Both fall back to uninitialized. The automatic decision is accepted. |
| Model `streak.json` | Both block, even with the dated path selected | Model diagnostics report the read error. Contest selection is accepted. |
| Scanned pick JSON | Status and the automatic read block | Both skip the unreadable pick. The automatic decision is accepted. |
| Contest ledger | Status blocks in its ledger diagnostic. The live loader does not open the ledger. | Status reports an unreadable ledger. The live decision is accepted. |

- Reproduced round 1’s false acceptance at the saver, model and scanned-pick FIFOs. HEAD emits no automatic verdict before the bounded interrupt. With a valid dated observation, live selection bypasses the saver and pick scan. The status counterfactual still reaches those reads. Base status returned on the masked pick FIFO because it evaluated only the dated path. The ledger diagnostic retains base behavior.

## 3. Fetch cooldown stamps

Both `last_alert_at` and the actionable category map were independently seeded before each failure and success. `Now` below is the acknowledged failure’s UTC timestamp. Success sends no DM.

| Seed | Base failure → HEAD failure | Success stamp, base → HEAD |
| --- | --- | --- |
| Aware, inside six hours | Suppress and retain on both | Retain on both |
| Aware, at and outside six hours | Send and stamp now on both | Retain on both |
| `9999-12-31T23:59:59-01:00` | Suppress and retain on both | Retain on both |
| Trailing `Z` inside six hours | Suppress and retain on both | Retain on both |
| `+0000` inside six hours | Suppress and retain on both | Retain on both |
| Naive stamp or date only | Exit 1 with `TypeError` and seed intact → send, stamp now, exit 2 | Retain → discard |
| Garbage | Send and stamp now on both | Retain → discard |
| Number `42` | Exit 1 with `AttributeError` and seed intact → send, stamp now, exit 2 | Retain → discard |
| Aware `Z` date/time separator inside cooldown | Send and stamp now → suppress and retain | Retain on both |
| `Z` separator without offset | Send and stamp now on both | Retain → discard |

- The table shows the actionable stamp. The JSON logs also record the legacy summary and other categories. Success drops an invalid legacy field and keeps a category object after removing invalid entries. A legacy seed is expanded across categories on failure only when usable, so base can carry invalid leftovers where HEAD discards them.
- The two handovers explain the differences. Malformed and offset-naive values cannot establish cooldown. The native parser accepts uppercase `Z` as the date/time separator with a trailing `Z` or numeric offset. The base’s global substitution broke that separator. Using the same aware parser for validation and subtraction also preserves the year-edge stamp without overflowing on conversion to UTC.
- **Yes, the newly recognized separator form can suppress for longer than six hours.** At 2027-07-15 16:00 UTC, `2027-07-16Z16:00:00Z` suppresses at +7h on HEAD and sends at +30h. Base sends at both. The `Z`-separator form of the year-edge stamp also suppresses only on HEAD. This follows the existing six hours from the stored instant and the amendment’s requirement to honor any parseable aware stamp. Canonical future stamps already do this on base.

## 4. Round-1 controls

- **B1** The rendered repair command ran on three paths, including spaces and quotes. The required live decision then accepted dated streak 0. Saver bytes and mtime stayed fixed.
- **B2** Missing and mismatched MLB entry at initial, T-30 and T-15 used unconfirmed-delivery wording, attempted legs and the verification command. Delivered, private-locked and legacy DM bytes matched base. Receipt binding, escalation markers and the all-clear also matched.
- **B3** Single outcomes showed only the committed hitter. Doubles showed both executed hitters. Invalid leg summaries were omitted, and the bound positive receipt still led to exit 0 without a DM.
- **B5** A prior `[]` or `null` allowed switched success through observation, status, ledger and saver. Failure sent its DM, replaced status, exited 2 and left the pick tree intact.
- **B6** The B8a row and the three CLI cutovers require a successful fetch, independent acceptance and an active-year saver read. Failed fetch retained dated authority through retry. Settled completion initialized the flag and unsettled completion exercised conditional init and reread.

## 5. Test changes and mutations

- The seven nonblocking FIFO expectations were replaced as directed. The restart and ET-clock spies moved to `Path.read_text`. Both switch-open controls still require one read. The ordinary-file parity cases and cooldown preservation sequences add assertions. The saver spy has the coverage gap above.
- Isolated mutations in requested order produced **8, 0, 2, 3 failures**. Nonblocking decision reads, old timestamp validation and dropping the offset were detected. Skipping the saver in the status decision survived, with 99 passed. The saved real-file probe distinguishes that mutant.
- One focused pytest process ran both WR-B files, LightGBM optional, all web saver tests and the changed CS4b1 switch-snapshot test. **112 passed.**

## 6. Remaining diff

- Reviewed the full base-to-HEAD diff and both amendments. No additional production finding. `git diff --check 9e0ec28a HEAD` passed and the tracked worktree is clean.

## What I ran

- Archived the base and round-1 source and generated four independent scratch mutations. Retained the loader pickles, status and cooldown JSON, round-1 control replays, `saver-value-*.log` and `verified-comparisons.log` with the probes.
- Ran the focused and mutation jobs sequentially through `~/bts-runs/yield_run.sh`. See `focused.log`, `mutations.log`, `mutant-*.log` and the wrapper exit logs. Used pinned ET clocks, synthetic files and captured transports. The focused run permitted the local HTTP fixture on loopback. No external requests or full fast-suite run.
