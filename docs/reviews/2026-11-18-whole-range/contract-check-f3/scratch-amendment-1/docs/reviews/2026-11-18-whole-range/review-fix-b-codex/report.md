# Whole-range fix set B review

## Verdict

**BLOCK** · `2dce391c` reviewed against `9e0ec28a`. Two reproduced contract failures.

## Blocking findings

1. **B4 prints accepted when next-day live selection blocks on the saver FIFO.** At 12:00 ET on July 15, 2027, install a current switch, today's dated observation and a FIFO with no writer at `account_state/saver_state.json`. A switched fetch with active 7, best 9 and no settled predictions exits 0 and installs a matching automatic-v2 observation. Today's live selection succeeds on the dated observation. Status exits 0 and prints `automatic path alone: accepted (streak 7, recorded 2027-07-15T16:00:00Z)`. At the same time next day the ordinary required loader blocks at the saver read. `cutover-fifo.log` shows the full command sequence and bounded timeout. `status_probe.py` also reproduced misleading acceptance with a FIFO at `streak.json` or a scanned dated pick path.
   - At `src/bts/season_contract.py:255–276`, the status-only reader turns a FIFO into `OSError`. The saver reader at `src/bts/saver_state.py:52–56` converts that to an uninitialized flag and the decision succeeds. Preserve the regular-file reader and default live behavior. Have the status decision adapter surface a dedicated would-block error when the live decision would open a FIFO, and print it on the refused line. Keep the independent ledger diagnostic and the existing saver fallback for ordinary unreadable files. Assert the successful-fetch-plus-dated cutover and the following day's bounded live read.
2. **B5 discards a parseable aware cooldown stamp.** At 16:00 UTC on July 15, 2027, seed either `last_alert_at` or the actionable map stamp with `9999-12-31T23:59:59-01:00` and run a fetch with a recipient and a profile with object-valued predictions. Base exits 2 and suppresses the DM. HEAD exits 2, sends, and replaces the stamp. Both lanes are captured in `fetch_probe.py` and the base/HEAD JSON. `datetime.fromisoformat` accepts the stamp and aware subtraction works. The validation at `src/bts/cli.py:1730–1738` instead calls `_reading_time_utc`, which discards the stamp when conversion would cross year 10000. Validate with the same timestamp parser used for the cooldown comparison, require an offset, and avoid normalizing to a UTC calendar date. Cover both storage layouts and preservation on a successful fetch.

## Notes

- `load_decision_streak_state` forwards a custom reader to the saver on the automatic path. The model-only branch at `src/bts/contest_state.py:576` calls the default saver reader. The probe injected conflicting saver bytes and verified both behaviors. No production custom-reader caller takes the model-only branch.
- A same-date, scoreable unconfirmed decision still governs copy when the pick file has a delivery acknowledgment or different player metadata. That priority is explicit in B2. A record whose embedded date is stale can still satisfy the pre-existing scoreable gate and lead to the old warning. The changed branch requires today's embedded date.
- Missing or unknown string actions reach `no_decision`. An action stored as a JSON list still raises `TypeError` in the unchanged decision parser, before day-outcome rendering.
- The first scratch run omitted the tracked scheduler replay fixture. I staged it from the archives and reran the full base and mutation comparisons. Counts below use those reruns.

## 1. Default loader compatibility

- `loader_probe.py` ran archived base source and HEAD on the same directory at four ET instants around ordinary midnight and New Year. `cmp` matched the pickles for **2,112 states** after masking only the changed health advice. Compared returned dataclasses, exception type, message, args and explicit cause, plus alert level, source, incident key and text. Covered absent/model-only, v1/v2, yesterday/today/tomorrow v3, winning/expired legacy, switch and saver variants, model and resolved/unresolved pick inputs, both required modes, corrupt/empty/non-object/non-UTF-8 bodies, directories and injected `PermissionError`. Default reads blocked on FIFO in both revisions at all six exercised paths.
- Audited the scheduler, CLI live run/preview, orchestrator, dashboard, health and saver calls. Only the status inspection threads non-default options into the decision loader. Only its fallback passes `omit_selected_dated=True`. B5 uses the new reader independently for fetch status. The default reader identity check preserves the saver loader's existing default decoding and passed the injected-reader automatic control.

## 2. B4 status

- `status_probe.py` ran **33 cases** per revision. On HEAD every case emitted one independent verdict and preserved file types, bytes and mtimes. The four main combinations, expired dated, winning legacy, corrupt manual/auto/switch, wrong activation and missing auto produced the expected reasons and warning placement. Tested FIFO and directory at switch, auto, manual, saver, ledger, model and dated pick paths.
- With ordinary files the verdict matched the next day's loader in the same activation. A fetch that installed auto then exited 2, an uninitialized saver, a reading that aged past 14 hours and a two-pick settlement gap still yielded an accepted live decision. Old/future timestamps printed the stored value and missing/naive ones printed `unavailable`. Tested `--season` for another year on both sides of New Year. A current Dec 31 activation correctly refuses once the ET year changes. FIFO exceptions are in finding 1.

## 3. B1 saver warning

- `health_probe.py` extracted the warning's shell arguments and ran the real command with MLB values against absolute paths containing spaces and both quote styles, and a relative path with spaces. All three HEAD writes left today's required decision accepted on dated streak 0 and preserved the saver flag's bytes and mtime. The base command exited 2. Single and bundled DMs kept their incident keys and dedup day.
- The matrix matched every other alert identity and text. The template audit covered the three dated saver repairs, 6b-1 morning and 4c post-delivery warnings, the shared live refusal and revoke templates. Both exact-text suites passed.
- The real health runner, isolated to its contest check, processed a Dec 31 slate after midnight with Sep 26 coverage and emitted Jan 1 in the command. Dec 31's command then exited 2 and Jan 1's succeeded. This improves the command's applicability after midnight. The underlying EOD check still judges contest state at wall time while dispatch deduplicates by slate day.

## 4. B2 entry watchdog

- `watchdog_probe.py` exercised missing and wrong pending entries through the real verifier. Initial, T-30 and T-15 messages gave 45, 25 and 10 minutes to submit in the fixture. Each unconfirmed message said delivery was unconfirmed, named the checked legs as the attempted pick, distinguished missing from mismatch, and asked for DM/MLB verification and status for the same picks path. No qualifying same-date unconfirmed record received `Fix it now!`.
- Delivered, private-locked and legacy fallback console and DM bytes matched base. Across all 12 sequences, exit codes, normalized marker state and receipt count matched. The probe verified the receipt's date, matching decision hash and verifier reason at each marker. The all-clear was byte-identical and sent once. Existing controls covered cutoff, escalation retry, unverified entry and midnight binding. The scheduler crash/correction replay produced all three verification DMs for an unconfirmed double.

## 5. B3 day-outcome

- `outcome_probe.py` ran eleven shapes with and without a matching receipt. Valid singles printed A on HEAD versus A + B on base, in stdout and the problem DM. Valid doubles printed A + B on both. Null or whitespace primary names were omitted, and a null double leg left A. Decision bytes stayed fixed.
- Without a receipt, renderable commits printed `entry_unverifiable`, sent a DM and exited 1. A positive bound receipt printed `entered`, sent nothing and exited 0. A list-valued leg or numeric name crashed the base renderer, suppressing classification and the problem DM and returning 1 even with the positive receipt. HEAD restored the appropriate classification and exit policy. Missing/unknown string actions matched base at `no_decision`, exit 1, and list-valued action still errored on both.
- The dashboard uses the decision action and the pick file for executed legs. The watchdog, check-results and shadow evaluation use actual pick slots. Skip-policy shadow uses the decision's action/source for eligibility. Found no other reader that presents the decision record's runner-up as an executed leg on a single.

## 6. B5 fetch status

- `fetch_probe.py` ran both CLI lanes with `[]`, `null`, bad UTF-8, broken legacy and category stamps, FIFO, directory and an unwritable parent. In repairable HEAD cases success exited 0 and wrote observation → status → ledger → saver. Failure DMed then replaced status and exited 2 with observation, ledger and saver untouched. Directory/parent failures diagnosed the path and exited 2 after the successful lane installed the observation or the failed lane attempted its DM. FIFO was replaced atomically. Base blocked or raised before later steps for the corresponding hostile cases.
- Unswitched success wrote status and ledger only. Dry success wrote nothing and dry failure still DMed and wrote failure status. Failure sends that raised or returned no acknowledgment left cooldown unstamped. Existing interrupt and retry cases passed.
- Under pinned fetch times, canonical legacy and category records matched base byte-for-byte across failure, success, suppressed retry and retry past six hours. The automatic and ledger bytes matched too. The saver file stamps use the separately imported wall clock, so compared saver state and write order. An unusual `Z` separator accepted by Python's parser suppresses a DM on HEAD that the base's global `Z` substitution allowed. The valid aware year-edge regression goes in the other direction, as in finding 2.
- Amendment 1 changes only the stub acknowledgment and the after-observation status replacement exit expectation. Both passed with the before-observation exception control. The other edited tests adjust acknowledgment/exit expectations or spy on the one status switch open.

## 7. B6 checklist

- `checklist_probe.py` ran entry intent, revoke, status, dated write, pre-enable fetch/status, enable, post-enable fetch/status and saver read. The failed fetch retained dated acceptance and independent refusal until retry. Settled success auto-initialized the saver and the old unconditional init exited 1 on base. Unsettled success left it uninitialized and the conditional init/reread succeeded on HEAD.
- The row correctly requires successful fetch and independent acceptance plus the saver read, with dated scheduling pending cutover and a reread if initialization races. The synthetic cron control ran real show/install against a fake crontab. Scheduler stop/wait/restart were procedural boundaries. The box and live cron were untouched.

## 8. Tests and mutations

- Two WR-B files on base with the archived fixtures produced **58 failed, 18 passed**. The contract cases detected bare advice, old nag text, wrong names or renderer crashes, missing status verdict, fetch status crashes and the old checklist. Edge failures covered FIFO blocking, missing status reader instrumentation, bad stamps, restart verification wording and the masked cutover. Delivered/private/legacy, valid double and other surviving controls are listed in `base-tests-complete.log`.
- All six mutants were killed. Failing test counts in requested order were **9, 17, 3, 5, 1, 4**. The replay fixture was present on every final mutant run.
- The two new files plus LightGBM optional and web saver passed together, **89 passed**, with loopback-only socket access. The first sandboxed run had 88 passes and the expected socket bind denial. The new files passed at pinned native wall clocks in July 2027, the 2027 ET year boundary and July 2028, **76 each**. The targeted existing controls, including the three-day sequence and amendment assertions, passed **1,936**.

## 9. Remaining diff

- No additional tracked-source findings. Checked the touched file range, existing 4c/6b-1 exact-text tests and the checklist row. Fix set A was outside this checkout. `git diff --check 9e0ec28a HEAD` passed and the tracked worktree was clean.

## What I ran

- Base `git archive 9e0ec28a src` and a HEAD scratch archive. Retained the `*_probe.py` scripts, JSON/console evidence and loader pickles in this directory. `cutover-fifo.log` and `cooldown-format.log` are supplemental captured CLI checks.
- Ran pytest under `~/bts-runs/yield_run.sh` one invocation at a time. Evidence is in `head-loopback.log`, `clock-*.log`, `base-tests-complete.log`, `mutant-*.log` and `controls.log`. Used pinned ET fixtures, synthetic state and captured transports. No external requests or full fast-suite execution.
