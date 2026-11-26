# W20 · Change set C1 handover

Commit `d7d569a6` on `fix/wr-c1`. Base `3154f518`.

## Implementation

| Site | Behavior |
|---|---|
| `src/bts/health/runner.py:127` | After calibration, sample UTC once for explicit `today`. If it is earlier than the sampled ET date, pass the processed slate's last ET microsecond converted to UTC, plus `historical_day`. Default calls still use the check's own clock. |
| `src/bts/health/runner.py:140,305` | Render a historical slot crash with the fixed reason and original crash identity. After the existing WARN policy, move a selected historical WARN's attention label before the pointer in the DM. |
| `src/bts/health/contest_state.py:43` | Format each historical alert with the processed slate date, branch reason and exact pointer. Select the historical readers only in this mode. |
| `src/bts/contest_state.py:319,431,469` | Share parsing and selection with historical regular-file readers for manual, automatic and both coverage scans. Render validation reasons from fixed fields. |
| `src/bts/season_contract.py:74,123` | Read the switch safely in historical mode and render from the refusal branch before building live advice. |
| `src/bts/saver_state.py:52` | Read the historical flag with the regular-file guard and the same saver validator. |
| `CLAUDE.md:94` | W20 operator guidance, retained-evidence record and receipt limits. B8b(1) describes pre-C1 behavior. |

## Callers

- Production `scheduler.run_day` at `src/bts/scheduler.py:3544` passes the parsed processed ET date, `_contest_state_required(config)` and the existing health inputs. `bts schedule` chooses the explicit or launch ET day. The runner is the sole production caller of `health.contest_state.check`. Omitted, equal and future `today` pass just the picks directory and `expected`. Earlier explicit dates also pass the endpoint and `historical_day`.
- Morning check and live refusal use the decision loader. Status uses inspection and live decisions. The differential check exercised accepted and refused fixtures at 10:00. Their status output, decision or refusal, morning DM, same-day Alert values and DM body matched base after normalizing the temporary directory.
- `tests/health/test_runner.py` supplies explicit fixture dates and default or explicit `expected`. `tests/test_scheduler.py::test_eod_health_uses_run_date_and_pa_attribution` captures the unchanged scheduler kwargs. `tests/test_wr_c2_recovery.py::test_unknown_receipt_isolated_from_cutoff_and_eod` supplies an earlier processed day and patches the health slots to isolate receipt budgets. No existing test needed an L3 clock pin or assertion edit.
- Direct tests in `tests/health/test_contest_state.py`, `test_contest_dated_cs4a{,_review}.py`, `test_saver_cs3{,_review}.py`, `test_season_contract_cs4b1_edges.py`, `test_season_contract_cs4b2{,_review}.py` and `test_wr_b_{contract,edges}.py` use a pinned `now`, explicit or default `expected` and the default mode. Archived review drivers call the checks directly with a pinned clock or run the real runner.

## Contract coverage

| Row | Tests and result on the base |
|---|---|
| H1 | `test_mode_sample_and_last_et_microsecond` was RED for summer and winter UTC kwargs. `test_legacy_expiry_uses_last_et_microsecond` exercises expiry just before, at and just after the endpoint. `test_eod_uses_processed_observation_and_safe_recovery_copy` and the real Carbon scheduler adaptation `test_eod_scheduler_uses_processed_observation` reproduced the false page. Omitted, equal, future and sampled-same-day rollover matched the live path. |
| H2 | `test_historical_contest_uses_et_seasons_and_utc_grace` covers December 31 accepted and revoked switch, same and other flag seasons, and D+2/D+3 when wall time is several days later. `test_historical_state_paths_are_bounded` confirmed attempted reads on writerless FIFOs with a child deadline, plus a directory and the dated mask. Both coverage scans were observed. JSON and UTF-8 cases are in the exact-reason matrix. |
| H3 | `test_every_historical_contest_reason_is_exact_and_command_free` was RED and compares exact messages plus level, source, identity and ordering against live checks pinned to each processed endpoint. The matrix incorporates the driver cases, date validation, switch checks, bad UTF-8 and the overlong saver integer. `test_oom_path_pages_only_in_same_day_mode` was RED in the historical case and verifies exact base same-day copy and the derived OOM page per Amendment 1. |
| H4 | `test_historical_warn_pointer_survives_attention_decoration` was RED in both WARN-only and mixed DMs. It verifies the raw WARN, exact DM string and repeat suppression. `test_historical_failed_delivery_can_retry_then_deduplicates` covers failure, retry and repeated D. Existing C2 tests exercise both receipt orders and the EOD replay. |
| H5 | `test_historical_pointer_in_worst_operator_states` was RED for missing state at 00:15 and 02:00, dated saver, New Year, incompatible 01:10 replacement, 10:00 acceptance after an earlier refusal, no entry, unconfirmed entry, after-cutoff copy and coverage from later settled picks. The offline scheduler driver ran the old CLI follow-through and confirmed that a synthetic pre-settlement count of 12 was accepted at 00:30 and still selected at 10:00. |

## L1 sentence checks

| Sentence | Worst state and consequence of following it | Test |
|---|---|---|
| `Prior slate <D> ET.` | January 1 delivery names the December 31 slate. No reason authorizes action in the new season. | `test_historical_pointer_in_worst_operator_states[new_year]` |
| This check used files available at run time. | The 01:10 replacement can create or erase a finding. The sentence describes that evidence and authorizes no action. | `test_historical_pointer_in_worst_operator_states[replaced_auto]` |
| Do not enter picks, retry delivery, or change contest state from this alert. | D+1 can be missing usable state. Its morning check and live refusal supply current advice independently. | `test_historical_pointer_in_worst_operator_states[missing_at_0015]`, `[missing_at_0200]`, `[past_cutoff]` |
| Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. | Pending or uncertain entries require confirmation in MLB. A known no-entry slate has no unsettled result to wait for. No clock hour proves settlement. | `test_historical_pointer_in_worst_operator_states[dated_saver]`, `[no_entry]`, `[entry_uncertain]` |
| Use the new day's morning state check, when available, and live refusal for current diagnosis. | A skipped morning check can be silent. Silence verifies nothing. The acceptance fixture also demonstrates a silent morning after a real change of current state. | `test_historical_pointer_in_worst_operator_states[morning_silent]` |
| Season contract status shows current files. | Status can display live ways out before settlement. The preceding MLB verification prerequisite applies to any correction. | `test_historical_pointer_in_worst_operator_states[missing_at_0200]` and `.codex-review/wr-c1/compare_live.py` |

Each diagnostic sentence below is exercised by `test_every_historical_contest_reason_is_exact_and_command_free`. Its fixture supplies the diagnosed failure and asserts no action or values are embedded in the full message.

| Exact diagnostic sentence | Test |
|---|---|
| Contest streak observation is missing from account_state. | Exact reason matrix |
| The season switch is absent. | Exact reason matrix |
| There is no automatic observation for the switch activation. | Exact reason matrix |
| The saver state in the dated observation contradicts its streak. | Exact reason matrix |
| The saver counted as spent. | Exact reason matrix |
| A legacy or expired manual override file is present. | Exact reason matrix |
| The selected legacy manual override is ineligible for a live decision. | Exact reason matrix |
| The manual contest streak file is malformed or unreadable. | Exact reason matrix |
| The automatic contest streak file has an invalid streak. | Exact reason matrix |
| The season switch is corrupt because its season field failed validation. | Exact reason matrix |
| The season switch is revoked. | Exact reason matrix |
| The season switch is for another season. | Exact reason matrix |
| The automatic observation uses an unsupported schema. | Exact reason matrix |
| The automatic observation schema is missing, null or invalid. | Exact reason matrix |
| The automatic observation is for another season. | Exact reason matrix |
| The automatic contest streak file failed validation for season. | Exact reason matrix |
| The automatic observation and season switch have different activation IDs. | Exact reason matrix |
| The automatic contest observation is undated. | Exact reason matrix |
| The contest streak file has no coverage date, so coverage and freshness could not be verified. | Exact reason matrix |
| The contest streak file has a coverage date beyond the allowed future grace. | Exact reason matrix |
| Coverage trails by one settled pick, the expected ledger lag. | Exact reason matrix |
| Coverage trails by more than one settled pick. | Exact reason matrix |
| The predictions array can trail the live counter. | Exact reason matrix |
| The saver flag is uninitialized or unreadable. | Exact reason matrix |
| The saver flag contradicts the recorded streak. | Exact reason matrix |
| The saver flag is for another season. | Exact reason matrix |
| The automatic contest streak file is malformed or unreadable. | Exact reason matrix |
| The automatic contest streak file has an invalid saver value. | Exact reason matrix |
| The manual contest streak file has an invalid streak. | Exact reason matrix |
| The manual contest streak file has an invalid saver value. | Exact reason matrix |
| The dated manual contest streak file failed validation for effective_date. | Exact reason matrix |
| The dated manual contest streak file failed validation for season. | Exact reason matrix |
| The dated manual contest streak file failed validation for expires_at. | Exact reason matrix |
| The dated manual contest streak file failed validation for active_streak. | Exact reason matrix |
| The dated manual contest streak file failed validation for best_streak. | Exact reason matrix |
| The dated manual contest streak file failed validation for saver_state. | Exact reason matrix |
| The dated manual contest streak file failed validation for recorded_at. | Exact reason matrix |
| The dated manual contest streak file failed validation for source. | Exact reason matrix |
| The dated manual contest streak file failed validation for reason. | Exact reason matrix |
| The automatic contest streak file has an invalid streak or best streak. | Exact reason matrix |
| The automatic contest streak file failed validation for activation_id. | Exact reason matrix |
| The season switch file is malformed or unreadable. | Exact reason matrix |
| The season switch is corrupt because its schema_version field failed validation. | Exact reason matrix |
| The season switch is corrupt because its status field failed validation. | Exact reason matrix |
| The season switch is corrupt because its activation_id field failed validation. | Exact reason matrix |
| The season switch is corrupt because its switched_at field failed validation. | Exact reason matrix |
| The season switch is corrupt because its revoked_at field failed validation. | Exact reason matrix |
| The season switch is corrupt because its note field failed validation. | Exact reason matrix |
| The contest state health check failed before it could finish. | Exact reason matrix |

## Hostile inputs and retained evidence

- Interrupts propagate through the real runner. The parse interruption test checks that the observed contest file bytes survive. An ordinary exception retains the crash CRITICAL identity and logs the traceback, and its message uses the fixed historical reason.
- Writerless FIFOs are exercised in separate deadline-bounded children. The child isolates the contest slot because other EOD checks retain their readers. Existing WARN bookkeeping can still block on a FIFO. A guard is applied at each existing attempted historical contest read and follows regular symlink targets. No guard opens a masked file.
- The UTC endpoint fixes the historical ET year across reads. A sampled same-day run can cross midnight before the live check samples. An explicit D restart takes a fresh mode sample. An undated scheduler start after midnight selects the new day.
- After-D automatic fetch, switch or saver rewrites can hide a real alert or invent one. A fresh 01:10 observation can hide an actual missing-state CRITICAL or raise an activation-mismatch CRITICAL. Replacing a dated manual observation can hide a genuine dated saver WARN and add missing-state or switch CRITICALs. File timestamps and manual request logs cannot establish what was installed on D. The second manual read can see a different version from selection.
- Coverage and latest resolved still include retained picks after D. Two later settled picks can make a valid D observation appear behind or change INFO to WARN. A D result retained after midnight still counts.
- Historical silence verifies neither past nor live readiness. The same is true of morning-check silence.
- The generic dispatcher writes failed status for a raised send. A qualifying retry includes the established failure notice. A normal return without a usable message ID still records sent. An interrupt after remote acceptance can repeat the DM. A confirmed D suppresses repeated keys while the receipt represents D. C2's outcome receipt remains independent and retains its acknowledgement requirement.
- Under Amendment 1 a sanitized historical contest alert cannot create a false OOM page from discarded text. The same-day behavior, including a path named `oom`, still produces the existing page. The actual journal, restart and memory sources retain their handling.

## Verification

- Ran the base contract's five drivers in eight modes. The scheduler mode needed the Carbon directory on `PYTHONPATH` to compensate for the driver's fixture path. Its offline run reproduced the 00:15 false alerts and real CLI acceptance of the old copy.
- Escalated clock RED on the base produced 5 failures. Escalated base-source matrix produced 36 failures and 5 passes. The interrupt case in that run used a candidate-only reader symbol, so its failure does not demonstrate a regression. The final interrupt test injects at JSON parsing in either source tree. The masked FIFO on the base was opened after the retained observation expired on the wall clock.
- The first focused runs identified shared temporary receipt paths in my test helper and an incorrectly future-dated 10:00 morning fixture. Each fixture now has isolated bookkeeping and the morning acceptance fixture records its reading at 09:00.
- The final focused run passed 2,968 cases in 25.48 seconds. It included the C1 tests, existing runner and scheduler coverage, C2 receipt interleaving, B and CS3/4a/4b live checks, and the LightGBM optional-import and web saver tests in the same process.
- The one escalated fast-suite run on committed `d7d569a6` finished in 26:07 with 10,437 passed, 37 failed, 16 errors, 25 skipped, 6 deselected and 1 xfailed. The 53 FAILED and ERROR IDs exactly match `~/bts-runs/gate-failed.txt`; the full set had zero additions and zero omissions. The wrapper recorded exit 1 for those baseline failures. See `.codex-review/wr-c1/gate.log` and `.watch`.
- Four offline mutation checks rejected whole-second endpoint, missing UTC grace, command-bearing historical copy, and an attention label after the fixed pointer.
- Offline differential of current-mode output on the base and candidate matched after normalizing the synthetic path.

## Alert copy by branch

### absent

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest-account streak state expected but missing at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/absent/picks/account_state
````

Historical message.

````text
Prior slate 2027-03-24 ET. Contest streak observation is missing from account_state. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: season 2027 switch is absent.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/absent/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: stop the scheduler and wait for any fetch; run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/absent/picks` (while not switched, this records the reading in the ledger only); inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/absent/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/absent/picks`; then run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/absent/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is absent. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### approved_no_auto

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: there is no automatic observation for this activation.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/approved_no_auto/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/approved_no_auto/picks`; the season is already switched, so a switched fetch writes an observation for this season and current activation.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. There is no automatic observation for the switch activation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_saver

WARN · saver_state. Same-day sample at the slate endpoint.

````text
Streak Saver not_earned in dated observation /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/dated_saver/picks/account_state/contest_streak.manual.json, season 2027, contradicts streak 12; counting the saver as spent. Check MLB and run ONE correction to the observation. Earned and unused: `bts set-contest-streak --date 2027-03-24 --streak 12 --best-streak 12 --saver active --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/dated_saver/picks`. Already used: `bts set-contest-streak --date 2027-03-24 --streak 12 --best-streak 12 --saver used --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/dated_saver/picks`. Not earned: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver not_earned --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/dated_saver/picks`, replacing N and B with the active and best streaks MLB shows. Changing the saver flag does not correct this observation.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The saver state in the dated observation contradicts its streak. The saver counted as spent. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### legacy_expired

WARN · contest_state. Same-day sample at the slate endpoint.

````text
legacy/expired manual override present at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/legacy_expired/picks/account_state/contest_streak.manual.json; archive it (auto contest_streak.json is the live source)
````

Historical message.

````text
Prior slate 2027-03-24 ET. A legacy or expired manual override file is present. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### legacy_selected

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: the selected legacy manual override is no longer accepted for a live pick.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/legacy_selected/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
This dated observation REPLACES the legacy override; enable and the fetch cannot repair this. Automatic use must wait until the winning legacy override is removed or expires.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The selected legacy manual override is ineligible for a live decision. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### corrupt_manual

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest streak state malformed at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_manual/picks/account_state/contest_streak.manual.json: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
````

Historical message.

````text
Prior slate 2027-03-24 ET. The manual contest streak file is malformed or unreadable. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### corrupt_auto

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest streak invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_auto/picks/account_state/contest_streak.json: '`bts season-contract enable`\nnext line'
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest streak file has an invalid streak. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### corrupt_switch

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 3: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks/account_state/season_contract.json: season must be an integer.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/corrupt_switch/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its season field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### refused_revoked

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: season 2027 switch is revoked.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_revoked/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: stop the scheduler and wait for any fetch; run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_revoked/picks` (while not switched, this records the reading in the ledger only); inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_revoked/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_revoked/picks`; then run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_revoked/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is revoked. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### refused_other

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: season 2027 switch is other_season (file season 2028).
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_other/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: stop the scheduler and wait for any fetch; run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_other/picks` (while not switched, this records the reading in the ledger only); inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_other/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_other/picks`; then run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/refused_other/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is for another season. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### schema_text

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: schema_version is '`bts fetch-contest-streak`\nnext line', not 'bts_contest_streak_auto_v2'.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/schema_text/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/schema_text/picks`; the season is already switched, so a switched fetch writes an observation for this season and current activation.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic observation uses an unsupported schema. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### schema_none

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: schema_version is missing, null or not a string; look at status for the exact value.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/schema_none/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/schema_none/picks`; the season is already switched, so a switched fetch writes an observation for this season and current activation.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic observation schema is missing, null or invalid. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### auto_season

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: automatic observation season is 2028, not 2027.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/auto_season/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/auto_season/picks`; the season is already switched, so a switched fetch writes an observation for this season and current activation.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic observation is for another season. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
automatic contest season invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/auto_season/picks/account_state/contest_streak.json: must be an integer
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest streak file failed validation for season. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### auto_activation

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: automatic observation activation_id is 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', not the switch's 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/auto_activation/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/auto_activation/picks`; the season is already switched, so a switched fetch writes an observation for this season and current activation.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic observation and season switch have different activation IDs. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### undated

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
automatic contest observation is undated at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/undated/picks/account_state/contest_streak.json
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest observation is undated. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### missing_source

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest state has no source_date (/var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/missing_source/picks/account_state/contest_streak.json); no settled rounds in the profile, so coverage/freshness cannot be verified — investigate the fetch
````

Historical message.

````text
Prior slate 2027-03-24 ET. The contest streak file has no coverage date, so coverage and freshness could not be verified. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### future_source

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest state source_date=2027-03-27 is in the FUTURE (now 2027-03-25); file is corrupt/untrusted — investigate
````

Historical message.

````text
Prior slate 2027-03-24 ET. The contest streak file has a coverage date beyond the allowed future grace. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### coverage_one

INFO · contest_state. Same-day sample at the slate endpoint.

````text
contest state lags 1 settled pick (expected coverage lag — the predictions array trails the live activeStreak counter): source_date=2027-03-22, latest resolved 2027-03-24; refreshes on the next fetch
````

Historical message.

````text
Prior slate 2027-03-24 ET. Coverage trails by one settled pick, the expected ledger lag. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### coverage_two

WARN · contest_state. Same-day sample at the slate endpoint.

````text
contest coverage lag: 2 settled picks are newer than source_date=2027-03-22 (/var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/coverage_two/picks/account_state/contest_streak.json); latest resolved pick 2027-03-24. The predictions array trails the live activeStreak counter by >=2 -- usually a transient settlement lag (a broken fetch DMs separately)
````

Historical message.

````text
Prior slate 2027-03-24 ET. Coverage trails by more than one settled pick. The predictions array can trail the live counter. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### flag_missing

WARN · saver_state. Same-day sample at the slate endpoint.

````text
Streak Saver flag uninitialized for season 2027 (no saver_state.json) while streak is 12; counting the saver as spent. Check the saver in MLB and run ONE of these. Not earned yet this season: `bts saver-state --init not_earned --season 2027`. Earned and unused: `bts saver-state --init active --season 2027`. Already used: `bts saver-state --init used --season 2027`
````

Historical message.

````text
Prior slate 2027-03-24 ET. The saver flag is uninitialized or unreadable. The saver counted as spent. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### flag_contradiction

WARN · saver_state. Same-day sample at the slate endpoint.

````text
Streak Saver flag not_earned for season 2027 contradicts streak 12; counting the saver as spent. Check the saver in MLB and run ONE of these. Earned and unused: `bts saver-state --init active --force --season 2027`. Already used: `bts saver-state --use --season 2027`. If MLB shows the saver not earned yet, streak 12 is a stale contest reading (dated 2027-03-22): leave the flag as it is and correct the contest streak with `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/flag_contradiction/picks`. Replace N, B and S with MLB's active streak, best streak and saver state (one of not_earned, active or used).
````

Historical message.

````text
Prior slate 2027-03-24 ET. The saver flag contradicts the recorded streak. The saver counted as spent. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### new_year_refused

WARN · saver_state. Same-day sample at the slate endpoint.

````text
Streak Saver flag uninitialized for season 2027 (stale-season flag, season 2028) while streak is 12; counting the saver as spent. Check the saver in MLB and run ONE of these. Not earned yet this season: `bts saver-state --init not_earned --season 2027`. Earned and unused: `bts saver-state --init active --season 2027`. Already used: `bts saver-state --init used --season 2027`
````

Historical message.

````text
Prior slate 2027-12-31 ET. The saver flag is for another season. The saver counted as spent. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: season 2027 switch is revoked.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-12-31 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/new_year_refused/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: stop the scheduler and wait for any fetch; run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/new_year_refused/picks` (while not switched, this records the reading in the ledger only); inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/new_year_refused/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/new_year_refused/picks`; then run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/driver/new_year_refused/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-12-31 ET. The season switch is revoked. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### automatic_bad_utf8

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest streak state malformed at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/automatic_bad_utf8/picks/account_state/contest_streak.json: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest streak file is malformed or unreadable. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### automatic_bad_saver

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest saver_available invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/automatic_bad_saver/picks/account_state/contest_streak.json: []
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest streak file has an invalid saver value. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### manual_bad_streak

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest streak invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/manual_bad_streak/picks/account_state/contest_streak.manual.json: False
````

Historical message.

````text
Prior slate 2027-03-24 ET. The manual contest streak file has an invalid streak. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### manual_bad_saver

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
contest saver_available invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/manual_bad_saver/picks/account_state/contest_streak.manual.json: []
````

Historical message.

````text
Prior slate 2027-03-24 ET. The manual contest streak file has an invalid saver value. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_effective_date

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_effective_date/picks/account_state/contest_streak.manual.json: check 2, effective_date: must be a padded YYYY-MM-DD string
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for effective_date. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_season

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_season/picks/account_state/contest_streak.manual.json: check 3, season: must be an integer equal to effective_date's year
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for season. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_expires_at

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_expires_at/picks/account_state/contest_streak.manual.json: check 4, expires_at: must be a timezone-aware ISO timestamp
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for expires_at. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_active_streak

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_active_streak/picks/account_state/contest_streak.manual.json: check 5, active_streak: must be an integer from 0 to 57
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for active_streak. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_best_streak

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_best_streak/picks/account_state/contest_streak.manual.json: check 6, best_streak: must be an integer from active_streak to 57
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for best_streak. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_saver_state

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_saver_state/picks/account_state/contest_streak.manual.json: check 7, saver_state: must be not_earned, active or used
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for saver_state. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_recorded_at

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_recorded_at/picks/account_state/contest_streak.manual.json: check 8, recorded_at: must be a timezone-aware ISO timestamp
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for recorded_at. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_source

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_source/picks/account_state/contest_streak.manual.json: check 9, source: must be a string
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for source. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### dated_reason

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
dated contest observation invalid at /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/dated_reason/picks/account_state/contest_streak.manual.json: check 10, reason: must be a string when present
````

Historical message.

````text
Prior slate 2027-03-24 ET. The dated manual contest streak file failed validation for reason. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### auto_best_streak

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
automatic contest streak invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/auto_best_streak/picks/account_state/contest_streak.json: active_streak=12, best_streak=3
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest streak file has an invalid streak or best streak. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### auto_activation_id

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
automatic contest activation_id invalid in /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/auto_activation_id/picks/account_state/contest_streak.json: must be 32 lowercase hexadecimal characters
````

Historical message.

````text
Prior slate 2027-03-24 ET. The automatic contest streak file failed validation for activation_id. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_invalid_json

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 1: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks/account_state/season_contract.json: Expecting property name enclosed in double quotes: line 1 column 2 (char 1).
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_invalid_json/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch file is malformed or unreadable. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_schema_version

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 2: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks/account_state/season_contract.json: schema_version must be bts_season_contract_v1.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_schema_version/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its schema_version field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_status

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 4: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks/account_state/season_contract.json: status must be automatic or revoked.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_status/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its status field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_activation_id

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 5: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks/account_state/season_contract.json: activation_id must be 32 lowercase hexadecimal characters.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_activation_id/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its activation_id field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_switched_at

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 6: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks/account_state/season_contract.json: switched_at must be present, null or a timezone-aware ISO timestamp; non-null when status is automatic.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_switched_at/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its switched_at field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_revoked_at

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 7: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks/account_state/season_contract.json: revoked_at must be present, null or a timezone-aware ISO timestamp; non-null when status is revoked.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_revoked_at/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its revoked_at field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### switch_note

CRITICAL · contest_state. Same-day sample at the slate endpoint.

````text
Live decision refused: corrupt season switch, check 8: /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks/account_state/season_contract.json: note must be null or a string.
Alternative dated path — TEMPLATE: `bts set-contest-streak --date 2027-03-24 --streak N --best-streak B --saver S --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks`. Read N, B and S from MLB (active streak, best streak and saver state).
Alternative automatic path: inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks`; deliberately repair or remove the corrupt file, or replace it with enable. None of these verifies the endpoints or installs an observation, and doing so can reveal another blocker. Stop the scheduler and wait for any fetch before a changing enable. For current-season evidence, first repair or remove the corrupt switch to a nonautomatic state, run `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks` (ledger only while not switched), inspect `bts season-contract status --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks` and compare the readings with MLB; only then run `bts season-contract enable --season 2027 --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks`, then `bts fetch-contest-streak --picks-dir /var/folders/5j/s3fw5hps4rz2b0dc6r0l0jr00000gn/T/wr-c1-copy-z8qtsp87/extra/switch_note/picks` again to install an observation for this activation. If a legacy manual still wins, replace it with the dated template above.
One repair can expose another blocker. A non-zero fetch exit does not prove no observation was installed. Status output is not itself a verification of what MLB shows.
````

Historical message.

````text
Prior slate 2027-03-24 ET. The season switch is corrupt because its note field failed validation. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

### hostile_exception

CRITICAL · health_runner. Same-day sample at the slate endpoint.

````text
health check 'contest_state' crashed: ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has 5000 digits; use sys.set_int_max_str_digits() to increase the limit
````

Historical message.

````text
Prior slate 2027-03-24 ET. The contest state health check failed before it could finish. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````

## Full captured DM samples


````text
BTS health WARN attention:
- [saver_state] Prior slate 2027-03-24 ET. The saver flag contradicts the recorded streak. The saver counted as spent. (1st observed day) This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````


````text
BTS health CRITICAL alert(s):
- [contest_state] Prior slate 2027-03-24 ET. The season switch is revoked. This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.

WARN attention:
- [saver_state] Prior slate 2027-03-24 ET. The saver flag contradicts the recorded streak. The saver counted as spent. (1st observed day) This check used files available at run time. Do not enter picks, retry delivery, or change contest state from this alert. Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry. Use the new day's morning state check, when available, and live refusal for current diagnosis. Season contract status shows current files.
````
