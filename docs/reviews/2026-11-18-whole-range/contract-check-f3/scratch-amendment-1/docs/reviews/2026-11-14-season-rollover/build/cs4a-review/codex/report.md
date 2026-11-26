# Verdict

**SIGN** — reviewed `6fceaa3f` against `22e3ebf9`. No blocking production mismatch found. No tracked file changed.

# Blocking findings

None. I found neither a covered input with a wrong new outcome nor an unexplained difference near the twelve rows. This is an empirical review of the recorded inputs, not an exhaustive proof over arbitrary files or filesystem schedules.

# Notes

- **N1, existing code / follow-up:** without an observation in effect, a saver flag containing a 5,000-digit integer raises plain `ValueError` in both decision and health. `src/bts/saver_state.py:53` catches JSONDecodeError but not this decoder failure. Reproduced identically on base and branch: `matrix_{base,branch}.json`, key `absent|valid|huge|1|1`. Extend the defensive decoder exception handling in a follow-up. Today's dated observation isolates this flag successfully. This is the previously recorded CS3 limitation, not a CS4a regression.
- **N2, test coverage:** the checked-in 690 cases do not invoke the policy resolver or scheduler delivery and sample only some streak values. Independent policy/delivery probes below cover that gap; promote representative cases into durable regressions. No falsely GREEN new behavior was found.
- No objection to a contract-table row. The stated deployment, correction-after-selection, dashboard/flag, and request-log limits are respected and are not findings.

# What the pick does

`matrix_probe.py` extracted base `src` with `git archive`, ran separate Python workers and recorded 5,040 paired real-file combinations. Outcomes include reader, decision and health, both require/expected settings, six automatic-file kinds, and seven flag kinds.

| Matrix group | Paired cases | Different | Assessment |
|---|---:|---:|---|
| Absent / legacy / unreadable controls | 2,520 | 0 | Preserved |
| Valid dated observations, today/yesterday/tomorrow, with/without ignored expiry | 1,008 | 872 | Required by rows 8–11; every other-date outcome equals base with manual absent |
| Readable invalid v3, checks 2–10 | 1,512 | 1,512 | Required numbered rejection / CRITICAL |

`matrix_summary.json` lists every changed combination and changed component; full results are `matrix_base.json` and `matrix_branch.json`. No difference is left unclassified.

For streaks 0–57, every observation saver, seven flag states (including stale season, corruption and over-long integer), six automatic states, both require modes and two model-file states: **29,232 decision loads and 613,872 policy comparisons passed**. Probabilities are 0.70–0.90 inclusive in steps of 0.01, with a second candidate one point lower. The decision always uses the observation streak and saver; `active` is available, `used` is spent, and `not_earned` is available below 10 only. Status is fresh and best is trusted.

Another 14,616 comparisons cover opening and terminal-tail dates with best equal to current streak or 57. `model_isolation_probe.py` adds 2,088 loads / 43,848 comparisons with current-season conflicting, malformed, absent and stale-season model snapshots, while actual opens of the flag, automatic file and later settled local picks are forbidden. All passed. Total: **672,336 policy comparisons**, each resolving both actual and expected inputs. Checked-in base/tail hashes are recorded in `policy_summary.json`; neither artifact was mocked.

Without an observation in effect, the matrix matches base-with-manual-absent, including its existing errors. An invalid v3 raises rather than disappearing or taking the legacy path. Valid legacy schemas missing/null/unknown/v1/v2 preserve behavior. `parser_probe_final.py` adds **4,760 exact normalized base/branch parser comparisons**, including automatic files carrying v3 with invalid dated-only fields.

`delivery_probe.py` exercises real `run_cascade` → SSH-result decoding → `run_and_pick` → policy → scheduler delivery → pick file and `decision.json`. Only external prediction/schedule/DM transports are captured; an audit hook forbids DNS/connect. On July 15, observation 7/not_earned beats flag used and automatic 2: single, decision saver true. On July 16 without a new observation: automatic 2/flag used gives double, saver false. Pick notifications and scoreable decision records agree with selection; only the second day reads the flag. No delivery code consults legacy expiry or re-derives a saver that contradicts today's observation.

# Operator sequences

`operator_sequences.json` records one directory and real CLI/loader calls:

- Opening Night with last season's automatic 22, model 44 and used flag: observation 0/not_earned gives 0 with saver reserved; the old flag stays untouched.
- Next morning without a new observation: automatic 22 and current-season uninitialized saver, spent. This is the required intermediate behavior.
- Morning observation 1, then corrections 2 and 3: each new value becomes authoritative. Legacy-mode setter then replaces v3 with v2 streak 4.
- Streak 10/not_earned warns and counts saver spent; correction to active removes the warning and makes saver available.
- Next-day forgiven miss entered as 10/used keeps streak 10 and counts saver spent. A real fetch with captured auth/profile/round transports writes automatic 11 and auto-earns the flag; today's decision remains observation 10/used.

All three health correction commands were executed through the real CLI (the third after replacing the documented N/B placeholders with MLB fixture counts). Each sets the named saver and removes the warning. Quoted directories with spaces work.

Health sweep: **2,088** saver × streak × expected × automatic combinations. The sole alert for today's observation is WARN/source `saver_state` at not_earned, streak 10–15; every other combination is silent, with no flag read. Date/coverage/legacy alerts did not fire for these in-effect fixtures; out-of-effect fixtures follow base health. Real attention/dispatch/send_dm serialization produced the dated correction DM today and the ordinary flag-correction DM the next day; complete outgoing texts are in `delivery_summary.json`. No live service acceptance was tested.

# Tests and false greens

Own frozen-clock harness, with pandas preloaded, ran the exact checked-in file under 2027-07-15 and 2028-07-15. Each year: **branch 690 passed; base 680 failed, 10 passed**. Per-function audit of all 25 functions and exact node outcomes: `test_review.md`, `tests_*.json`.

The ten GREENs match the named controls: saver-without-date rejection (1), source/source-date rejection with existing state both ways (4), fixed legacy bytes/output (1), wrong-schema legacy parsing (3), automatic-v3 parsing (1). The first five are already rejected by base Click parsing; the last five are preservation assertions. All 22 functions containing new behavior fail on the base. Deliberate in-memory mutations of the real parser/writer make all five wholly-preservation nodes fail (`control-mutations.log`); no tracked code was mutated.

Spies wrap the production call sites: command's local import of `util.atomic_write_text`, Path/builtins reads, decision's `cs.load_saver_state`, health's local import from `saver_state`, and the shared predicate aliases. No required positive behavior is established only by mocking its implementation. Predicate delegation, no-op absence of writer calls, sentinels, fault/retry boundaries, and nested real invocations are genuinely exercised.

# Rows 1 to 12

1. **Holds:** 100 fixed-clock legacy option combinations have identical exit/output/file bytes on base and branch (`legacy_*.json`); original preservation test passes.
2. **Holds:** 50 rejection cases, no application opens/mkdir/replacement before rejection, explicit defaults and both boolean spellings; unpadded July 5 accepted and normalized (`command_summary.json`, `parser-final.log`). Base rejects the new options.
3. **Holds:** read → append → atomic replace traced; open/partial-write/flush/close faults prevent replacement; replacement/read-only-folder/directory faults preserve old authority; all retry successfully (`command_summary.json`, five checked-in fault tests).
4. **Holds:** no-op opens only for reading, no writer calls; absent/legacy/invalid/unreadable/other-date corrections covered; reason absent/empty/changed and next-day rejection exercised (`command_summary.json`, original CLI tests).
5. **Holds:** exact final line and own snapshot output; nested real writers at read/append/replace/output plus interruption/retry preserve the stated replacement ordering and success meaning (original interleaving tests; own os.replace races).
6. **Holds:** 2,183 validation cases include missing/type/range/timestamp failures and cross-property pairs, always the lower numbered failure; another 36 check-1/property pairs reject at file/decode handling before auto is read. Check 1 retains CS1 errors. Literal numeric/date/duplicate-key probes and 4,760 preservation comparisons pass.
7. **Holds:** all dated mappings and other-observation None defaults checked; source search finds one date predicate and one validator; all four operations invoke that predicate (original shared-predicate test, matrix/parser results).
8. **Holds:** today's valid v3 returns before corrupt/refused auto is opened; yesterday/tomorrow never become fallback, even with far-future ignored legacy expiry (`matrix_summary.json`, actual-open isolation probe).
9. **Holds:** full policy sweep and actual-open tripwires prove saver authority, fresh/trusted status and model isolation; one captured clock (`policy_summary.json`, `model_isolation_summary.json`, `command_summary.json`).
10. **Holds:** no legacy/expired warning for valid dated files on any date; invalid files produce CRITICAL; legacy outcomes unchanged (matrix health records).
11. **Holds:** full 0–57 health sweep, actual flag-read tripwire, correction commands and serialized attention DMs agree with the row (`policy_summary.json`, `delivery_summary.json`).
12. **Holds:** diff touches only CLI, contest reader, contest health and the new test file. Legacy/automatic controls and real fetch/delivery sequences show no unexplained output changes.

# What I ran

All test/family runs used `/Users/eric/bts-runs/yield_run.sh` with a log in this directory and `env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`. All pytest calls additionally disable cacheprovider. A small direct read-only base CLI control is saved in `base_new_mode.json`: exit 2 for unknown --date, no application directory created. No full fast suite, network, account access, tracked edits, commits, or pattern kills.

- `matrix_probe.py` → `matrix.log`: 5,040 paired combinations; 2,183 validation cases; 628,488 policy comparisons; 2,088 health cases; success.
- `command_probe_retry.py` → `command-retry.log`: 100 paired legacy options, 50 rejections, seven failure/retry classes, 29 existing-file kinds, replacement races, and clocks; success.
- Boundary clocks: start/end and adjacent microseconds, rollback, March 14 / November 7 DST, Dec 31 / Jan 1. Reader/decision/health/new command each read wall time once. `final_edges_probe.py` / `final-edges.log` additionally advances wall time during the real replacement: started Dec 31 23:59:59.9, both prints Jan 1 00:00:00.1; stored date/season/recorded time and in-effect output still use the captured start. Its 36 check-1/property pairs also pass.
- `delivery_probe.py` → `delivery.log`: operator sequence, real captured-transport selection/delivery, corrections and health DMs; success.
- `pytest_probe.py <src> <year> <label>` → `tests-2027.log`, `tests-2028.log`, `tests-base2027.log`, `tests-base2028.log`; outcomes above. Base exits 1 as expected.
- `parser_probe_final.py` → `parser-final.log`: 4,760 preservation comparisons plus literal CLI/file dates, -0/57/58/bool/float/NaN and duplicate keys; success.
- `control_mutations.py` → `control-mutations.log`: five expected test failures, driver success. `model_isolation_probe.py` → `model-isolation.log`: 2,088 loads / 43,848 policy comparisons; success.
- Initial command/parser attempts are preserved: one assertion counted its own verification reads, another misclassified an intentional exact-v3 difference, and one imported pandas under the substituted clock (exit 139). Corrected copies were created and run without editing submitted scripts. These are reviewer-harness errors, not production findings.
- Wrapper logs emit a sandbox setpriority warning, but jobs execute and final successful runs exit 0; no run was duplicated while its test process was active. `git diff --check 22e3ebf9 HEAD` passed and tracked-tree status is unchanged.
