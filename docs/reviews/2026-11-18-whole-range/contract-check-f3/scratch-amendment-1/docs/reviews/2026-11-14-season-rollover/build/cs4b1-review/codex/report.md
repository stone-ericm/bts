# Verdict

**SIGN** — reviewed `8421e3cd..28ae3aa9` against kickoff version 2. No blocking production discrepancy, unexplained nearby behavior change, or contract-row dispute found. This signs the slice under its stated limits; deployment remains deferred through CS6.

# Blocking findings

None demonstrated.

# Notes

- Test precision: `tests/test_season_contract_cs4b1.py:297` skips the status/ledger assertions for valid `predictions=[]`. Its successful dry-run case survives a mutant that persists status/ledger. Add explicit no-persistence assertions there if strengthening this individual test. Other switch-path tests and the independent matrix already cover that behavior; this is not a false production claim without coverage.
- Evidence tooling: substituting datetime/date classes changes the season-ledger recipe fingerprint's bound callable names. The broader frozen run had 1 failure/1,574 passes; the identical fingerprint test fails on archived base with that substitution. With normal datetime types the same focused inventory passes all 1,575 tests. The seven changed files pass in both requested years.
- Initial harness diagnostics are retained: `edges.log` includes five snapshot-helper failures and path-normalization differences, corrected in `command_edges_v2.py`/`review_support_v2.py`. The first matrix wrapper continued despite its nice error and overlapped its escalated retry; both initial datasets were discarded. `final-fetch_matrix.py.log` records the clean serial rerun that produced the final datasets. No pytest runs overlapped.

# The fetch

`fetch_matrix.py` archives base `src` without touching tracked files and runs **47,040 cases per version**: 40 switch fixtures (absent, eight individual failures, all 28 failure pairs, other-season, revoked, automatic) × 28 prior automatic-file fixtures × 21 responses × both dry settings. Check-1 pairs use a non-object containing the other malformed record, since an invalid top-level object cannot simultaneously satisfy the object checks.
Prior files include absent/v1/v2/mismatched identifier/mismatched season, missing and wrongly typed v2 fields, identifier length/case/character failures, legacy field refusals, malformed JSON, non-UTF8, non-object and another account. Responses include settled/unsettled/empty lists, all required prediction-shape failures, empty rounds, 503/429, null profile, both streak bounds/types, active above best, bad round conversion and paired shape/HTTP failures.
`matrix-{base,branch}.jsonl.gz` retain every command's exit/output/exception, every application file's before/after bytes, directories/modes, status record, captured DMs, helper calls and actual filesystem audit events. `matrix-comparison.json`, `matrix-branch-summary.json` and `file-order-audit.json` report no unexpected results.

- Automatic: observation replacement → success status replacement → ledger append → conditional real saver helper. The 100 successful non-dry automatic cells stamp v2/Y/identifier. Status retains its separate success clock and existing cooldown fields.
- Absent/other-season/revoked: all 300 successful non-dry cells replace status then append ledger, including cold directories. No automatic write/open-for-write/replacement, saver read/write/helper call or fetch DM occurs. Successful dry runs perform no persistence or saver work.
- Corrupt: valid readings fail with exit 2, failure status and captured throttled DM, without observation/ledger/saver work; dry-run does the same. Base ignores switch corruption and follows its old persistence path. Fresh case status files make the matrix's handled-failure DMs observable; existing cooldown tests also pass.
- Precedence: authentication and identity precede profile; a prior other-account file wins over every switch/response failure at both versions. Prior non-object/non-UTF8 files preserve exit 1 without status/DM. A null profile fails before rounds; a rounds 503/429 beats null predictions at both versions. After successful rounds, missing key → non-list → non-object row → empty mapping precede date conversion and active type/sign → best type/sign → best≥active → new best≤57. `near-{base,branch}.json` adds 32 paired response cases each.
- `command_edges_v2.py` fails observation, status, ledger, partial append, real saver lock, real saver state write and audit append across all five switch states. Completed prefixes stand; propagating persistence failures exit 1 without a new fetch alert; corrupt-switch alert handling can already have sent its DM before its status write fails. A partial ledger fragment survives; saver audit failure remains nonfatal. No rollback is introduced.
- The entry clock is at `src/bts/cli.py:2196`, before authentication. Fifty-six rollover probes advance during authentication, profile, rounds, observation/status writing, ledger opening and saver writing, using both source years. Observation season, both observation/ledger timestamps, switch evaluation and saver gate/argument stay at the captured 2027 instant; 2028 source dates do not invoke earning. Enable advances during its read/write yet writes captured Y/UTC; the next operation sees the new year.

Switch/enable coverage comprises 122 fixtures plus the optional-note supplement: missing keys, wrong JSON types, naive/offset/null timestamps, duplicate keys, NaN, huge integers, deep nesting, directories, dangling/looping symlinks, real unreadability and injected lstat failure. Validation precedes year comparison; lowest failed check wins. Changing enables install fresh UUID4 identifiers, carry revoked_at only from valid same-season snapshots, and read/write no other application file. No-ops preserve bytes/identifier/note. Wrong-year usage reads no application state. Five enable failure points preserve the old switch before replacement; interrupted output after replacement leaves an installed switch and a no-op retry.
Status additionally covers all 243 present/absent/broken combinations of its five files, each with actual-year and other-year inspection, 36 filesystem/decoder neighbors, cold picks paths and malformed ledger lines. Mutation guards and byte/tree comparisons show no writes or creations. Later sections survive failures, refused values are unavailable, and only corrupt switch exits 1. Equal identifiers beside revoked are explicitly labeled comparison-only, as required.

# Operator sequences

`operator-sequence-realistic.json` records each step's command, files, required LIVE decision and expected health result on one directory; resolved local picks appear on their actual days.

| Step | LIVE decision / health | Saver flag |
|---|---|---|
| Last-season v1 and used flag, July 15 | Existing automatic streak 12 still selected; prior-year best untrusted; saver WARN | 2026 used, read as uninitialized in 2027 |
| Dated observations July 15/16/17 and ledger-only fetches | Today's streak 3/best 8/saver used; fresh/trusted; no health alerts; automatic bytes frozen | Prior-year bytes unchanged |
| Status, enable, first automatic fetch | Dated observation still wins; health unchanged; first fetch stamps new activation | Fetch initializes 2027 not_earned |
| Fetch failing at observation, then at status | Both exit 1; first preserves prior observation, second installs streak 8/best 10 before failing; dated decision stays 3/used | not_earned unchanged |
| Hand-edited valid revoke and fetch | Ledger-only fetch preserves automatic bytes; dated decision/health unchanged | not_earned unchanged |
| Re-enable and status before another fetch | Fresh identifier; status reports mismatch; dated decision still wins | not_earned unchanged |
| July 18 before matching fetch | Dated observation expired; LIVE still uses mismatched automatic 8; coverage WARN | not_earned |
| Matching fetch July 18 | LIVE still 8; coverage WARN remains because profile coverage trails three local settled picks | Promoted to active |
| First January 1, 2028 fetch | Other-season switch makes fetch ledger-only; LIVE still uses frozen 8, best untrusted; coverage/saver WARNs | 2027 active bytes unchanged, read as uninitialized |

These automatic selections after revoke/mismatch/year rollover are the explicit CS4b-1 limit, not findings. Dated saver authority and the independent flag can differ; fetch earning does not change today's dated saver.

# Tests and false greens

Independent frozen harness: **551 passed in 2027; 551 passed in 2028**, across every added/touched file. Archived-base run of the two new files: **158 failed, 29 passed**. Availability failures for absent commands/module prove availability only; branch probes exercise their inner assertions.
The only base-green controls are named fully in `test-audit-summary-final.json`:

- `test_response_validation_and_alerts`: null/string/non-object-row rejection on real/dry paths (6), and valid empty-list dry-run (1).
- `test_shape_failure_precedence`: null profile and null predictions paired with failing rounds (2).
- `test_persistence_prefixes[observation-automatic]` (1).
- `test_prior_identity_guard_precedes_profile_and_switch` across five states/two dry settings (10).
- `test_decision_and_health_never_read_switch` across manual presence/switch corruption/both settings (8).
- `test_saver_audit_failure_nonfatal_and_helper_no_transition` (1).

Twelve scratch-source mutation families, together with the base failures, make **every one of the 187 new cases fail**. Controls detect altered failure exits/precedence, identity bypass, switch-reading consumers, swallowed observation-write errors, missing saver calls, fatal saver audit and empty-list rejection. Individual mutation logs and reached assertions are retained. The successful empty-list test's narrower dry-run assertion is noted above.
Spies target production call sites: local fetch imports resolve the patched auth/profile/saver functions; CLI rounds and JSON writers are the functions invoked; Path.open observes the real ledger append; builtin open observes saver lock/audit; enable imports the patched util writer at callback time. Real-helper mutations independently confirm migration coverage.

Five migrated files retain their original assertions except the intentional writer-bound/schema changes:

- `test_contest_fetch.py`: two snapshot tests gain fixed-year automatic fixtures. Both earning tests retain flag assertions and add real-helper call/season assertions; disabling earning fails both, including cold-at-10.
- `test_cli_integration.py`: three automatic fixtures retain active/source/lag/no-DM/atomic-temp assertions; the intended schema assertion changes v1→v2 and adds season/identifier checks.
- `test_saver_cs3.py`: fetch helper switches captured Y; null-source uses explicit empty predictions with nonempty rounds. Other-year tests assert no helper call; current-year tests still initialize/promote. Forced other-year earning fails all 12 fetch cases; omitted earning fails current-year fetch while manual controls pass.
- `test_contest_dated_cs4a_review.py`: switch fixture lets the real fetch promote the flag while retaining the used dated saver. Omitting earning fails this test.
- `test_contest_state_cs1_review.py`: validate/build now reject best 58 as required; a directly written invalid v1 retains reader refusal and manual recovery coverage. No unrelated assertion was removed or weakened.

# Rows 1 to 8

1. **Holds** — cold/status fixtures: base unknown-command exit 2; branch independent read-only sections/exit 0 or corrupt-switch 1. Probes: `near-*.json`, 243 status combinations and 36 broken-neighbor cases.
2. **Holds** — revoked/absent enable: base exit 2; branch one switch replacement, fresh ID, no-op retry and prescribed messages. Probes: 122 switch fixtures, five write/output failures, UUID4 and clock supplements.
3. **Holds** — missing predictions/empty rounds/best58 accepted at base now fail before persistence; valid empty list remains valid. Probes: fetch matrix, paired response comparisons and unchanged standalone-v1 builder controls.
4. **Holds** — unswitched base replaces v1 and may earn; branch freezes observation/saver and records status/ledger. Automatic branch stamps v2; corrupt switch rejects. Probes: full matrix and actual-open/rename order audit.
5. **Holds** — both versions preserve completed prefixes and legacy prior-file crashes; branch dry-run additionally evaluates switch, rejecting corrupt snapshots. Probes: 35 persistence combinations, partial append, audit failure, both dry settings.
6. **Holds** — base five-key ledger versus augmented branch line; nine mixed/duplicate prediction rows partition exactly. Duplicate round IDs mapped across years follow the actual lookup's last mapping. Probes: `ledger-counts.json` and all readers below.
7. **Holds** — 3,740 reader configurations per version × four APIs; 3,575 cells identical after path/default-field normalization, all 165 changed cells exact-v2 consequences. Invalid v2 season precedes ID after legacy checks; manual-v2 metadata stays None and receives specified trust. Probes: reader comparison and parser-order supplement.
8. **Holds** — decisions/health retain selection, fallback and switch-nonread behavior in both settings; only prescribed v2 trust/refusal differs. Probes: reader matrix, near-row controls, operator sequence and regressions. No uncovered nearby difference found for any row.

Ledger reader inventory, each actually fed old and augmented lines (`ledger-readers.json`): `bts.contest_ledger.parse_latest_ledger`/likely_save, `bts.web.saver_dashboard_context`, `bts.data.backup.verify_restored_ops_tree`, `boundary_shadow_census.load_ledger`, and the actual P05 main AST reader statements give identical projected results. `season_ledger.sources.contest_ledger.parse_contest_ledger` preserves new keys in raw line JSON; hashes/derived IDs consequently differ. Its reconciliation accounting/locators and contest helpers retain identical analytic values. Full `compile_bundle` comparison passes across 68 occurrence, 190 ledger, 10 slot and 192 reconciliation rows; raw evidence, hashes and ID references are the permitted differences.

# What 4b-2 needs from this

The shared ordered switch snapshot and parsed v2 metadata are sufficient; no impediment found. Enforce season AND activation equality after existing validation/selection precedence, while today's dated observation bypasses automatic refusal. Cover legacy/unknown schemas, re-enable before new fetch, failed fetch after replacement, and January rollover. Health must apply its expected-setting refusal without assuming old warnings enforce the switch. Implement revoke/quarantine and retain scheduler/fetch quiescence: no lock or cancellation exists. Preserve operator reconciliation of the separate saver flag. Enable remains an operator attestation; rejected readings have no ledger line; unswitched files freeze.

# What I ran

- All probes install DNS/connect-denying audit hooks; auth, profile, rounds and DM transports are captured. No network, box, account, checkout overlay, tracked edit or full fast-suite run.
- `final.log` / `final-fetch_matrix.py.log`: clean 94,080-command base/branch matrix; 47,040 branch file-order checks; zero violations. Base exit counts 0/2/1 = 16,000/27,680/3,360; branch = 800/42,880/3,360.
- `edges-v2.log`: 502 successful checks, zero failures; 7,480 reader cells/29,920 API calls; zero unexplained differences. Small direct supplements: response pairs/command comparisons, actual file-order analysis, optional note/UUID4/legacy-check precedence and realistic operator sequence.
- `tests.log` plus per-run JSON/logs: two frozen changed-file runs (551 each), base new tests (158 red/29 green), eleven mutation suites; `final.log` adds the twelfth mutation and migrated-helper mutation checks. Every new case can fail.
- `final-ledger_consumers.py.log`: mixed counts, every identified ledger reader, reconciliation and full compiler pass.
- `finish.log`: base frozen fingerprint artifact reproduced; normal focused suite **1,575 passed**. Required wrappers ran pytest families sequentially. `git diff --check` clean; tracked tree unchanged.
