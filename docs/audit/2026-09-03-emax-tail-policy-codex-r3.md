# Round 3 adversarial code review: E[max season best] tail policy

## Verdict

**BLOCK.** The direct 9/04 computation is sound: with the shipped artifacts,
`(streak=0, best=18 trusted, d=24, two executable games)` resolves to `double`,
`objective=emax_season_best`, `effective_best=18`, and tail SHA
`dc5d0c99...f949f4`. I also regenerated the policy from the artifact's rates and
compared all 195,112 action cells; they are identical.

The weakest claim is instead that one structured decision reaches every finalization
path. A fallback-refresh failure or restart recovery can deliver that exact tail
double and write a v3 `decision.json` with `objective`, best, and tail SHA all null.
Two other accepted safety contracts are also not implemented: best-streak trust is
inferred from the pathname rather than validated provenance, and a corrupt/missing
base silently disables the tail artifact's advertised hard binding.

Do not deploy until findings 1-3 are fixed. Findings 4-7 are integrity/analytics
defects; the current checked-in artifact itself is not exhibiting finding 4.

## Severity-ranked findings

### P1 — Cached fallback and restart recovery discard the decision that chose the pick (A, F, G, H)

The normal live route carries `SelectionResult.decision` through delivery, but the
safety-net route deliberately returns the cached `DailyPick` with `selection=None`
when refresh raises (`src/bts/scheduler.py:2018-2024`) or produces no selection
(`src/bts/scheduler.py:2026-2033`). Final fallback passes that null selection to the
delivery chokepoint (`src/bts/scheduler.py:3010-3016`). The commit writer derives
*all* state/objective fields from `selection`, not from the cached pick
(`src/bts/scheduler.py:921-936`). The analogous restart path short-circuits a
delivered/locked existing pick before `run_and_pick`, also returning
`selection=None` (`src/bts/scheduler.py:1537-1555`, `1573-1590`), and its recovery
writer explicitly commits only `source="unknown"` plus candidates
(`src/bts/scheduler.py:691-720`). `DailyPick.tail_policy_sha256` is present, but is
not read by either writer.

Concrete input -> wrong output, reproduced against the working tree:

```text
input:  2026-09-04 cached double, tail_policy_sha256="t"*64,
        fallback refresh selection=None, private delivery
output: schema=bts_daily_decision_v3, action=double, source=unknown,
        objective=None, best_streak=None, effective_best=None,
        tail_policy_sha256=None
then:   decision_objective(record) == "reach57"
```

The pick action remains `double`, but the authoritative record falsely identifies it
as a legacy reach-57 decision and cannot reproduce why it was played. The same output
occurs if the process saves/delivers the tail pick and dies before its decision write,
then classifies the existing pick after restart. `decision_objective` maps every null
objective to reach57 (`src/bts/daily_decision.py:38-48`), so the tail day is also
admitted to reach-57 alignment metrics (`src/bts/health/mdp_policy_alignment.py:90-95`,
`132-135`).

Persist the complete `PolicyDecision` with the preview/candidate (or in durable
scheduler state) before any delivery boundary. The cached fallback and classification
recovery must carry that exact saved structure into `_write_commit_decision`; they
must not re-read current contest state. Make v3 reject an absent/invalid objective so
this cannot silently become reach57.

The false green is
`tests/test_scheduler_tail_objective.py:94-106`: it calls `_write_commit_decision`
directly and hand-supplies every field, bypassing the paths that lose them.
`tests/test_scheduler_decision_record_integration.py:501-518` does exercise the real
failed-refresh safety net, but asserts only action/scoreability/delivery status and
therefore passes with every v3 objective field null. Add normal-delivery,
failed-refresh, delivered-existing restart, and `delivery_attempted` restart cases
for the 9/04 tail double, asserting the final record's objective, supplied/effective
best, and exact tail SHA.

### P1 — A filename and same-year date can authorize a false terminal stop (D, H)

The generic parser never validates `schema_version` or binds the declared `source`
to the file kind (`src/bts/contest_state.py:123-151`). The trust function then treats
anything not named exactly `contest_streak.manual.json` as automatic provenance and
checks only that `source_date.year == now_year`; it does not reject a future
observation (`src/bts/contest_state.py:73-96`). This makes both copied files and
symlinked paths trust-elevating boundaries.

Two concrete inputs both returned `(18, "trusted")` in a direct check:

```text
now=2026-09-19, path=contest_streak.json, source=mlb_bts_profile,
active=0, best=18, source_date=2026-12-31

now=2026-09-19, path=contest_streak.json, source=manual_cli,
schema/manual contents, active=0, best=18, source_date=2026-09-18,
override_expires_at=2026-09-01
```

At `s=0,d=9`, trusted `m=18` produces `skip` because the maximum reachable best is
18. Both observations should be untrusted, which degrades to `m=s=0` and continues
playing. A symlink named `contest_streak.json` to the expired manual file has the same
failure because `ContestStreakState.path` retains the symlink's auto filename.

Validate the schema and required source/provenance fields for each state-file kind;
do not infer manual-versus-auto solely from `path.name`. Reject a `source_date` after
the current ET date, and add the New-Year ET/UTC edge cases to prove the comparison
uses the same ET season/date basis as `now_year`. Continue treating an expired legacy
manual fallback as untrusted.

### P1 — Base failure turns off the promised base-SHA binding (C, E, H)

`_load_mdp` initializes `base_sha=None`; if base loading or the subsequent hash read
fails, it still calls `load_tail_policy(..., expected_base_sha=None)`
(`src/bts/strategy.py:58-70`). The loader checks the pair only when the expected SHA
is non-null (`src/bts/simulate/tail_policy.py:257-262`). The health check repeats the
same fail-open pairing logic (`src/bts/health/tail_policy.py:44-56`).

Concrete input -> wrong output, reproduced:

```text
base mdp_policy.npz: corrupt bytes
tail base_policy_sha256: "0"*64
state: 2026-09-04, s=0, trusted best=18, p=.72, executable DD
output: base_error set, tail loaded, action=double, degraded_reason=None
expected by the accepted failure contract: forced single with degraded_reason
```

This does not route into the 0.80 heuristic or the base zero region, but it still
fails the explicit rule that *base corrupt* is an artifact-failure shape and should
select the forced tail fallback. It also means “hard-bound to the base” is false when
the base is least trustworthy. `tests/test_tail_policy_strategy.py:244-248` codifies
the wrong behavior by requiring a corrupt base to keep an unverifiable tail.

Keep the loads error-isolated, but do not make an unverified tail actionable: retain
it for diagnostics if useful, mark it unusable for decisions, and force the tail
fallback whenever the current base SHA cannot be established. The health source must
also report the pair as unverified/invalid, not merely report the base error while
accepting the tail.

### P2 — The “strict” loader accepts a non-optimal policy and provenance-free manifest (E)

The exhaustive check validates only the stop/play partition. Any `single`/`double`
substitution in a beatable cell remains non-skip and therefore passes
(`src/bts/simulate/tail_policy.py:207-220`, `247-273`). `manifest` need only be a dict,
and `solver` need only be a nonempty string (`src/bts/simulate/tail_policy.py:263-267`).

I changed only the production-shaped live cell `[s=0,m=18,d=24,saver=0,q=0]`
from action code 2 (`double`) to 1 (`single`), set `manifest={}` and
`solver="wrong"`, wrote it with the test-only validation bypass, and loaded it through
the production loader. The loader accepted it and `lookup_tail_action(...)` returned
`single`; the exact solver returns `double`. Thus the loader proves only “stop iff
unbeatable,” not that the file is the declared exact E[max] policy or that its rates
came from the declared profile corpus.

Either deterministically re-solve from the embedded rates and compare the full table
on load/health, or pin and verify the complete release artifact SHA. Also validate
the required manifest keys/types and cross-check `n_bins`, counts, hits/both, seasons,
seed/parquet counts, and input hash. The health fixture currently blesses an empty
manifest (`tests/health/test_tail_policy_health.py:27-39`), so it cannot detect this
class of artifact.

### P2 — The policy SHA can describe different bytes than the loaded actions, and loading is unbounded (E)

`load_tail_policy` first reads the path to compute `raw`, closes it, and then reopens
the pathname through `np.load`; it stores the hash of the first read alongside arrays
from the second (`src/bts/simulate/tail_policy.py:337-372`). An atomic replacement or
symlink retarget between those opens yields `tail.sha256=sha(A)` but decisions from
artifact B. `_load_mdp` has the same two-open race between `load_policy` and
`sha256_file` (`src/bts/strategy.py:61-64`). That defeats the decision/pick provenance
precisely during an artifact deployment.

The loader also materializes the arrays before checking shape and accepts any
`max_days >= 28` (`src/bts/simulate/tail_policy.py:232-233`, `348-377`). A small
compressed NPZ that declares a huge table can consume unbounded memory before the
fallback receives a `TailPolicyError`; process OOM is not a forced-single fallback.

Open/read once and call `np.load(io.BytesIO(raw), allow_pickle=False)` so validation,
actions, and SHA share one byte string. Reject non-regular/surprising symlink inputs
or make the file-descriptor semantics explicit. Bound raw and ZIP-member sizes before
decompression, and require the v1 horizon exactly (`max_days == 28`) unless a later
schema explicitly expands it.

### P2 — v3 is nullable and tail records can be silently reclassified as legacy reach57 (F)

The v3 writer defaults every new objective field to null while still emitting the v3
schema (`src/bts/daily_decision.py:51-56`, `72-85`). The reader validates only action,
boolean scoreability, and presence of a date (`src/bts/daily_decision.py:96-112`). A
minimal v3 `{action:"skip", source:"mdp", scoreable:false, objective:null}` is
therefore accepted and `is_reach57_mdp_skip` returns true via the legacy default
(`src/bts/daily_decision.py:38-48`). This can admit a malformed tail skip to the
pre-registered shadow. The carried old 9/03 skip intentionally exercises the same
shape—new v3 with null objective—in `tests/test_scheduler_tail_objective.py:83-91`.

Default a missing objective to reach57 only for schema v1/v2. Require v3 objective to
be one of the two enums and enforce objective-consistent field shapes. When migrating
the old 9/03 `final_skip_candidate`, write explicit `objective="reach57"`; legacy
semantics do not require manufacturing an internally incomplete v3 record.

### P2 — Tail decisions are excluded after, not before, boundary construction (F)

The census accepts v3 records, then builds `primary_ps`, `primary_dates`, and the
same-date flat sample from *all* decisions (`scripts/audit/boundary_shadow_census.py:170-188`,
`239-246`). It excludes tail-objective rows only later in the per-row evaluation
(`scripts/audit/boundary_shadow_census.py:287-310`). Tail rows therefore do not appear
as evaluated rows, but their probabilities still change the candidate boundaries
used on reach57 rows.

Concrete sample -> wrong output:

```text
reach57 primary p: .70,.71,.72,.73,.74
correct primary quintiles: .708,.716,.724,.732
add five excluded tail records: .90,.91,.92,.93,.94
current primary quintiles: .718,.736,.904,.922
```

Filter to reach57 decisions before constructing every decision-date-derived boundary
sample, and add a mixed v2/v3 census test that asserts both row exclusion and sample
membership. Merely asserting that the tail row has `state_source=excluded_tail_objective`
would remain a false green.

## Attack-surface disposition

- **A:** The normal `run_single_check -> run_and_pick -> select_pick` route produces
  the correct 9/04 double and carries the complete decision. Cached fallback and
  recovery do not; finding 1 blocks.
- **B:** A genuine new single/double clears an old 9/03 skip in the normal cycle
  (`src/bts/scheduler.py:2677-2684`) and in both fallback sites; the EOD writer checks
  both `committed_pick_written` and the on-disk scoreable record before overwriting
  (`src/bts/scheduler.py:723-747`). I found no path in the reviewed change where a
  successfully written real commit is replaced by the old skip. If no new decision
  occurs, the old skip is semantically reach57, but the nullable-v3 encoding is
  finding 6.
- **C:** Regime selection precedes artifact lookup, and I found no tail-regime route
  into the 0.80 heuristic or base-table zero region for absent/partial injected dicts;
  `effective_pick_bar` is display-only and catches malformed inputs. Base failure
  nevertheless violates the promised forced-fallback behavior in finding 3.
- **D:** Finding 2 blocks. Aware expiry parsing and the default live ET-year
  conversion are otherwise consistent in the traced path.
- **E:** Solver recurrence, current-day `d` indexing, exact tie order, stop mask, and
  current artifact all checked out. Findings 3-5 cover remaining loader guarantees.
- **F:** Scheduler-state nested metadata and DailyPick production/shadow round-trips
  checked out. Findings 1, 6, and 7 cover the broken mixed-record consumers.
- **G:** The decisive missing test is the real 9/04 scheduler delivery/recovery matrix
  asserting final v3 metadata, not a helper call supplied with ideal metadata.
- **H:** Findings 1-3 should block tonight's deploy.

## Verification performed

- Repository base `3ec2fc5`; this review made no tracked-file edits (the supplied
  working tree remained dirty).
- Shipped hashes verified: base
  `66d154717ae51afb3343ee4bec8138c60bd1056e46a3de449043f4e9f76b93b4`, tail
  `dc5d0c9924431c11c43e720bf9e02903b709a157c0ebb699fdea43e7aaf949f4`.
- Full shipped table compared to a fresh solve from its embedded rates:
  `195112/195112` actions equal.
- `tests/simulate/test_tail_policy.py`: **38 passed**.
- Requested non-slow suite: **2100 passed, 4 failed, 2 deselected**. All four failures
  were sandbox socket-bind `PermissionError`s in three `test_sd_notify.py` tests and
  one `test_web_saver.py` test; no tail test failed. The claimed 2104 green cannot be
  fully reproduced under this filesystem/network sandbox.
- Rebuild dry-run reproduced 218,880 rows, 24 seeds, 3,600 late seed-days, 150
  distinct dates, `p_hit=2641/3600`, `p_both=1984/3600`, and the 9/04 double.
- Independent malformed-artifact, corrupt-base, best-trust, and cached-fallback
  probes described above.
- `git diff --check` clean.
