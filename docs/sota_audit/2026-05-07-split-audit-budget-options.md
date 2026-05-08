# Split-Audit Compute Budget Options

- **Generated**: 2026-05-07
- **Depends on**: `docs/sota_audit/2026-05-07-real-split-audit-plan.md`
- **Current plan verdict**: `phase_d_outer_eval_falsified`

## Verdict

Structured verdict: `phase_d_outer_eval_falsified`.

The candidate is now pooled-policy methodology on a determinism-certified raw profile surface, and Eric authorized up to `$1000` on 2026-05-07 for the audit/canary plan. Phase A shipped, the Phase B one-seed OCI profile canary passed, and the Phase B2 three-box OCI scaling canary passed across all three Ashburn availability domains. Vultr is currently blocked by an API IP allowlist error. Phase C completed as a 100-unique-seed Hetzner-plus-OCI audit using existing seed files only, with clean retrieval and teardown on both providers. Phase D then falsified the pooled-policy candidate on the 2025 outer-evaluation surface. If this pricing snapshot is more than 30 days old when activated for a new run, re-quote provider rates before provisioning.

## Recommendation

Use a **Hetzner-plus-OCI pooled-policy audit** as the Phase C launch structure. This is the best fit for the stated goal: SOTA methodology and winning BTS, under the `$1000` ceiling, because it uses existing disjoint seed files and keeps the load-bearing Hetzner-vs-OCI determinism question in scope.

Completed active allocation:

- 48 Hetzner seeds on up to 5 `CPX62` boxes
- 52 OCI seeds across the live-validated Ashburn AD pool
- no Vultr seeds in this launch

The original planning cost was about `$700`-`800`, but the passed OCI canaries showed raw profile generation is much cheaper than broad screening: one five-season OCI seed loop took about `40m39s`, the end-to-end one-seed canary took about `47m57s`, and the three-box multi-AD canary completed execution, retrieval, and teardown in under one hour. The Phase C planning estimate was about `$18.54` raw compute and `$40`-`45` with conservative overhead; the completed OCI r2 leg was about `4 * 11.95h * $0.3984/h = $19.04` raw compute before provider billing minimums or incidental storage/network charges.

The older **medium** tier remains the cheapest serious fallback. It gives a 48-seed audit comparable to the existing Hetzner 48-seed evidence, but it has weaker provider-diversity evidence and slower wall-clock under the current five-machine Hetzner cap.

The tier sections below remain reference baselines. The active decision rule is
the pre-registered 48 Hetzner / 52 OCI plan above. Use **budget** only as a
smoke audit or falsification pass; treat the old **luxury** tier as a larger
confirmation option, not the first launch under the `$1000` ceiling.

The active Phase C launch is complete, and Phase D has reported:

- the selected pooled-policy candidate is pre-registered at a commit SHA
- the Phase C provider/seed allocation is fixed with Claude
- seed file counts, hashes, and disjointness are recorded
- all `100` seed profile artifacts were retrieved
- all exact Phase C audit workers were deleted and the OCI workers were
  confirmed `TERMINATED` by exact-ID provider lookup
- `data/validation/phase_d_pooled_policy_outer_eval_2026-05-08.json`
  reports outer-evaluation mean P(57) gap `-0.062987` with provider-stratified
  seed-bootstrap CI `[-0.065250, -0.060757]`
- detailed result memo:
  `docs/sota_audit/2026-05-08-phase-d-pooled-policy-outer-eval.md`

## Capacity Snapshot

Eric provided Hetzner/Vultr capacity on 2026-05-07. Eric then authorized OCI verification with "lets verify" on 2026-05-07. OCI was checked read-only through the OCI CLI, then live one-seed and three-box scaling canaries were launched and torn down cleanly.

| Provider | Current limit | Increase path / lead time | Evidence source | Planning implication |
| --- | --- | --- | --- | --- |
| Hetzner | user-reported `5` machines; live Phase C obtained `4/5` because the fifth hit `server_limit` | Eric can ask for an increase in about `4` days | User-provided 2026-05-07 plus Phase C launch | Use the four obtained `CPX62` boxes for this launch. Revisit medium/luxury wall-clock estimates after the quota increase request is available. |
| Vultr | `30` machines / `$2500` ceiling | API IP allowlist must include current egress IP before launch | User-provided 2026-05-07 plus failed B2 Vultr canary | Capacity is likely enough, but current API access is blocked: `Unauthorized IP address: 2600:4041:5976:5800:e82e:1bd3:c1f2:1210`. Use Vultr only if fixed quickly; otherwise exclude from Phase C. |
| OCI | quota math allows more, but live Phase C 12-box launch degraded after `4/12` with `LimitExceeded` | No increase needed for the four-box Phase C relaunch; investigate practical instance/concurrent-launch limits before larger OCI runs | Read-only OCI CLI checks 2026-05-07 plus one-seed, three-box profile canaries, and Phase C launch attempt | Use `4` OCI E5 boxes for this launch. The one-seed canary passed in `US-ASHBURN-AD-1`; the three-box canary passed across `US-ASHBURN-AD-1`, `AD-2`, and `AD-3` with retrieval and teardown clean. |

OCI quota math: `83 OCPU/AD / 8 OCPU/box = 10` full boxes per AD, with OCPU as the binding constraint. Memory is not binding because `1250 GB/AD / 32 GB/box = 39` boxes per AD. Across three Ashburn ADs, the verified quota is `3 * 10 = 30` planned boxes.

OCI launch readiness is complete for raw profile generation at small multi-AD scale. `~/.oci/config` works, `VM.Standard.E5.Flex` is available in Ashburn, the project depends on `oci>=2.170.0`, and the public subnet `public subnet-bts-audit-vcn` allowed SSH and public IP assignment in the canaries. The subnet OCID was supplied through `OCI_SUBNET_OCID` for the runs because the macOS Keychain entry was still absent.

Using the full 30-box OCI quota still needs caution, not only read-only quota math. `OCIProvider.create()` now rotates successful launches across resolved ADs and treats exhausted `LimitExceeded`, `OutOfCapacity`, and transient 5xx launch failures as next-AD fallbacks. The three-box canary proved subnet compatibility, artifact retrieval, teardown, and provider provenance across ADs at small scale.

## Pricing Snapshot

Cloud compute prices were checked on 2026-05-07. Treat these as planning estimates, not invoices. They exclude storage, network transfer, public IPv4 extras, failed provisioning retries, and local development time.

| Provider | Planning shape | Unit price used | Source |
| --- | --- | ---: | --- |
| Hetzner | `CPX62`, 16 vCPU / 32 GB / 640 GB, Germany/Finland | `$0.0953/hour` | Hetzner 2026-04 cloud price adjustment |
| Vultr | Optimized Cloud Compute, 16 vCPU / 32 GB / 300 GB | `$0.476/hour` | Vultr pricing page |
| OCI | `VM.Standard.E5.Flex`, 8 OCPU / 32 GB | about `$0.3984/hour` | Oracle E5 Flex OCPU + memory pricing |

OCI is no longer one-seed or multi-AD canary/retrieval-gated in this repo for raw profile generation: the 2026-05-07 canaries retrieved artifacts cleanly and tore down all instances. Larger OCI launches should still preserve graceful-degradation and partial-retrieve safety.

## Runtime Calibration

These tiers use local historical run logs, not live cloud API checks:

| Surface | Boxes / seeds | Observed per-seed runtime | Notes |
| --- | ---: | ---: | --- |
| `data/hetzner_results/audit_full_48seed_v2` | 4 boxes / 48 seeds | mean `14.10h`, range `13.46h`-`14.80h` | Best cost anchor for fixed-n large audits. |
| `data/vultr_results/audit_ext_n100_v4` | 26 boxes / 52 seeds | mean `38.04h`, range `30.69h`-`44.15h` | Better burst capacity, much higher cost per seed-hour. |
| Hetzner pooled-bin raw profile logs | 8 boxes / 32 raw profile seeds | mean `60.9m`, median `61.3m`, range `51.2m`-`71.4m` | Closest existing non-OCI raw-profile workload, without `pa_predictions_*.parquet`. |
| OCI profile canary | 1 box / 1 raw profile seed | seed loop `40m39s`; end-to-end driver `47m57s` | `data/oci_results/pooled_profile_canary_2026-05-07`; retrieved 5 backtest parquets and 5 per-PA prediction parquets; teardown clean. |
| OCI scaling canary | 3 boxes / 3 raw profile seeds | about `41m` queue runtime per seed; end-to-end under `1h` | `data/oci_results/pooled_profile_scaling_canary_2026-05-07`; all three Ashburn ADs represented; retrieve and teardown clean. |

The historical broad Phase 1 screening runs are not the same workload as raw profile generation and are too conservative for the Phase C raw-profile budget. OCI now has retrieved one-box and three-box raw-profile canary surfaces; Phase C should estimate cost from raw-profile runtimes before provisioning.

Runtime may be higher under deterministic training than these historical logs. `BTS_LGBM_DETERMINISTIC=1` sets LightGBM deterministic mode plus `force_row_wise=True`; treat `10%`-`20%` extra runtime as a reasonable planning buffer until the split-aware launcher has a fresh dry run.

Cross-provider pooling has a determinism caveat. `CLAUDE.md` records that OCI E5.Flex drifted versus Hetzner on identical seed `42` without the determinism flag. Luxury/provider-diverse runs must either set `BTS_LGBM_DETERMINISTIC=1` across all providers after a deliberate re-baseline, or keep provider tags explicit and avoid interpreting pooled cross-provider seeds as pure seed variation.

## OCI Acceleration Sensitivity

This is not a launch budget. It records the old pre-canary sensitivity table for comparison. The one-seed OCI canary has now proved that OCI E5 runtime and retrieval are healthy for raw profile generation at single-box scale.

If OCI runtime matches Hetzner's `14.10h/seed` anchor, plausible acceleration looks like:

| Variant | Shape | Raw compute sensitivity | Wall-clock sensitivity | Extra blockers |
| --- | --- | ---: | ---: | --- |
| Medium acceleration | 48 seeds on 5 Hetzner + 4 OCI | about `$156` | about `3.1` days before deterministic-runtime buffer | One-seed OCI retrieval is now proven; old table still overstates runtime for raw-profile work. |
| Luxury balanced | 48 seeds on 5 Hetzner + 4 OCI, plus 52 Vultr seeds on 26 Vultr boxes | about `$1,098` | about `3.2` days before deterministic-runtime buffer | Keeps Vultr provider-diversity leg; old table still overstates raw-profile runtime. |
| Luxury OCI-heavy | 100 seeds on 5 Hetzner + 20 OCI | about `$476` | about `2.4` days before deterministic-runtime buffer | Requires a multi-box OCI canary proving the driver's AD-rotation/fallback path in live launch and retrieval conditions. |

Do not use this old table as the Phase C budget commitment. The canary supplied measured runtime, retrieve status, determinism metadata, and provider provenance; a refreshed Phase C budget should use those values plus a multi-AD scaling caveat.

## Budget Tier

**Shape**: 4-5 Hetzner `CPX62` boxes, 16 seeds.

**Expected cost**: about `$22` raw compute; plan for `$25`-`40` with overhead.

Cost math: `16 seeds * 14.10h/seed * $0.0953/h = $21.50`, plus provisioning, idle, retrieve, retry, and deterministic-runtime overhead.

**Expected wall clock**: about `2.5`-`3` days with four boxes; about `2` days with the current five-machine Hetzner cap, before deterministic-runtime buffer.

Activation blockers: none for the pre-registered Hetzner-plus-OCI Phase C launch.

Capacity feasibility: fits inside the current Hetzner `5`-machine cap.

**Use when**:

- candidate is speculative
- goal is to falsify obvious failures cheaply
- Eric wants a quick signal before spending on a full 48/100-seed audit

**Evidence posture**:

- Good for sign/direction and catastrophic-failure detection.
- Not enough for a final deployment argument.
- `survives_outer_eval` at 16 seeds is smoke-only and does not clear separate production gates.
- If inconclusive, graduate to medium rather than interpreting noise.

## Medium Tier

**Shape**: 5 Hetzner `CPX62` boxes, 48 seeds.

**Expected cost**: about `$65` raw compute; plan for `$70`-`100` with overhead.

Cost math: `48 seeds * 14.10h/seed * $0.0953/h = $64.51`, plus provisioning, idle, retrieve, retry, and deterministic-runtime overhead.

**Expected wall clock**: about `5.5`-`6` days with the current five-machine Hetzner cap; plan for `6`-`7` days with deterministic-runtime buffer. With the previous four-box surface, this was about `7` days.

Activation blockers: none for the pre-registered Hetzner-plus-OCI Phase C launch.

Capacity feasibility: the Hetzner-only default fits inside the current Hetzner `5`-machine cap, but uses all five boxes. A single provisioning failure degrades throughput unless the run waits for the quota increase or reallocates to another provider. Any faster Hetzner-only medium variant needs the quota increase Eric can request in about `4` days. If speed matters enough to spend more, revisit the OCI acceleration sensitivity after the one-seed canary measures runtime and retrieval.

**Use when**:

- candidate is pre-registered at a commit SHA
- Eric wants the cheapest serious split audit fallback
- low cost is more important than fast turnaround

**Evidence posture**:

- Cheapest serious fallback, no longer the recommended default under the `$1000` ceiling.
- Comparable scale to the existing Hetzner 48-seed surface.
- Strong enough to decide whether a candidate deserves luxury/provider-diverse confirmation or should be stopped.

## Luxury Tier

**Shape**: 100-seed provider-diverse audit.

Default allocation:

- 48 Hetzner seeds on 5 `CPX62` boxes
- 52 Vultr seeds on 26 optimized 16 vCPU / 32 GB boxes
- optional OCI-heavy acceleration after retrieval/provenance checks pass

**Expected cost**: about `$1,000` raw compute for the 48 Hetzner + 52 Vultr split; plan for `$1,200`-`1,500` with overhead. If the goal is faster wall-clock by bursting closer to 100 seeds on Vultr, plan closer to `$1,800`-`2,200`.

Cost math for default provider-diverse run:

- Hetzner leg: `48 seeds * 14.10h/seed * $0.0953/h = $64.51`
- Vultr leg: `52 seeds * 38.04h/seed * $0.476/h = $941.78`
- Combined raw compute: about `$1,006`, before overhead.

Cost math for all-Vultr burst upper bound: `100 seeds * 38.04h/seed * $0.476/h = $1,810`, before overhead.

**Expected wall clock**:

- about `5.5`-`7` days under the current five-Hetzner / 30-Vultr caps, depending on deterministic-runtime overhead and whether the Hetzner 48-seed leg remains the critical path
- about `6`-`8` days for a 100-seed all-Vultr burst under the current 30-machine cap, unless the actual candidate workload is materially faster than the historical broad Phase 1 surface
- faster luxury runs require either a Hetzner quota increase, a shorter candidate workload, or a proven OCI provider pool; see the OCI acceleration sensitivity section for non-commitment upside estimates

Activation blockers: none for the pre-registered Hetzner-plus-OCI Phase C launch; Vultr remains deferred.

Vultr cap check: the all-Vultr raw compute estimate is about `$1,810`, but a `10%`-`20%` deterministic-runtime buffer plus provisioning/retry overhead can push the launch close to the user-reported `$2500` ceiling. Re-quote before provisioning.

Capacity feasibility: the default luxury allocation fits inside the current Hetzner `5`-machine cap and Vultr `30`-machine cap, using 5 Hetzner boxes and 26 Vultr boxes, but Vultr API access must be fixed first. A 100-seed all-Vultr burst also fits the 30-machine cap, but must be re-costed against the `$2500` ceiling immediately before launch. Verified OCI E5 quota supports up to 30 planned OCI boxes across Ashburn, and the driver now has AD-rotation/fallback support plus live three-AD retrieval/teardown evidence.

OCI inclusion rule:

- live `VM.Standard.E5.Flex` capacity: verified read-only on 2026-05-07, re-check if stale at launch time
- credentials/launch readiness: one-seed launch passed with `~/.oci/config` and `OCI_SUBNET_OCID`; the macOS Keychain `oci-subnet-ocid` entry is still absent
- one-seed retrieval canary: passed on 2026-05-07
- AD spreading and multi-AD retrieval/teardown canary: passed on 2026-05-07 with one box in each Ashburn AD
- determinism/provider tags: driver metadata support exists for remote profile seeds (`provider`, `box_name`, `box_region`, `run_kind`, `queue_mode`, `determinism_intent`, `launch_command_env`, `profile_seasons`); keep OCI out of the pooled result unless retrieved artifacts are checked and any final analysis preserves provider tags

The OCI one-seed retrieval canary was a small spend: about `47m57s * $0.3984/h = $0.32` raw compute before provider billing minimums or incidental storage/network charges.
The three-box OCI scaling canary was also small: about `3 * 0.76h * $0.3984/h = $0.91` raw compute before provider billing minimums or incidental storage/network charges.

**Use when**:

- medium tier is positive or strategically inconclusive
- Eric wants provider-diversity evidence
- wall-clock time matters more than cost
- candidate is close enough to production that richer FDR/OPE/conformal gates may be exercised

**Evidence posture**:

- Best chance of detecting seed/provider instability.
- Best support for richer audit-verdict FDR because it can produce more independent season/seed evidence.
- Still not a deploy by itself; separate production gates remain required.

## Decision Rule

1. Launch the pre-registered Hetzner-plus-OCI Phase C audit using `--seeds-file`.
2. Keep Vultr deferred unless the Phase D result needs a separate provider-diversity addendum.
3. If Vultr is later fixed, treat any Vultr work as a separate pre-registered addendum, not a silent extension of this audit.
4. Use the old budget tier only for a smoke/falsification pass, not for the main deployment-grade argument.

If the candidate changes, stop and re-pre-register before spending.

## Source Links

- Hetzner: https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/
- Vultr: https://www.vultr.com/pricing/
- Oracle: https://www.oracle.com/cloud/iaas-paas/

## Verification

Docs plan plus OCI canary result. Verified locally:

```bash
git diff --check
oci limits service list --compartment-id <tenancy> ...
oci limits value list --compartment-id <tenancy> --service-name compute ...
oci limits resource-availability get --compartment-id <tenancy> --service-name compute --limit-name standard-e5-core-count ...
oci limits resource-availability get --compartment-id <tenancy> --service-name compute --limit-name standard-e5-memory-count ...
oci compute shape list --compartment-id <tenancy> --shape VM.Standard.E5.Flex ...
oci network subnet list --compartment-id <tenancy> ...
OCI_SUBNET_OCID=<public-subnet-ocid> UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py --run-kind profiles --provider oci --boxes 1 --seeds 1 ...
find data/oci_results/pooled_profile_canary_2026-05-07 -maxdepth 3 -type f | sort
jq . data/oci_results/pooled_profile_canary_2026-05-07/audit_validation_split.json
OCI_SUBNET_OCID=<public-subnet-ocid> UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit_driver.py --run-kind profiles --provider oci --boxes 3 --seeds 3 ...
find data/oci_results/pooled_profile_scaling_canary_2026-05-07 -maxdepth 4 -type f | sort
jq . data/oci_results/pooled_profile_scaling_canary_2026-05-07/audit_validation_split.json
```
