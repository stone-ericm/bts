# Split-Audit Compute Budget Options

- **Generated**: 2026-05-07
- **Depends on**: `docs/sota_audit/2026-05-07-real-split-audit-plan.md`
- **Current plan verdict**: `plan_blocked_no_viable_candidate`

## Verdict

Structured verdict: `budget_options_documented_blocked_on_activation`.

The tiers are ready for planning, but none should run until candidate intake, live resource inventory, and split-aware cloud orchestration are complete. Eric's 2026-05-07 capacity numbers are recorded below as partial inventory, not a substitute for live quota, credential, cost, and artifact-provenance checks. If this pricing snapshot is more than 30 days old when activated, re-quote provider rates before provisioning.

## Recommendation

Use the **medium** tier as the default once Eric names a candidate stack and compute budget. It gives a 48-seed audit comparable to the existing Hetzner 48-seed evidence, keeps spend controlled, and avoids the large Vultr premium unless wall-clock time or provider-diversity evidence is worth paying for.

Use **budget** only as a smoke audit or falsification pass. Use **luxury** only for a candidate that has already survived the medium tier or for a candidate Eric is seriously considering taking through production gates.

All tiers remain blocked until:

- candidate intake is complete
- live Hetzner/Vultr/OCI resource inventory is authorized and recorded; Eric's capacity notes below are partial inventory only
- `scripts/audit_driver.py` gets split-flag pass-through or a dedicated split-audit launcher exists

## User-Provided Capacity Snapshot

Eric provided this live capacity snapshot on 2026-05-07:

| Provider | Current limit | Increase path / lead time | Evidence source | Planning implication |
| --- | --- | --- | --- | --- |
| Hetzner | `5` machines | Eric can ask for an increase in about `4` days | User-provided 2026-05-07 | Use up to five `CPX62` boxes now. Revisit medium/luxury wall-clock estimates after the quota increase request is available. |
| Vultr | `30` machines / `$2500` ceiling | Not specified | User-provided 2026-05-07 | Enough for the default luxury Vultr leg and likely enough for all-Vultr raw compute, but overhead and deterministic-runtime buffer must be checked against the `$2500` ceiling before launch. |
| OCI | unknown | Must be checked live | User-provided 2026-05-07 | Keep OCI to the one-seed canary path until live service limits are verified. |

## Pricing Snapshot

Cloud compute prices were checked on 2026-05-07. Treat these as planning estimates, not invoices. They exclude storage, network transfer, public IPv4 extras, failed provisioning retries, and local development time.

| Provider | Planning shape | Unit price used | Source |
| --- | --- | ---: | --- |
| Hetzner | `CPX62`, 16 vCPU / 32 GB / 640 GB, Germany/Finland | `$0.0953/hour` | Hetzner 2026-04 cloud price adjustment |
| Vultr | Optimized Cloud Compute, 16 vCPU / 32 GB / 300 GB | `$0.476/hour` | Vultr pricing page |
| OCI | `VM.Standard.E5.Flex`, 8 OCPU / 32 GB | about `$0.398/hour` | Oracle E5 Flex OCPU + memory pricing |

OCI remains quota/retrieval-gated in this repo. Use it as a capacity canary until a live inventory proves service limits and result retrieval are healthy.

## Runtime Calibration

These tiers use local historical run logs, not live cloud API checks:

| Surface | Boxes / seeds | Observed per-seed runtime | Notes |
| --- | ---: | ---: | --- |
| `data/hetzner_results/audit_full_48seed_v2` | 4 boxes / 48 seeds | mean `14.10h`, range `13.46h`-`14.80h` | Best cost anchor for fixed-n large audits. |
| `data/vultr_results/audit_ext_n100_v4` | 26 boxes / 52 seeds | mean `38.04h`, range `30.69h`-`44.15h` | Better burst capacity, much higher cost per seed-hour. |

These are conservative for a single exact candidate because the historical surfaces were broad Phase 1 screening runs. The first split-aware launcher should render the exact command and re-estimate cost from the actual candidate workload before provisioning.

Runtime may be higher under deterministic training than these historical logs. `BTS_LGBM_DETERMINISTIC=1` sets LightGBM deterministic mode plus `force_row_wise=True`; treat `10%`-`20%` extra runtime as a reasonable planning buffer until the split-aware launcher has a fresh dry run.

Cross-provider pooling has a determinism caveat. `CLAUDE.md` records that OCI E5.Flex drifted versus Hetzner on identical seed `42` without the determinism flag. Luxury/provider-diverse runs must either set `BTS_LGBM_DETERMINISTIC=1` across all providers after a deliberate re-baseline, or keep provider tags explicit and avoid interpreting pooled cross-provider seeds as pure seed variation.

## Budget Tier

**Shape**: 4-5 Hetzner `CPX62` boxes, 16 seeds.

**Expected cost**: about `$22` raw compute; plan for `$25`-`40` with overhead.

Cost math: `16 seeds * 14.10h/seed * $0.0953/h = $21.50`, plus provisioning, idle, retrieve, retry, and deterministic-runtime overhead.

**Expected wall clock**: about `2.5`-`3` days with four boxes; about `2` days with the current five-machine Hetzner cap, before deterministic-runtime buffer.

Activation blockers: candidate intake, authorized live resource inventory, and split-aware cloud launcher.

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

Activation blockers: candidate intake, authorized live resource inventory, and split-aware cloud launcher.

Capacity feasibility: fits inside the current Hetzner `5`-machine cap, but uses all five boxes. A single provisioning failure degrades throughput unless the run waits for the quota increase or reallocates to another provider. Any faster Hetzner-only medium variant needs the quota increase Eric can request in about `4` days.

**Use when**:

- candidate intake is complete
- Eric wants the default serious split audit
- low cost is more important than fast turnaround

**Evidence posture**:

- Recommended default.
- Comparable scale to the existing Hetzner 48-seed surface.
- Strong enough to decide whether a candidate deserves luxury/provider-diverse confirmation or should be stopped.

## Luxury Tier

**Shape**: 100-seed provider-diverse audit.

Default allocation:

- 48 Hetzner seeds on 5 `CPX62` boxes
- 52 Vultr seeds on 26 optimized 16 vCPU / 32 GB boxes
- optional OCI E5 Flex canary only after live quota and retrieval checks pass

**Expected cost**: about `$1,000` raw compute for the 48 Hetzner + 52 Vultr split; plan for `$1,200`-`1,500` with overhead. If the goal is faster wall-clock by bursting closer to 100 seeds on Vultr, plan closer to `$1,800`-`2,200`.

Cost math for default provider-diverse run:

- Hetzner leg: `48 seeds * 14.10h/seed * $0.0953/h = $64.51`
- Vultr leg: `52 seeds * 38.04h/seed * $0.476/h = $941.78`
- Combined raw compute: about `$1,006`, before overhead.

Cost math for all-Vultr burst upper bound: `100 seeds * 38.04h/seed * $0.476/h = $1,810`, before overhead.

**Expected wall clock**:

- about `5.5`-`7` days under the current five-Hetzner / 30-Vultr caps, depending on deterministic-runtime overhead and whether the Hetzner 48-seed leg remains the critical path
- about `6`-`8` days for a 100-seed all-Vultr burst under the current 30-machine cap, unless the actual candidate workload is materially faster than the historical broad Phase 1 surface
- faster luxury runs require either a Hetzner quota increase, a shorter candidate workload, or a different proven provider pool

Activation blockers: candidate intake, authorized live resource inventory, and split-aware cloud launcher.

Vultr cap check: the all-Vultr raw compute estimate is about `$1,810`, but a `10%`-`20%` deterministic-runtime buffer plus provisioning/retry overhead can push the launch close to the user-reported `$2500` ceiling. Re-quote before provisioning.

Capacity feasibility: the default luxury allocation fits inside the current Hetzner `5`-machine cap and Vultr `30`-machine cap, using 5 Hetzner boxes and 26 Vultr boxes. A 100-seed all-Vultr burst also fits the 30-machine cap, but must be re-costed against the `$2500` ceiling immediately before launch.

OCI inclusion rule:

- verify live `VM.Standard.E5.Flex` AMD OCPU service limit
- verify credentials through `~/.oci/config` or the expected keychain entries
- run a one-seed retrieval canary end-to-end before scaling
- keep OCI out of the pooled luxury result unless determinism/provider tags are recorded

The OCI one-seed retrieval canary is a small spend if it runs at the planned shape: `1 seed * 14.10h/seed * $0.398/h = $5.61`, before overhead.

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

1. Start at **budget** if the candidate is weak or exploratory.
2. Start at **medium** if the candidate is credible and Eric wants the real split audit.
3. Use **luxury** only after medium produces a result worth confirming, or when turnaround/provider diversity is worth a four-figure spend.

If no candidate stack is named, keep the split-audit state at `plan_blocked_no_viable_candidate`.

## Source Links

- Hetzner: https://docs.hetzner.com/general/infrastructure-and-availability/price-adjustment/
- Vultr: https://www.vultr.com/pricing/
- Oracle: https://www.oracle.com/cloud/iaas-paas/

## Verification

Docs-only plan. Verified locally:

```bash
git diff --check
```
