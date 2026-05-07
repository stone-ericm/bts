# Split-Audit Compute Budget Options

- **Generated**: 2026-05-07
- **Depends on**: `docs/sota_audit/2026-05-07-real-split-audit-plan.md`
- **Current plan verdict**: `plan_blocked_no_viable_candidate`

## Verdict

Structured verdict: `budget_options_documented_blocked_on_activation`.

The tiers are ready for planning, but none should run until candidate intake, live resource inventory, and split-aware cloud orchestration are complete. Eric's 2026-05-07 Hetzner/Vultr capacity numbers and the read-only OCI quota checks are recorded below as partial inventory, not a substitute for launch canaries, cost re-quotes, and artifact-provenance checks. If this pricing snapshot is more than 30 days old when activated, re-quote provider rates before provisioning.

## Recommendation

Use the **medium** tier as the default once Eric names a candidate stack and compute budget. It gives a 48-seed audit comparable to the existing Hetzner 48-seed evidence, keeps spend controlled, and avoids the large Vultr premium unless wall-clock time or provider-diversity evidence is worth paying for.

Use **budget** only as a smoke audit or falsification pass. Use **luxury** only for a candidate that has already survived the medium tier or for a candidate Eric is seriously considering taking through production gates.

All tiers remain blocked until:

- candidate intake is complete
- live Hetzner/Vultr/OCI resource inventory is authorized and recorded; the capacity notes below are partial inventory only
- `scripts/audit_driver.py` gets split-flag pass-through or a dedicated split-audit launcher exists
- OCI-specific launch readiness is fixed before any OCI box is requested: `oci-subnet-ocid` must be available to the driver, a one-seed canary must retrieve artifacts end-to-end, and AD spreading must exist before requesting more than about 10 OCI E5 boxes

## Capacity Snapshot

Eric provided Hetzner/Vultr capacity on 2026-05-07. Eric then authorized OCI verification with "lets verify" on 2026-05-07. OCI was checked read-only through the OCI CLI; no instances were launched and no spend was incurred.

| Provider | Current limit | Increase path / lead time | Evidence source | Planning implication |
| --- | --- | --- | --- | --- |
| Hetzner | `5` machines | Eric can ask for an increase in about `4` days | User-provided 2026-05-07 | Use up to five `CPX62` boxes now. Revisit medium/luxury wall-clock estimates after the quota increase request is available. |
| Vultr | `30` machines / `$2500` ceiling | Not specified | User-provided 2026-05-07 | Enough for the default luxury Vultr leg and likely enough for all-Vultr raw compute, but overhead and deterministic-runtime buffer must be checked against the `$2500` ceiling before launch. |
| OCI | `standard-e5-core-count`: `83` OCPU available and `0` used in each Ashburn AD; `standard-e5-memory-count`: `1250` GB available and `0` used in each Ashburn AD | No increase needed for the variants below; request more only if a future plan needs more than 30 E5 boxes | Read-only OCI CLI checks 2026-05-07 | At the planned 8 OCPU / 32 GB shape, quota supports 10 boxes per AD, 30 boxes total. This is enough to make OCI a meaningful accelerator once canary/retrieval and provenance gates pass. |

OCI quota math: `83 OCPU/AD / 8 OCPU/box = 10` full boxes per AD, with OCPU as the binding constraint. Memory is not binding because `1250 GB/AD / 32 GB/box = 39` boxes per AD. Across three Ashburn ADs, the verified quota is `3 * 10 = 30` planned boxes.

OCI launch readiness is not complete. `~/.oci/config` works for read-only service-limit queries, `VM.Standard.E5.Flex` is available in Ashburn, the project already depends on `oci>=2.170.0`, and a public subnet named `public subnet-bts-audit-vcn` exists with public IP assignment allowed. However, `scripts/audit_driver.py` currently requires the subnet OCID in the macOS Keychain entry `oci-subnet-ocid`, and that entry was missing during verification. Add that secret or teach the driver a config/env fallback before running the one-seed canary.

Using the full 30-box OCI quota also likely needs a small driver change. The current `OCIProvider.create()` resolves all ADs, but each create attempt starts from the first AD and treats exhausted `LimitExceeded` retries as fatal instead of trying the next AD. That is acceptable for 1-4 box canaries, but scaling beyond roughly 10 E5 boxes needs explicit AD spreading or LimitExceeded fallback to reach the verified cross-AD capacity.

## Pricing Snapshot

Cloud compute prices were checked on 2026-05-07. Treat these as planning estimates, not invoices. They exclude storage, network transfer, public IPv4 extras, failed provisioning retries, and local development time.

| Provider | Planning shape | Unit price used | Source |
| --- | --- | ---: | --- |
| Hetzner | `CPX62`, 16 vCPU / 32 GB / 640 GB, Germany/Finland | `$0.0953/hour` | Hetzner 2026-04 cloud price adjustment |
| Vultr | Optimized Cloud Compute, 16 vCPU / 32 GB / 300 GB | `$0.476/hour` | Vultr pricing page |
| OCI | `VM.Standard.E5.Flex`, 8 OCPU / 32 GB | about `$0.3984/hour` | Oracle E5 Flex OCPU + memory pricing |

OCI remains canary/retrieval-gated in this repo even though live quota is now verified. Use it as an accelerator only after a one-seed launch retrieves artifacts cleanly and records provider/determinism provenance.

## Runtime Calibration

These tiers use local historical run logs, not live cloud API checks:

| Surface | Boxes / seeds | Observed per-seed runtime | Notes |
| --- | ---: | ---: | --- |
| `data/hetzner_results/audit_full_48seed_v2` | 4 boxes / 48 seeds | mean `14.10h`, range `13.46h`-`14.80h` | Best cost anchor for fixed-n large audits. |
| `data/vultr_results/audit_ext_n100_v4` | 26 boxes / 52 seeds | mean `38.04h`, range `30.69h`-`44.15h` | Better burst capacity, much higher cost per seed-hour. |
| OCI | live quota only | unknown | No current retrieved split-audit runtime surface in this plan. The one-seed canary must measure runtime and retrieval before OCI enters a launch budget. |

These are conservative for a single exact candidate because the historical surfaces were broad Phase 1 screening runs. OCI now has verified E5 quota, but no current retrieved split-audit runtime surface in this plan. The first split-aware launcher should render the exact command and re-estimate cost from the actual candidate workload before provisioning.

Runtime may be higher under deterministic training than these historical logs. `BTS_LGBM_DETERMINISTIC=1` sets LightGBM deterministic mode plus `force_row_wise=True`; treat `10%`-`20%` extra runtime as a reasonable planning buffer until the split-aware launcher has a fresh dry run.

Cross-provider pooling has a determinism caveat. `CLAUDE.md` records that OCI E5.Flex drifted versus Hetzner on identical seed `42` without the determinism flag. Luxury/provider-diverse runs must either set `BTS_LGBM_DETERMINISTIC=1` across all providers after a deliberate re-baseline, or keep provider tags explicit and avoid interpreting pooled cross-provider seeds as pure seed variation.

## OCI Acceleration Sensitivity

This is not a launch budget. It shows the upside if a canary proves that OCI E5 runtime and retrieval are healthy for the exact split-audit command.

If OCI runtime matches Hetzner's `14.10h/seed` anchor, plausible acceleration looks like:

| Variant | Shape | Raw compute sensitivity | Wall-clock sensitivity | Extra blockers |
| --- | --- | ---: | ---: | --- |
| Medium acceleration | 48 seeds on 5 Hetzner + 4 OCI | about `$156` | about `3.1` days before deterministic-runtime buffer | Does not need full cross-AD scaling, but still needs `oci-subnet-ocid` and one-seed retrieval canary. |
| Luxury balanced | 48 seeds on 5 Hetzner + 4 OCI, plus 52 Vultr seeds on 26 Vultr boxes | about `$1,098` | about `3.2` days before deterministic-runtime buffer | Same as medium acceleration; keeps Vultr provider-diversity leg. |
| Luxury OCI-heavy | 100 seeds on 5 Hetzner + 20 OCI | about `$476` | about `2.4` days before deterministic-runtime buffer | Requires multi-box OCI canary and driver AD-spreading support. |

If OCI runtime lands closer to Vultr than Hetzner, these wall-clock and cost sensitivities degrade. Do not use the OCI variants as budget commitments until the canary supplies measured runtime, retrieve status, determinism metadata, and provider provenance.

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

Capacity feasibility: the Hetzner-only default fits inside the current Hetzner `5`-machine cap, but uses all five boxes. A single provisioning failure degrades throughput unless the run waits for the quota increase or reallocates to another provider. Any faster Hetzner-only medium variant needs the quota increase Eric can request in about `4` days. If speed matters enough to spend more, revisit the OCI acceleration sensitivity after the one-seed canary measures runtime and retrieval.

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
- optional OCI E5 Flex canary and acceleration only after retrieval/provenance checks pass

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

Activation blockers: candidate intake, authorized live resource inventory, and split-aware cloud launcher.

Vultr cap check: the all-Vultr raw compute estimate is about `$1,810`, but a `10%`-`20%` deterministic-runtime buffer plus provisioning/retry overhead can push the launch close to the user-reported `$2500` ceiling. Re-quote before provisioning.

Capacity feasibility: the default luxury allocation fits inside the current Hetzner `5`-machine cap and Vultr `30`-machine cap, using 5 Hetzner boxes and 26 Vultr boxes. A 100-seed all-Vultr burst also fits the 30-machine cap, but must be re-costed against the `$2500` ceiling immediately before launch. Verified OCI E5 quota supports up to 30 planned OCI boxes across Ashburn, but OCI remains opt-in until launch readiness, canary evidence, and driver AD-spreading for variants above about 10 OCI boxes.

OCI inclusion rule:

- live `VM.Standard.E5.Flex` capacity: verified read-only on 2026-05-07, re-check if stale at launch time
- credentials/launch readiness: outstanding; `~/.oci/config` works, but `oci-subnet-ocid` was missing from Keychain
- one-seed retrieval canary: outstanding; must finish end-to-end before scaling
- AD spreading or LimitExceeded-to-next-AD fallback: outstanding before requesting more than about 10 OCI E5 boxes
- determinism/provider tags: outstanding; keep OCI out of the pooled luxury result unless tags are recorded

The OCI one-seed retrieval canary is a small spend if it runs at the planned shape: `1 seed * 14.10h/seed * $0.3984/h = $5.62`, before overhead.

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

Docs-only plan plus read-only OCI inventory. Verified locally:

```bash
git diff --check
oci limits service list --compartment-id <tenancy> ...
oci limits value list --compartment-id <tenancy> --service-name compute ...
oci limits resource-availability get --compartment-id <tenancy> --service-name compute --limit-name standard-e5-core-count ...
oci limits resource-availability get --compartment-id <tenancy> --service-name compute --limit-name standard-e5-memory-count ...
oci compute shape list --compartment-id <tenancy> --shape VM.Standard.E5.Flex ...
oci network subnet list --compartment-id <tenancy> ...
```
