# Live-Forward Daily Ops Runbook

Status: active for the #16 `decision_weighted_lgbm_v0` live-forward slice.

This runbook covers the safe daily path for fresh-target research artifacts and
the pending deploy/restore sequence for the shadow scheduler watchdog fix. It
does not clear production model changes.

## Anchors

- Candidate: `decision_weighted_lgbm_v0`
- Candidate training freeze commit: `5004b1c8b093da0f8acb11bd728430ebacbf92d3`
- Live-forward logging checkout: parity-guard backport branch
  `live-forward-frozen-parity-backport` at
  `a8632ce6cc863e1dd55d58215b96b50828437263`
- Artifact family: `bts_candidate_ranked_slate_pair_v1`
- Live-forward output root:
  `data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD`
- Resolved artifact output root:
  `data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD`
- Current production deploy branch before the fix:
  `15da954142da62089bb198873a7cca94e69c659c`
- Next deploy target: `origin/main` containing PR #64 and PR #66 plus any
  later merged runbook or research-tooling PRs. Verify the exact SHA with
  `git rev-parse origin/main` immediately before pushing `deploy`.

The pre-registration document is
`docs/sota_audit/2026-05-08-fresh-audit-pre-registration.md`. It clears
pre-outcome research logging only. The command must not write production picks,
model caches, posts, cloud assets, or `deploy`.

## Automated Capture

Preferred daily operation is the guarded one-shot runner:

```bash
cd /home/bts/projects/bts
.venv/bin/python scripts/live_forward_capture_once.py
```

This runner is safe to invoke from a frequent timer:

- exits successfully while `data/picks/YYYY-MM-DD.json` is absent;
- refuses to export if the pick file already has a non-null result;
- refuses partial artifact directories;
- verifies an existing manifest instead of exporting again;
- writes `capture_status.json` beside the artifact after export/verify attempts.

Install the user timer on Hetzner only after the production checkout contains
`scripts/live_forward_capture_once.py`:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/systemd/bts-live-forward-capture.service ~/.config/systemd/user/
cp scripts/systemd/bts-live-forward-capture.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now bts-live-forward-capture.timer
```

The timer polls every 15 minutes. It should replace the need for a human to
remember the daily pre-outcome capture.

## Safety Gates

Before running anything against production:

1. Confirm no live deploy is in progress.
2. Confirm the scheduler heartbeat is fresh and the dashboard is serving.
3. Do not kill broad process groups. Use only named `systemctl --user` services
   when a service restart is needed.
4. Do not push a stale local `main` branch to `deploy`. Push a verified remote
   ref or explicit commit.
5. Keep `shadow_model = false` until the watchdog fix is deployed and verified.
6. Do not deploy during lineup lock or result-polling windows unless the current
   incident requires it.

The shadow-model temporary mitigation on 2026-05-09 was:

```bash
shadow_model = false
```

in `/home/bts/.bts-orchestrator.toml`, with backup:

```bash
/home/bts/.bts-orchestrator.toml.bak-20260509-shadow-watchdog
```

Prefer editing only the `shadow_model` line during restore so unrelated config
changes are not rolled back by an old backup.

## Daily Live-Forward Export

Run this after production has refreshed the daily data snapshot and before the
slate outcomes are known.

On `bts-hetzner`:

```bash
cd /home/bts/projects/bts-live-forward
git rev-parse HEAD
```

The expected live-forward logging worktree commit is:

```bash
a8632ce6cc863e1dd55d58215b96b50828437263
```

Then export one date into a date-specific output directory:

```bash
cd /home/bts/projects/bts-live-forward
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  BTS_LGBM_DETERMINISTIC=1 BTS_LGBM_RANDOM_STATE=42 \
  nice -n 19 .venv/bin/bts experiment export-live-candidate-artifacts \
  --date YYYY-MM-DD \
  --candidate decision_weighted_lgbm_v0 \
  --output-dir /home/bts/projects/bts/data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD \
  --data-dir /home/bts/projects/bts/data/processed \
  --production-pick-file /home/bts/projects/bts/data/picks/YYYY-MM-DD.json \
  --top-n 10 \
  --no-refresh-data
```

Do not reuse an output directory across dates. The manifest is single-date and
the command will overwrite the same date directory if pointed at an existing
path.

## Artifact Verification

After PR #64 is deployed, use the production verifier:

```bash
cd /home/bts/projects/bts
.venv/bin/bts experiment verify-candidate-artifacts \
  --artifact-dir data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD \
  --expected-candidate decision_weighted_lgbm_v0 \
  --expected-date YYYY-MM-DD \
  --expected-git-commit a8632ce6cc863e1dd55d58215b96b50828437263 \
  --expected-top-n 10 \
  --require-live-preoutcome \
  --require-production-pick-snapshot \
  --save data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD/verification.json
```

Expected verifier posture:

- `ok = true`
- `failure_count = 0`
- manifest `run_kind = live_forward_preoutcome`
- manifest `fresh_target_claim = true`
- manifest `production_deploy_claim = false`
- production and candidate row counts equal `10`
- outcome fields are null
- `production_pick_snapshot` is present and date-matched

If the verifier fails, do not hand-edit the artifacts. Preserve the failure
report and investigate the export code or data snapshot.

## Post-Outcome Resolution

After the slate is final and the processed PA data includes the game outcomes,
resolve outcomes into a copied artifact directory. Do not mutate the original
pre-outcome artifact.

```bash
cd /home/bts/projects/bts
.venv/bin/bts experiment resolve-live-candidate-artifacts \
  --artifact-dir data/validation/decision_weighted_lgbm_v0_live_forward/YYYY-MM-DD \
  --output-dir data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD \
  --data-dir data/processed \
  --save data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD/resolution.json
```

Expected resolver posture:

- `complete = true`
- `missing_count = 0`
- resolved manifest `run_kind = live_forward_resolved`
- source manifest remains unchanged with null outcomes

If outcomes are incomplete, the resolver fails closed by default. Use
`--allow-partial` only for explicit forensics, and label those artifacts as
partial. Do not count partial artifacts toward the primary fresh-target slate
count.

Postponement handling:

- For actual BTS scoring, a locked pick whose game is postponed or cancelled is
  void for that slot only. It is neither a hit nor a miss and it does not wait
  for the future makeup game.
- If the other double-down slot played, score the day from the played slot only:
  void + hit advances by one; void + miss resets; both void leaves the streak
  unchanged.
- For live-forward artifacts, a postponed or void game normally has no PA row in
  `pa_YEAR.parquet`. The resolver must treat the left-only join as missing
  outcome evidence and fail closed unless `--allow-partial` is explicitly used;
  never coerce a missing/postponed outcome to `actual_hit = 0`.

Verify the resolved copy without the live pre-outcome null-outcome flag:

```bash
cd /home/bts/projects/bts
.venv/bin/bts experiment verify-candidate-artifacts \
  --artifact-dir data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD \
  --expected-run-kind live_forward_resolved \
  --expected-candidate decision_weighted_lgbm_v0 \
  --expected-date YYYY-MM-DD \
  --expected-git-commit a8632ce6cc863e1dd55d58215b96b50828437263 \
  --expected-top-n 10 \
  --save data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD/verification.json
```

Once resolved, a local comparison can be generated:

```bash
cd /home/bts/projects/bts
.venv/bin/bts experiment compare-candidate-artifacts \
  --artifact-dir data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD \
  --save data/validation/decision_weighted_lgbm_v0_live_forward_resolved/YYYY-MM-DD/comparison.json
```

Single-day comparisons are monitoring evidence only. The #16 cycle verdict
requires the pre-registered accumulated fresh-target analysis, not a one-day
scorecard delta.

## Deploy PR #64 and PR #66

The deploy bundle contains:

- PR #64: candidate artifact verifier CLI
- PR #66: scheduler watchdog around shadow prediction

Use a remote ref or explicit SHA. Do not rely on local `main` unless it has just
been verified clean and current.

```bash
git fetch origin
git rev-parse origin/main
git rev-parse origin/deploy
git push origin origin/main:deploy
```

Wait for the deploy workflow canary. The deploy workflow resets production to
`origin/deploy`, syncs dependencies, restarts `bts-scheduler` and
`bts-dashboard`, then checks:

- scheduler active
- dashboard active
- dashboard HTTP 200

After the workflow passes, verify production manually:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bts-hetzner \
  'cd /home/bts/projects/bts && git rev-parse HEAD && systemctl --user is-active bts-scheduler.service && systemctl --user is-active bts-dashboard.service'
```

The expected production HEAD after this deploy is the current `origin/main`
commit that includes PR #66.

## Restore Shadow Model

Restore shadow only after the deploy fix is live.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bts-hetzner \
  'perl -0pi -e "s/shadow_model\\s*=\\s*false/shadow_model = true/" /home/bts/.bts-orchestrator.toml && systemctl --user restart bts-scheduler.service'
```

Then verify:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bts-hetzner \
  'grep -n "shadow_model" /home/bts/.bts-orchestrator.toml && systemctl --user status --no-pager -l bts-scheduler.service | sed -n "1,30p"'
```

Watch the heartbeat through one sleep/poll cycle. Tomorrow's shadow artifact
should be generated by the watchdog-aware code path and should not starve
systemd.

## Rollback

If the deploy canary fails, use the workflow rollback result as the source of
truth before making manual changes.

If the scheduler becomes unhealthy only after restoring shadow:

1. Set `shadow_model = false`.
2. Restart only `bts-scheduler.service`.
3. Confirm heartbeat freshness.
4. Preserve logs for the failing shadow run.

Do not delete live-forward artifacts during rollback. They are research
evidence, not production state.

## Evidence To Record

For each date, record:

- export command date and host
- frozen worktree `git rev-parse HEAD`
- production HEAD at the time of export
- manifest path
- verifier path and pass/fail summary
- top production candidate and probability
- top candidate-model candidate and probability
- whether outcomes were still unknown at export time
- any deploy or config changes made during the same operating window

For the agent bus, send concise evidence rather than a narrative transcript.

```bash
/Users/stone/agent-room/bin/agent-bts-bus send \
  --from codex \
  --to claude \
  --kind evidence \
  --thread bts-motor-session \
  --body "YYYY-MM-DD live-forward export complete: manifest=... verify_ok=... prod_head=... frozen_head=..."
```

## 2026-05-09 Day 1 Baseline

Day 1 was exported from the frozen worktree and verified before outcomes were
known.

- Date: `2026-05-09`
- Frozen worktree: `5004b1c8b093da0f8acb11bd728430ebacbf92d3`
- Production HEAD: `15da954142da62089bb198873a7cca94e69c659c`
- Manifest:
  `data/validation/decision_weighted_lgbm_v0_live_forward/2026-05-09/manifest.json`
- Verifier result: `ok=true`, `failure_count=0`
- Production top rank: Chandler Simpson (`802415`), `p_game_hit=0.764834`
- Candidate top rank: Chandler Simpson (`802415`), `p_game_hit=0.769287`
- Production pick actually locked later: Chandler Simpson, `p_game_hit=0.781091`
- Double-down pick: Carlos Cortes, `p_game_hit=0.730477`

The frozen artifact ranks and production scheduler picks are related but not
identical evidence streams. Treat the artifact pair as the #16 fresh-target
research log and the scheduler picks as production state.

The probability difference between the frozen artifact production rank
(`0.764834`) and the deployed production scheduler lock (`0.781091`) is expected
code drift between the frozen launch SHA (`5004b1c8...`) and the deployed
production SHA (`15da9541...`). For the #16 fresh-target comparison, the
candidate is evaluated against the production baseline at the frozen SHA so the
candidate and baseline are apples-to-apples. The deployed scheduler pick remains
the operational production stream.

The missing `2026-05-09.shadow.json` is an expected one-day gap from the
temporary `shadow_model = false` mitigation, not evidence about the candidate.
