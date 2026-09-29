# W1.5 incident register — record drafting guide

You draft register records for BTS (an MLB "Beat the Streak" pick service) from repository evidence only.
Every record is reviewed by the lead afterwards; your job is an accurate, well-evidenced DRAFT.

## Hard rules
- Read-only on the repository `/Users/eric/projects/bts`: `git show`, `git log`, reading files. Never edit it,
  never commit, never read anything under `/Users/eric/projects/bts/data/`, no ssh, no network, no `gh`.
- Write only your output files: `<scratch>/drafts/<EID>.json` (one per episode) and `<scratch>/drafts/<batch>.notes.md`.
  (`SCR` = `<scratch>`)
- Never state a fact you have not read in a cited artifact. Unknown = `"unknown"` (or a bound), never a guess.
- Do not restate 2026 pick outcomes (who hit, streak values, win/loss figures, calibration numbers) unless the
  incident IS about that value (e.g. "local streak 10 vs contest 8"); when you do, add `"exposure_row": "X-01"`
  to that evidence item.

## Inputs
- `<scratch>/packs/<EID>.json`: the episode row (working table: id, dates, one-line summary, classes, preliminary
  disposition/tier, sources) and, for each cited commit: authored time (ISO with offset), subject, full body,
  changed files, `first_live_log` (log-bound deploy interval from retained deploy logs: `live_by`,
  `not_live_before`, installed `sha`) and `first_success_deploy_candidate` (first successful deploy run whose
  head contains the commit: run id, time, head sha, whether its log was retained).
- The schema: `<scratch>/build/scripts/audit/incident_register/record_schema.json`.
- The design (dispositions, classes, evidence rules): `/Users/eric/projects/bts/docs/superpowers/specs/2026-09-29-incident-register-design.md` §2, §3, §6.1, §7.
- The model record: `<scratch>/drafts/E83.json` — copy its conventions exactly.
- Repo docs you may cite: `docs/audit/**`, `ARCHITECTURE.md`, `INCIDENT.md`, `docs/optimization-ideas.md`, plans/specs.
  `git log --all --format='%h %aI %s' -S<text>` / `--grep` to find related commits; `git show <sha>` for diffs.

## Out-of-scope rows
Rows whose preliminary disposition is `EX` (model quality, product/terms-of-use decisions, research fleets that
are not part of the deployed service or its preservation) get NO record file. Write one line in your notes file
instead: id, why it is out of scope (design §1: the register measures no model quality; §2 scope), and where it
belongs (e.g. "W1.4b bin-collapse read"). If on reading the evidence it IS in scope, draft it normally and say so.

## Ids
`E<n>` → `"id": "I-<n zero-padded to 3>"` (E7 → I-007, E103 → I-103); `L0<n>` → `I-20<n>` (L06 → I-206);
`EX<n>` → `I-30<n>`. `"working_id"` = the episode id. `related` uses the same mapping.

## Field rules
- **disposition** (exactly one; design §2):
  - `observed_incident` — a live contract deviation evidenced by a machine observation or a CONTEMPORANEOUS
    operator report: a commit body or memo written ≤ 48 h after the event that describes what was seen live.
    A body written later is `inference` (later analysis) and cannot establish it alone.
  - `deployed_latent_defect` — a defect in deployed code/config, established by code + deployment evidence,
    with no evidenced firing. Keep it only when it matters to watchdog or restore coverage; otherwise say so in
    notes and still draft it (the lead decides).
  - `near_miss_control_held` — hazard present, an existing control held (name the control).
  - `pre_ship_exclusion` — introduced and fixed inside an unshipped change, or a test-only defect
    (`counted: false`).
  - `unresolved_candidate` — evidence insufficient: `missing_evidence` must name exactly what would settle it
    (e.g. "box journal for 2026-04-22 showing the restart burst — Phase 2 Route R", "deploy log for the fix
    (expired; Phase 2 deploy_history.txt)").
  `counted` is `false` only for `pre_ship_exclusion`.
- **classes**: primary + secondary from D delivery · E entry · G grading & state · A alerting · L liveness ·
  P decision contract · R research integrity · X external dependency · S preservation & deploy · U display.
- **tier**: `A` = delivery/entry/contest/state impact, alert storm, liveness failure, loss of recoverability, or
  material research-integrity failure; `B` = display/wording, research issue without data loss, near miss;
  `tier_pending` = impact unknown (then `tier_reason` is required). Give `tier_reason` always. If an UNFIXED
  residual can still change delivery/entry/state/alerting/recoverability, the tier is `A` (B → A rule).
- **axes**: `delivery_mode`: `public` for 3/29–5/19 (only public Bluesky posting existed; DM delivery merged
  5/19, `1c6fad6`), `unknown` for 5/20–7/05, `dm` for 7/06–9/13 (the 7/06 audit memo records prod
  `pick_delivery = "dm"`), `private` from 9/14; use `unknown` if the record is not about a specific date.
  `policy_objective`: `reach57` for decision-related occurrences before 9/03 (the reach-57 MDP; `0abf503`
  added the E[season-best] tail regime on 9/03), `emax_season_best` only for tail-regime decisions,
  `none` when no decision is involved, else `unknown`. `stream`: production | shadow_v1 | shadow_v2 |
  skip_shadow | live_forward | d8_research | leaderboard | backup | deploy | cron_job | dashboard.
  `authority`: what was authoritative for the delivery/entry/grade concerned — `contest` | `local` |
  `not_applicable` | `unknown`.
- **contract**: the behaviour that was violated, in one sentence, with its source (code constant, doc,
  rules page, design section). No contract you cannot source.
- **mechanism**: numbered causal links; each cites code at the DEFECTIVE ref = the parent of the fix commit
  (`git rev-parse --short <fix>^`), `symbol` = function name, `ref_basis`: `log` only if the pack shows that
  ref (or a descendant without the fix) installed in a retained log at the time; else `candidate`.
- **fix** (one entry per mechanism link):
  - `implemented`: `{"commits": [...], "complete_fix_set": true|false}` or `"unfixed"` / `"config_only"`.
  - `deployed`: if the fix commit's `first_live_log.not_live_before` is non-null →
    `{"basis": "log", "sha": <first_live_log.sha>, "live_by", "not_live_before", "run_id" (the run whose
    deployed SHA is that sha — find it in docs/audit/2026-09-29-incident-register-evidence/deploy/deploy_runs.json),
    "evidence": [ev..]}`; otherwise, if `first_success_deploy_candidate` exists →
    `{"basis": "candidate_ancestry", "sha": <candidate.head_sha>, "live_by": <candidate.created_at>, "run_id"}`
    and a note "log bound: live by <first_live_log.live_by>"; otherwise `"unknown"`. Config-only fixes on
    the box: `"unknown"` unless an operator report dates it (`"basis": "operator_report"`).
  - `mitigated`: operator action + bound, from evidence; else `"none"`/`"unknown"`.
  - `verified_recovered`: `"unknown"` unless evidence shows a verified recovery.
- **fixtures**: leave all four lists empty and add the note "fixtures: filled from Task 5/6 acceptance objects
  at build time". If the pack shows a test that reproduces the incident, name it in notes.
- **residual**: every known remaining gap or review deferral (commit bodies often list them) with
  `reaches_production` true/false and its source.
- **watchdog**: the W4 rank-2 check that would have caught it: `boundary` ∈ delivery | entry | restart |
  singleton_slate | private_vs_contest | grading | preservation | none; `trigger` (an observable condition on
  durable state/logs, with timing); `recovery_assertion` (what proves recovery). Use `none` with a reason in
  `trigger` only for display-only or pre-ship items.
- **evidence**: ids `ev1..evN`; `kind` ∈ machine_observation (log line, state field, deploy run, contest-ledger
  line) | contemporaneous_operator_report (commit body/memo ≤ 48 h after the event; give `written_at` = commit
  authored time or memo date) | inference (code or later analysis); `strength` = `primary` (the artifact
  itself) or `reported` (a note repeating a claim whose artifact is not located); `locator` precise ("commit
  <sha> message body", "docs/audit/<file> §<section>", "deploy run <id> (deploy_runs.json)"); `what` = one
  sentence of what it shows. Deploy runs from `deploy_runs.json` are machine observations.
- **occurrences**: one per occurrence (date). `onset`, `first_detectable`, `operator_awareness`,
  `operator_action`, `mitigation`, `restored_verification` are bounds: `{"at": t, "evidence": [...]}`,
  `{"not_before"/"not_after": t, "evidence": [...]}`, or `"unknown"` / `"none"` / `"not_applicable"`.
  Times: ISO with offset (`2026-08-13T12:49:00-04:00`) or date-only (`2026-08-13`, read as ET). The fix
  commit's authored time is a valid `not_after` for operator awareness. `first_machine_detection`:
  `{"detector": ..., "at": bound}` or `"none"`/`"unknown"`. `alert`: attempted/confirmed/failed bounds.
  `latencies` (minutes, `{"min_minutes", "max_minutes"}`) only when both endpoints are bounded; else
  `"unknown"` or `"not_applicable"` (e.g. no machine detection happened).
- **routes**: `H` = cited commit shas; `R` = `"pending_phase2"`; `X` = doc paths / issue or PR numbers.
- **notes**: short strings for anything the lead must know (doubts, alternative readings, splits/merges).

## Procedure per episode
1. Read the pack; `git show` each commit (diff) and read any cited doc. Search for related commits if the
   pack looks incomplete (`git log --grep`, `-S`). Verify the preliminary row — it can be wrong.
2. Decide disposition + tier with the rules above; if the episode actually bundles distinct defects, draft ONE
   record and say "split?" in notes with the proposed parts.
3. Write `<scratch>/drafts/<EID>.json`, then validate YOUR files (draft mode), e.g.:
   `cd <scratch>/build && UV_CACHE_DIR=/tmp/uv-cache uv run --with jsonschema==4.23.0 python -c "import json,sys; from scripts.audit.incident_register.records import validate; rs=[json.load(open(f)) for f in sys.argv[1:]]; print([e for e in validate(rs, publish=False) if 'unknown related record' not in e])" <scratch>/drafts/E07.json <scratch>/drafts/E08.json`
   Fix every error (a `related` id pointing to another batch is fine — it is filtered above; list it in your notes).
4. In `<scratch>/drafts/<batch>.notes.md`: per record one line — id, disposition, tier, and any doubt.

Reply at the end with: records written, their dispositions/tiers in a compact table, and the doubts list.
