# BTS lead kickoff brief (handoff of 2026-10-06)

You are the new BTS lead session, started fresh by the herdr manager in workspace `wA`. The previous lead (session `1cf2ed05…`, pane `wA:p1`) ran the C1 research cycle from 10/04 to 10/06 and finished with today's D7 deploy. It took no work steps after writing this brief.

**Read first, in this order:**
1. `CLAUDE.md` (project rules);
2. this brief;
3. the BTS memory hub `~/projects/claude-shared/memory/bts_index.md` (its STATE line);
4. the C1 cycle index `docs/sota_audit/2026-10-04-c1-cycle-index.md`;
5. the exposure register `docs/audit/2026-09-22-exposure-register.md` (§C rows from `C1-…` down, and `D7-2026-10-06`).

Before branching, run `git fetch origin && git log main..origin/main --oneline`, because other sessions push.

## 1. Current state

**Production (bts-hetzner) runs `f882411`, deployed 2026-10-06 at 11:00:46 EDT** (GitHub run 37482265317; register row **D7-2026-10-06**).
- **What shipped** (Eric's D7 at 10:07 and his 10:13 extension, both relayed by the manager):
  - **C1/C2** (`04cde0a` + `2769a7e`): private delivery never touches a transport; a legacy `shadow_mode`-only delivery config is refused.
  - **P4** (`1c0e759` + `7da1b3b`): the declared `[scheduler].entry_intent` and an intent-aware `scripts/cron-setup-hetzner.sh`. `show` and `install` now refuse without an agreeing intent.
  - **P1** (`5f5217e` … `3cbe960`): a pick-entry receipt per `check-pick-entered` run (`docs/ops/pick-entry-receipt-v1.md`).
  - **P2** (`5c87de6` … `65a28a6`): a reconcile receipt per `bts reconcile` run (`docs/ops/reconcile-receipt-v1.md`).
  - **Slate v2** (`a6f80b5`): each slate row persists `game_time` and the schedule `status`.
  - **The C-03 reconcile cutoff** (`ce6676d`): reconcile never regrades a date past 08:00 ET the next day.
- **What was excluded and is not on main:**
  - **The W0 watchdog:** removed from main in `f882411`, because `cli.py` had registered it as `bts watchdog`. Resume from `c429c41`.
  - **The 4a serving witness:** reverted in `6a419a6`. Resume from `e66b440`.
- **Not deployed and not runnable:** the deferred research code (`scripts/audit/c1_r4a/`, `c1_r3/`, `c1_r4b/`) is on main but closed, because its `admission.json` is null.
- **Gates passed before the deploy:**
  - a whole-range deploy-gating review: r1 BLOCK, then r2 SIGN (`docs/audit/2026-10-06-d7-deploy-codex-r1.md`, `-r2.md`);
  - the seven producer-set certificates re-run with the frozen W1.5 tooling at `f882411` and accepted (`docs/audit/2026-10-06-d7-recert/`);
  - the fast suite at `f882411`: 3907 passed.
- **Box config `~/.bts-orchestrator.toml`:** `pick_delivery = "private"` (the season is over and the box is silent), plus `entry_intent = "research"`, added 10/06. The backup from before that edit is `~/.bts-orchestrator.toml.bak-20261006-pre-d7`; the season-over snapshot is `.bak-20260914-season-over`.
- **The crontab:**
  - **Installed 10/06** with `entry_intent = "research"`. It is byte-identical to `~/crontab.d7-installed-20261006`.
  - **What changed against the pre-install backup `~/crontab.bak-20261006-pre-d7-install`:**
    - added: the C-03 line `40 7 * * * … uv run bts reconcile >> /home/bts/logs/cron.log 2>&1 # BTS-HETZNER`;
    - dropped: the commented-out season-over `check-pick-entered` line;
    - dropped: 5 blank lines (the manager's call under the 10/03 delegation, not Eric's).
  - **What it holds now:** 18 BTS lines, and no entry checker.
- **The scheduler:** idle daily until 10:00 ET (no games; postseason). The dashboard returns HTTP 200 on the tailnet.
- **The deploy smoke test (10/06 11:12):** a manual `bts reconcile` exited 0 with "No scoring changes detected. Streak: 0". It published a sealed receipt in `data/health_state/reconcile_receipts/2026-10-06/`. That receipt is your baseline:
  - `outcome: completed`, `corrections: []`, `degraded: []`, `replay: {state: unavailable}`;
  - 8 days, all `past_cutoff`;
  - `producer.revision` = `f882411…`.

  `data/picks/streak.json` is `{"streak": 0, "saver_available": false, …}`, mtime 2026-09-18 23:06, unchanged.
- **Season start (checklist `docs/ops/2027-season-start.md`):** B1 is done. A5 still needs, at season start and with Eric's D6:
  - `pick_delivery = "dm"`;
  - `entry_intent = "enter"`;
  - `cron-setup-hetzner.sh install`, after `set -a && . ./.env && set +a`;
  - a restart inside a sleep window.

**Where the records live:**
- **The cycle index:** `docs/sota_audit/2026-10-04-c1-cycle-index.md`.
- **Rulings and deferrals:** the register's rows `C1-…` and `D7-2026-10-06`; corrections in `docs/audit/2026-09-corrections-index.md` (C-06, C-07).
- **Every Codex review:** `docs/audit/2026-10-0{4,5,6}-c1-*-codex-*.md`.
- **Plans:** `docs/superpowers/plans/2026-10-0{4,5,6}-c1-*.md`.
- **Specs:** `docs/superpowers/specs/2026-10-0{5,6}-c1-r2-*.md`.

## 2. First task: verify today's deploy (Wed 10/07, after 07:45 ET)
The first scheduled P2 receipts come from the 02:00 and 07:40 ET reconcile crons on 10/07. Check them read-only (`ssh bts-hetzner`, user `bts`, venv python):

```bash
ssh bts-hetzner 'cd ~/projects/bts && ls -la data/health_state/reconcile_receipts/2026-10-07/ \
  && .venv/bin/python - <<PY
from pathlib import Path
from bts.reconcile_receipt import discover          # sealed, untombstoned receipts only; run_date is a STRING
rs = discover(Path("data/picks"), "2026-10-07")
print(len(rs))
for r in rs:
    print(r["started_at"], r["outcome"], r["error"], r["corrections"], r["replay"], r["degraded"],
          r["producer"]["revision"][:7], sorted({d["state"] for d in r["days"]}))
PY
grep -n "Reconciling\|Streak:\|CORRECTIONS\|Traceback\|Error" ~/logs/cron.log | tail -6
stat -c "%y %s" data/picks/streak.json; cat data/picks/streak.json
crontab -l | cmp - ~/crontab.d7-installed-20261006 && echo crontab-unchanged
git rev-parse --short HEAD'
```

**Pass:**
- **Receipts:** exactly two sealed receipts for 2026-10-07 (one started near 02:00, one near 07:40). Each has a `.json.sealed` sidecar and no `.failed` file.
- **Each receipt:** `outcome == "completed"`, `error` null, `corrections == []`, `degraded == []`, revision `f882411`, and every `days[].state` is `past_cutoff`. `replay` is `unavailable` or `saved` with streak 0.
- **The cron log:** two "No scoring changes detected. Streak: 0" lines from 10/07, and no Traceback.
- **The rest:** `streak.json` content unchanged ({streak 0, saver false}), the crontab unchanged, and HEAD `f882411`.

**Failure:**
- **A missing receipt:** a cron didn't run, or publication failed. Look in `~/logs/cron.log` and for `.failed` tombstones.
- **A bad receipt:** `outcome` `raised` or `incomplete`, a non-empty `degraded`, any correction or `CORRECTIONS FOUND`, or a day not `past_cutoff` with a `write`.
- **Changed state:** any change to a pick file or `streak.json`, or a crontab that differs.

**What to do on a failure:** report it to the manager. A production fix needs Eric's D7. Don't hot-edit the box; see `INCIDENT.md`.

## 3. Then a new cycle, C2: the deferred items, one at a time

**Scope:**
- **Don't resume C1's exhausted rounds.** C1 deferred every candidate; its rounds are spent.
- **Start with a short planning pass:**
  - write a C2 proposal that sets each item's scope, order, compute cap, box-job budget and deadline (model it on `docs/audit/2026-10-04-d4-cycle-proposal.md`);
  - get Eric's approval through the manager before any build.
- **Infrastructure:** the C1 launcher, guard and ledger (`scripts/audit/c1/`, box limits SIGNED 10/06, r4) can be reused. Decide in the plan whether C2 gets its own ledger root and cap.

**Eric's order (10/06):**

### (a) The rank-2 watchdog (outcome / entry / restart)
- **Sources:**
  - registration `docs/sota_audit/2026-10-04-prereg-c1-watchdog.md` (FROZEN);
  - build plan `docs/superpowers/plans/2026-10-06-c1-r2-watchdog-build.md` (production code: reviewed until SIGN, then D7);
  - W1 evidence map and spec draft (unreviewed): `docs/superpowers/specs/2026-10-06-c1-r2-w1-evidence-map.md`, `…-w1-spec.md`.
- **Prerequisites now deployed:** C1/C2, P4, P1 and P2, all in `f882411`.
- **W0 (the foundation) history:**
  - **r1** BLOCK, B1–B7 (fixed in `ef1c917`);
  - **r2** BLOCK (fixed in `c429c41`);
  - **r3** BLOCK, `docs/audit/2026-10-06-c1-r2-watchdog-w0-codex-r3.md`. W0 was then deferred (row C1-r2-watchdog-deferral).
- **W0's open findings** (code at `c429c41`, `src/bts/watchdog/{root,clock,result,notify,runner,cli}.py`):
  - **R3-1:** a different checker can falsely recover a still-broken one, because `executed_ok` identifies an invocation by the callable's Python name.
  - **R3-2:** recovery overtakes an unresolved fault notice (`notify.py` `flush` sorts by `seq`, claims everything and keeps sending after a failure).
  - **R3-3:** a deleted notice is accepted, rewritten, and silences the alert.
  - **R3-4:** a SIGKILLed offending descendant is not observed as a gate failure (`run_confined`).
- **Deadline:** before activation, if Eric plays 2027; that is 2027 contest date 1 (Opening Day, late March 2027).

### (b) 4a (the calibration map), together with rank 3 (the PA-count model)
**4a:**
- **Sources:** registration `docs/sota_audit/2026-10-04-prereg-c1-calibration.md` (FROZEN); plan `docs/superpowers/plans/2026-10-06-c1-r4a-build.md` (its last section lists every open item); code `scripts/audit/c1_r4a/` at `0f9ffb0` (on main, closed).
- **History:**
  - **r1** (`docs/audit/2026-10-06-c1-r4a-codex-r1.md`): S (slate v2) SIGN, now deployed; R BLOCK, R1–R10.
  - **r2** (`…-r2.md`): S (the serving witness `e66b440`) BLOCK, S1–S3; R BLOCK, with R5–R9 closed. 4a was then deferred (row C1-r4a-deferral).
- **Open findings:**
  - **R1:** a zero-hit truncated suffix still counts as complete (`outcomes.py`). The fix reuses rank 3's reviewed team-PA accounting: a side's official `plateAppearances` equals its completed turns, which are `PA_ENDING_EVENTS` + `intent_walk` + `batter_interference` (`scripts/audit/c1_r3/count_verify.py`, `count_meta.py`). no_pa needs an explicit roster line.
  - **R2:** the contract must enforce the required flag, package and component coverage and the calibration-state combinations (`provenance.py`).
  - **R3:** the evaluation must retain the fit result it consumed, replay alone, and check the outcomes digest (`run.py`).
  - **R4:** the fit must be bound to the reviewed closure and the frozen fit manifest; a fitted map needs at least 25 known fit dates (`run.py`).
  - **R10:** production must persist `projected=false` for a posted lineup (`src/bts/model/predict.py` `_fetch_game_slots`), and `parsed` must count every parsed slate.
  - **The launcher path:** it must use `~/projects/bts-c1`.
  - **S1:** the model witness must come from the pipeline's actual cache/train decision and hash the serialization buffer.
  - **S2:** the calibration inputs and the fitted map must be witnessed.
  - **S3:** every ancillary hash failure must be contained (`src/bts/orchestrator.py` `predict_local`).
- **Deadlines:**
  - the serving witness and the `projected` fix deployed (D7) before 2027 date 1;
  - the calendar and the serving contract frozen under checklist A3 before the first 2027 capture;
  - X-32 published before any 2027 read;
  - the fit at 08:00 ET after date 40 and the evaluation after date 90;
  - the C1-style calendar stop at the contest end.

**Rank 3:**
- **Sources:** registration `docs/sota_audit/2026-10-04-prereg-c1-pa-count.md` (FROZEN); plan `docs/superpowers/plans/2026-10-05-c1-r3-count-build.md`; code `scripts/audit/c1_r3/`.
- **History:** the historical count build went r1 BLOCK → r2 BLOCK → r3 BLOCK → r4 BLOCK (`docs/audit/2026-10-06-c1-r3-build-codex-r{1,2,3}.md`, `docs/audit/2026-10-05-c1-r3-build-codex-r4.md`), then was deferred (row C1-r3-deferral). It never ran, and nothing read its inputs.
- **Open findings:**
  - **R3-5** (partly closed): receipt bytes, duplicates and availability.
  - **R4-1:** receipt witnesses are validated after duplicate collapse and per-outcome partitioning (`count_build.py:154–243`). Required: reject malformed or conflicting witnesses before joining or filtering.
- **Stored inputs:** the re-acquired 2021–25 feeds (12,148 games, receipts bound) are in `data/raw_c1/` on the box, with the caveats in the index row: per-game receipts only, and unreceipted schedules.
- **Deadlines:** its pre-lock count-forecast shadow archive, a production prerequisite, must run from the first 2027 game, so it must be deployed before Opening Day. The historical build can run any time before that.

### (c) 4b (the longest-streak policy, D1 Option 2)
- **Sources:** registration `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md` (FROZEN; the gate is Eric's trade table under D7); plan `docs/superpowers/plans/2026-10-04-c1-r4b-build.md`; code `scripts/audit/c1_r4b/` (built at `b2e7005`, later fixes `b56d870`..`3648120`, `ad29869`).
- **Correction to the handoff note:** r2 is **not** pending. Code went r1 BLOCK → r2 BLOCK → r3 BLOCK (a fresh session, whole range), then was deferred on 10/05 (Eric, Choice A; row C1-4b-deferral).
- **r3's findings** (`docs/audit/2026-10-05-c1-r4b-code-codex-r3.md`); the maths was unchallenged:
  - B1–B3 (claim release, `reviewed_commit` self-declaration, the manifest pin buffer) led to the shared admission gate `scripts/audit/c1/admission.py`; 4b must adopt it;
  - B4–B5 (cumulative CPU, overrun pause, launch lock) were fixed in the C1 box limits (SIGNED 10/06);
  - **still open:** B6 (`--verify` succeeding with unresolved requests or missing acquisition evidence) and B7 (a "parity-only" call that also runs an unregistered July sweep).
- **Deadline:** the trade table must be approved under D7 before any 4b policy affects picks; to matter for 2027, before Opening Day. It must also pass the checklist A2 tail/base pairing check.

### (d) P3 / W-state stays deferred
The draft `docs/superpowers/specs/2026-10-05-c1-r2-p3-consistent-capture-design.md` got design review d1 REVISE (`docs/audit/2026-10-06-c1-r2-p3-design-codex-d1.md`), then deferral (row C1-r2-p3-deferral). Reopen it only if the watchdog plan shows it is really needed.

## 4. Rules
- **The pace rule:** at most 2 Codex rounds per item (designs and research code). Without a SIGN the item is deferred, unless Eric or his delegate rules otherwise.
- **Production code:** reviewed until SIGN. Any deploy or activation needs Eric's D7. A review SIGN approves neither.
- **Decisions:**
  - **Asking:** put decisions in a decision dialog (AskUserQuestion). The herdr manager (`projects-30`, pane `w5:p1`) answers under Eric's 10/03 delegation ("go with your instinct on the bts calls"), and relays to Eric whatever is beyond its scope.
  - **Recording:** record exactly who decided, Eric or the manager. When unsure, ask the manager by SendMessage before writing the provenance.
  - **Mis-attribution has happened twice:** C-07, and the W0 r2 archive note.
- **Data and the box:**
  - an exposure-register row before any outcome-bearing read; no 2026 outcome reads (D3 RESERVE);
  - any 403 or 429 stops, with no rerun without Eric;
  - one box job at a time, through the C1 launcher;
  - $0 of new spend;
  - box-job CPU: 100 CPU-h for C1, with a stop-and-report at 50 (set C2's cap in the plan).
- **Never:**
  - blind-approve a Codex dialog. Decline an escalation request with `esc` and supply author-run evidence instead; the auto-mode classifier denied approving one;
  - touch the miniflux pane (`wA:t3`);
  - run `git checkout <file>` over uncommitted work;
  - use the Claude-in-Chrome extension;
  - copy the healthchecks ping URL anywhere (E107). Redact URLs when you print the crontab: `sed -E 's#https?://[^ "]+#<URL>#g'`.
- **No unapproved production code on main.** The next `main:deploy` ships everything, so revert or park it, as was done for the serving witness and W0. Deploy a specific reviewed SHA: `git push origin <sha>:refs/heads/deploy`.

## 5. Lessons from C1
- **Review churn is the main cost:**
  - **Rounds kept finding new defects,** often in the fix code itself: 4a r2 found S1–S3 in the new witness, plus R2 coverage gaps; W0 needed 3 rounds.
  - **Scope each item tightly in the plan,** and get the design reviewed before building.
  - **Reuse reviewed components** instead of inventing (e.g. rank 3's PA accounting for 4a's R1).
  - **Stop at round 2** rather than force it.
- **Validate the hardest assumption early:**
  - **4a's frozen completeness rule** ("a partial game with PA and zero hits is unknown") needed full PA accounting. Two rounds were spent on a hits-only witness.
  - **An accounting that could make every game unknown** has to be checked against real feed structure. That read needs an exposure row first; rank 3's accounting already has that evidence.
- **Test the producer-to-reader path, not hand-written fixtures.**
  - **The example:** 4a's `projected=False` fixture masked that production writes null.
  - **The rule:** when a reader depends on a production field, build the fixture by running the real producer.
- **Pin a mutation-test ledger before each review:**
  - Run every mutant with `python -B` and a fresh `PYTHONPYCACHEPREFIX`, restore by hash, and clear `__pycache__` before full suites.
  - **The stale-.pyc trap:** a same-size mutant swap can leave a stale `.pyc` that contaminates a later run. It recurred three times (see memory `reference_cross_project_gotchas.md`).
  - **Survivors mean vacuous tests,** e.g. a message regex matching the wrong error; tighten the test, don't argue.
- **Codex sandbox failures:**
  - Inside Codex's sandbox, socket binds (`test_sd_notify`, `test_web_saver`) and the C1 guard's pre-exec hook fail.
  - Give the reviewer your own unsandboxed run instead, labelled as author evidence.
- **Codex reviewers search the memory registry despite the prompt's ban.**
  - **How often:** at their first tool call, three times so far, the last in the deploy review r1.
  - **Handling:** they disclose it. Treat the review as not strictly blind and say so; this is flagged to Eric.
- **Whole-range review before any deploy:**
  - A fresh Codex session, Part 1 blind.
  - It found what the component reviews couldn't: the cron blank-line strip, and a missing certificate re-run at the candidate.
- **Capture box-side changes before making them:**
  - **The crontab:** use a fake `crontab` first on PATH, where `-l` cats the backup and `-` writes to a temp file, to get the exact proposal. Diff it, install, then `cmp` it.
  - **The workflow log** lacks remote stdout (`capture_stdout: false`). Verify HEAD, the units and `ActiveEnterTimestamp` on the box.
  - **The test gate** now takes about 10 minutes.

## 6. Codex helper conventions (herdr round-trip; the skill `herdr` plus memory `reference_herdr_setup.md`)
1. **The review checkout:** `git worktree add --detach ~/projects/<name> <commit>`, then `UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model`. Prompts and reports go under `.codex-review/<task>/` (gitignored).
2. **The tab:** `herdr tab create --workspace "$HERDR_WORKSPACE_ID" --cwd <checkout> --label codex --no-focus`.
3. **Start Codex:** `herdr agent start <name> --kind codex --pane <id> --timeout 120000 -- --no-daemon`, then read the pane.
   - **The update chooser:** Eric's standing ruling is "Update now". Send `herdr agent send-keys <id> 1`, wait for the installer's shell prompt, and start again.
4. **Bind it:**
   - send `agent prompt <id> "Reply with exactly the word READY. Use no tools." --wait --until working --until blocked --timeout 20000`;
   - check the visible `• READY`, its `Worked for` footer and an empty composer;
   - check that `agent get` shows a non-null `agent_session`;
   - record the pane, `terminal_id`, name and session. Compare all of them before every prompt or key.
5. **The prompt file:**
   - **It names:** the pinned commit, the scope, a "read this whole file first; no memory/registry search" opening, and the constraints (no `data/`, `.env` or config reads; no network, ssh or `gh`; no tracked edits; nothing outside the checkout; no escalation).
   - **It asks for:** a fixed report path and a marker `<TAG>-DONE-<first 8 hex of the report's sha256>`, then `DONE`.
   - **For research code under the admission gate,** also ask for an R-only receipt file with `## Verdict` / `**SIGN.**` / `Reviewed-commit: <40 hex>`.
6. **Send it:** `agent prompt <id> "Read <abs path> and follow its instructions exactly." --wait --until working --until blocked --timeout 20000`.
7. **Watch it** with the Monitor tool on `herdr-watch-agent <id> --terminal <tid> --session <sid> --name <name> --marker '^\s*•?\s*<TAG>-DONE-[0-9a-f]{8}\s*$'`.
   - **Write the marker regex exactly,** including `DONE-`. A wrong regex was caught once.
   - **Any watcher line,** SETTLED included, means: re-check the tuple, read the full pane, and verify the report's sha prefix equals the marker.
8. **Archive** each report verbatim to `docs/audit/<date>-<item>-codex-r<N>.md` and commit it.
9. **Wind down:**
   - `agent prompt <id> "/quit"`;
   - verify the agent is gone from `agent list` and a shell prompt shows;
   - `herdr tab close <tab>`, only for your own tab and never the workspace's last tab;
   - `git worktree remove`.

**The mutant runner** (it was kept in the old session's scratchpad, which a reboot clears, so here it is in full):

```python
"""Each mutant must make its named tests fail; the file is restored by hash after each.
spec JSON: [{"id", "file", "old", "new", "tests": [...]}]; "old" must occur exactly once."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
REPO = Path("/Users/eric/projects/bts")
for m in json.loads(Path(sys.argv[1]).read_text()):
    f = REPO / m["file"]; orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
    if s.count(m["old"]) != 1:
        print(m["id"], "ANCHOR x", s.count(m["old"])); continue
    f.write_text(s.replace(m["old"], m["new"]))
    try:
        env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York",
               "PYTHONPYCACHEPREFIX": tempfile.mkdtemp(), "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(["uv", "run", "python", "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *m["tests"]],
                           cwd=REPO, env=env, capture_output=True, text=True)
        print(m["id"], "RED" if r.returncode else "SURVIVED", (r.stdout.strip().splitlines() or ["?"])[-1])
    finally:
        f.write_bytes(orig); assert hashlib.sha256(f.read_bytes()).hexdigest() == h
```

**The fast suite:** `UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York uv run pytest -m "not slow" --ignore=tests/simulate --ignore=tests/model --ignore=tests/experiment --ignore=tests/validate -q -p no:cacheprovider`. It takes about 7 minutes now; clear `__pycache__` first.
