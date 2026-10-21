The read-only sandbox rejected the write. **`codex-out.md` was not created.** The review follows.

Reviewed the supplied observations, the local checkout verified at `619abda76272bf91bcfbfb717de51d4c42cb747b`, relevant project notes, installed CLI help and vendor documentation. Live cloud resources and secret values were not inspected.

1. **BLOCKER — The backup destination and independent recovery access are not reconciled.**

   Appendix A is empty. At the stated commit, [backup.py](/Users/eric/projects/bts/src/bts/data/backup.py:84) defaults to R2 unless `BTS_RESTIC_REPOSITORY` overrides it. The supplied `.env` inventory contains neither that override nor the required R2 variables. Configuration could be injected elsewhere; this does not invalidate the reported successful backups. It does leave the Storage Box claim unexplained. `INCIDENT.md` instead documents R2 and a Mac Keychain item named `r2-bts-restic-password`.

   The existing restore drill proves access **from the original server** to whichever repository it used. It tests only ops and parses selected JSON files; it does not verify archive or independent recovery from the Mac.

   **Close BEFORE delete:** identify the actual repository endpoint, configuration source and authentication. Preserve them outside the server. From the Mac, using independently stored credentials, run:

   ```sh
   restic snapshots --json
   restic ls 978d5e19
   restic ls db083dd5
   restic restore 978d5e19 --target <empty-ops-directory> --verify
   restic restore db083dd5 --target <empty-archive-directory> --verify
   ```

   Inspect actual restored paths: `CLAUDE.md` and `INCIDENT.md` disagree about whether paths retain the repository prefix.

   **Closing evidence:** recorded backend/full snapshot IDs, an independent credential source, both successful verified restores and readable archive content. Correct the Storage Box claim if the actual backend is R2.

2. **BLOCKER — The deletion evidence omits a recovery inventory for snapshot-only files and external machine configuration.**

   Verified sets are:

   - ops: `data/picks`, `data/health_state`.
   - archive: `data/leaderboard`, `data/hetzner_results`, `data/external`.
   - Both exclude `*.lock`.

   Other listed data trees, experiment checkouts and local changes, scripts, logs, `.env` and SSH material depend on the snapshot unless another backup is demonstrated. Also inventory `~/.bts-orchestrator.toml`, its season-over backup and `~/.bts-leaderboard-cookies.json`. If present, `data/validation/` is excluded; `CLAUDE.md:56` identifies accepted ledger artifacts there. Models/parquets may have separate R2 artifact copies—verify rather than declaring them lost.

   The snapshot preserves root-disk files, packages and the venv. It does not preserve attached Volume contents, RAM, external account settings or a reusable cloud provisioning specification. Provider Backups, if enabled, disappear with the server. [Snapshot scope](https://docs.hetzner.com/cloud/servers/backups-snapshots/overview/), [Backup lifecycle](https://docs.hetzner.com/cloud/servers/backups-snapshots/faq/).

   **Close BEFORE delete:** save these outputs outside the server:

   ```sh
   hcloud server describe 126463369 -o json
   hcloud primary-ip describe 126036156 -o json
   hcloud image describe 446156433 -o json
   ```

   Export applicable firewall rules/attachments, Networks/private addresses, labels, placement group, backup settings and original launch user-data. Identify attached Volumes and their separate recovery plan.

   Create an encrypted configuration bundle containing the actual crontab, installed units/drop-ins and enablement links, orchestrator configuration, `.env`, root/bts SSH configuration, fstab, network configuration, `/etc/cloud` and relevant `/var/lib/cloud` state. Verify it from the Mac. For each checkout, inspect `git status --short --untracked-files=all` and `git log --branches --not --remotes --oneline`.

   **Closing evidence:** readable off-server bundle/manifest, exported cloud specification, Volume disposition and explicit acceptance of remaining snapshot-only data. “Available, 8.91 GB” establishes neither file completeness nor bootability. These checks can use an offline Rescue mount; normal boot restarts jobs.

3. **SHOULD-FIX — The Healthchecks pause may already have been undone.**

   Cron remained active after the four user units stopped. Project notes place the owner's pause at 01:08 ET, while the box stayed up until approximately 01:40. Successful five-minute pings could resume monitoring; the scheduler heartbeat script can also send failure signals after the scheduler stops. Paused checks normally resume on incoming pings unless configured otherwise. [Paused-check behavior](https://healthchecks.io/docs/configuring_checks/).

   **Close now:** with the server confirmed off, inspect both exact checks and their ping histories. Pause them again if necessary and select “Ignore the ping, stay in the paused state” for the offseason. Do not send test pings.

   **Closing evidence:** both currently show `paused`, with no subsequent ping. If sticky pause is selected, spring requires explicit resumption.

4. **SHOULD-FIX — First boot runs old jobs before the proposed validation and deployment.**

   Stop-not-disable preserves enablement, not stopped state. Linger should survive on disk, allowing user services/timers to start before login; cron also resumes. Stopping a timer does not stop an already-triggered service.

   At `619abda`, the resolver timer has `Persistent=true`, permitting an immediate catch-up execution after months offline; capture has `Persistent=false`. Repository schedules also differ from the simplified inventory: capture polls every 15 minutes during 05–22 and resolve runs four times daily. Inspect installed units. [Persistent timers](https://www.freedesktop.org/software/systemd/man/251/systemd.timer.html), [Linger](https://www.freedesktop.org/software/systemd/man/249/systemd-run.html).

   **Close BEFORE delete:** write a held-first-boot procedure. Create powered off; use Rescue/offline mounting to mask cron and BTS services/timers before normal boot. Keep tailscaled offline during an isolated rehearsal. A late cloud-init `runcmd` stopping services cannot prevent earlier executions.

   Verify during recovery:

   ```sh
   loginctl show-user bts -p Linger
   ```

   As bts:

   ```sh
   systemctl --user cat bts-scheduler.service bts-dashboard.service \
     bts-live-forward-capture.timer bts-live-forward-resolve.timer
   systemctl --user list-timers --all
   crontab -l
   ```

   **Closing evidence:** recorded hold/release procedure and installed-unit inventory; recovery journals show no production execution before release. Include cron and timers in acceptance checks.

5. **SHOULD-FIX — Deploying winter code does not perform the 2027 configuration migration.**

   The actual workflow fetches and hard-resets to the triggering SHA, syncs dependencies and restarts two services. It does not install cron/unit templates or update `~/.bts-orchestrator.toml`.

   The [cron template](/Users/eric/projects/bts/scripts/cron-setup-hetzner.sh:91) still builds `--seasons 2026`. The existing [2027 season-start checklist](/Users/eric/projects/bts/docs/ops/2027-season-start.md:1) requires fresh cookies, account checks, configuration changes, artifact verification and separate activation approval. Snapshot restoration cannot renew authentication or obtain current contest rules.

   **Close BEFORE delete:** link that checklist into the restore procedure and distinguish infrastructure recovery from season activation. In spring, inspect the installed crontab, replace the fixed 2026 build behavior, and explicitly install reviewed cron/unit changes. Verify fresh authentication with:

   ```sh
   uv run bts fetch-contest-streak \
     --expected-username stonehengee --dry-run
   ```

   **Closing evidence:** installed 2027 cron diff, effective orchestrator configuration, fresh account observation, artifact/policy hashes, fixture rehearsal and recorded activation approval.

6. **SHOULD-FIX — “Whatever is in authorized_keys” is not an SSH-key selection; cloud-init changes the recovery assumptions.**

   `--ssh-key` accepts a registered Hetzner public-key name/ID. The bts Mac key and root GitHub deploy key are separate authorizations. For snapshots of official images, Hetzner documents retaining existing authorized keys, injecting selected keys and generating/ mailing a root password when no key is selected. Do not rely on the snapshot's old root password. [Snapshot login behavior](https://docs.hetzner.com/cloud/servers/faq/).

   A new instance ID normally causes per-instance cloud-init modules to run again. Hostname can change; host-key deletion defaults to enabled. Local overrides determine actual behavior. Original user-data must be recovered and reviewed, rather than blindly replayed. [First-boot detection](https://docs.cloud-init.io/en/latest/explanation/first_boot.html), [SSH and hostname modules](https://docs.cloud-init.io/en/23.4/reference/modules.html).

   **Close BEFORE delete:** preserve configuration and key fingerprints:

   ```sh
   ssh-keygen -lf /root/.ssh/authorized_keys
   ssh-keygen -lf /home/bts/.ssh/authorized_keys
   ```

   Select a known operator key for independent root access. During spring recovery check `cloud-init status --wait`, `hostnamectl`, `sshd -T` and resulting authorized keys. Verify the new host fingerprint through the trusted Hetzner console before changing known_hosts.

   Preserve the GitHub production environment's deploy key. The actual workflow uses the **`HETZNER_HOST` secret**, so update that secret if the public address changes.

   **Closing evidence:** independent root access, verified host fingerprint, understood cloud-init results and a subsequently authorized successful root-key deploy/canary.

7. **SHOULD-FIX — Tailscale identity reuse is conditional.**

   Disabled expiry does not prevent administrative deletion/revocation or tailnet-policy changes. Linux state can also be encrypted to a particular TPM, making copied disk state insufficient on another VM. There is no evidence here that this server uses that encryption; check it. [Tailscale state storage](https://tailscale.com/docs/features/secure-node-state-storage).

   **Close BEFORE delete:** record the device identity, address, tags/approval and non-ephemeral status. Inspect `systemctl cat tailscaled`, its version, state location and encryption configuration without exposing state contents. Preserve its registration if reuse is intended. Document public-IPv4/console access and re-enrollment as fallback.

   Do not run two instances carrying the same Tailscale state concurrently. Do not log out/delete the production identity from a rehearsal clone.

   **Closing evidence:** readable/migratable state or an explicit re-enrollment plan. In spring verify `tailscale status`, `tailscale ip -4` and a Mac `tailscale ping bts-hetzner`. Update address-dependent clients and confirm the dashboard binding if re-enrollment changes identity.

8. **SHOULD-FIX — The audit data relay is an actual external dependency.**

   [audit_driver.py](/Users/eric/projects/bts/scripts/audit_driver.py:133) uses bts-hetzner as a read-only rsync relay for `data/processed/` when `--data-relay` is enabled. Without that option, it uploads controller-local data. `scripts/check_audit_progress.sh` queries this box's dashboard. This contradicts an unconditional “nothing else depends on it”; it does not prove either path is currently running.

   **Close BEFORE delete:** inspect active audit controllers on Mac/pi5 for `audit_driver.py` and `--data-relay`. Confirm usable controller-local processed/model inputs, remove the relay option for winter work or configure a replacement through `BTS_RELAY_HOST_SSH`, `BTS_RELAY_HOST_PUBLIC`, `BTS_RELAY_USER` and `BTS_RELAY_DATA_DIR`.

   Search active LaunchAgents, cron/systemd jobs for the hostname, addresses, `--data-relay` and `BTS_DASHBOARD_URL`. Mark the progress dashboard unavailable. Explicitly accept cessation of static/lineup captures, previews, account checks and artifact synchronization; wanted offseason observations cannot be recreated later.

   **Closing evidence:** no live dependent audit run, or an explicitly redirected one, and acceptance of collection gaps. No additional active alarm source was established.

9. **SHOULD-FIX — Make disk, architecture and Primary-IP constraints explicit.**

   The recovery floor is the supplied **320 GB disk size**, not compressed image size or used space. Use a compatible x86 type with at least 320 GB. CPX42 is one choice; snapshot retention does not reserve March capacity or quota. Smaller disks or ARM require a different rebuild.

   **Close BEFORE delete:** save `hcloud server-type describe cpx42 -o json`, a compatible-type fallback and this proposed creation command, adding exported firewall/network settings:

   ```sh
   hcloud server create --name bts-mlb --type cpx42 --location hel1 \
     --image 446156433 --primary-ipv4 126036156 \
     --ssh-key <registered-operator-key> --start-after-create=false
   ```

   This allocates a billed server and requires Eric's approval before execution. Follow finding 4 before normal boot.
    At the supplied snapshot rate, October costs approximately €0.125/month and July €0.069. Together with retained IPv4 at €0.50, those resources total approximately **€0.69/month excluding VAT**. St

    **Closing evidence:** saved pre-delete recovery record and post-delete resource inventory, without project-wide cleanup.

VERDICT: DO NOT DELETE YET — reconcile and independently verify the actual restic repository, export the recovery configuration/inventory, and recheck the two paused monitors after shutdown.

