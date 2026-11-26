#!/usr/bin/env bash
# check_audit_progress.sh — query the bts-dashboard /api/audit-progress endpoint
# and pretty-print per-box progress + overall percentage.
#
# Uses env vars for configurability; defaults match the current Vultr n=100 run.
#
#   BTS_DASHBOARD_URL   (default: http://bts-hetzner:3003)
#   AUDIT_PROVIDER      (default: vultr)
#   AUDIT_DIR           (default: audit_ext_n100_v4)
#   AUDIT_SEEDS_FILE    (default: scripts/audit_seeds_extension_n100.txt)
#
# Usage:
#   scripts/check_audit_progress.sh                 # current Vultr run
#   AUDIT_PROVIDER=hetzner AUDIT_DIR=audit_full_48seed_v2 \
#     AUDIT_SEEDS_FILE=scripts/audit_seeds_default48.txt \
#     scripts/check_audit_progress.sh              # retrospective Hetzner run
set -euo pipefail

URL="${BTS_DASHBOARD_URL:-http://bts-hetzner:3003}"
PROVIDER="${AUDIT_PROVIDER:-vultr}"
DIR="${AUDIT_DIR:-audit_ext_n100_v4}"
SEEDS="${AUDIT_SEEDS_FILE:-scripts/audit_seeds_extension_n100.txt}"

endpoint="${URL}/api/audit-progress?provider=${PROVIDER}&dir=${DIR}&seeds_file=${SEEDS}"

# Fetch; curl -f → non-zero exit on HTTP >=400 so we can surface errors loudly.
payload="$(curl -sSf --max-time 60 "${endpoint}")" || {
    echo "ERROR: request failed: ${endpoint}" >&2
    exit 1
}

if ! command -v jq >/dev/null 2>&1; then
    # Graceful degradation when jq isn't installed.
    echo "${payload}"
    exit 0
fi

# Header + audit_attach block rendered as plain text (no tab alignment needed).
jq -r '
    "audit: \(.audit_dir)   scanned_at: \(.scanned_at)",
    "",
    (if .audit_attach then
        (if (.audit_attach.procs | length) > 0 then
            "audit_attach procs:",
            (.audit_attach.procs[] | "  pid=\(.pid)  etime=\(.etime)  \(.cmd[0:100])")
         else
            "audit_attach procs: NONE RUNNING (driver exited?)"
         end),
         ""
     else empty end)
' <<<"${payload}"

# Per-box table: rendered with tab-aligned columns.
jq -r '
    (["box","state","seeds","last_event"] | @tsv),
    (.boxes[] | [
        .name,
        .state,
        (if .expected_seeds != null
         then "\(.completed_seeds // 0)/\(.expected_seeds)"
         else "\(.completed_seeds // 0)/?" end),
        (.last_seed_event // "" | .[0:80])
    ] | @tsv)
' <<<"${payload}" | column -t -s $'\t'

# Overall footer.
jq -r '
    "",
    "overall: \(.overall.completed)/\(.overall.expected // "?") seeds, \(.overall.boxes_done)/\(.overall.boxes_total) boxes done\(if .overall.pct_seeds != null then " (\(.overall.pct_seeds)%)" else "" end)"
' <<<"${payload}"
