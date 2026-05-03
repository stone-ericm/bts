#!/bin/bash
# v2.5_provision.sh — Provision 4 Vultr boxes for harness-v2.5 ablation cells.
#
# Each box runs one ablation cell (010, 001, 011, 101). The script:
#   1. Provisions 4 Vultr vhp-12c-24gb-amd boxes in ewr with cloud-init that
#      installs uv and waits at /root/cloud-init-done.
#   2. Polls until all 4 boxes are active with IPs.
#   3. Rsyncs src/, pyproject.toml, uv.lock to each box.
#   4. Rsyncs data/simulation/ parquets to each box.
#   5. Runs `uv sync --extra model` on each box.
#   6. Writes /tmp/v2.5/instances.tsv: cell_label<TAB>instance_id<TAB>ip
#
# Usage: bash scripts/v2_5_provision.sh [--dry-run]
#
# Prereqs:
#   - vultr-api-token in macOS Keychain (service=vultr-api-token, account=claude-cli)
#   - SSH pubkey at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub (registered in Vultr)
#   - data/simulation/ parquets present in the working directory (gitignored)
#
# Conventions mirror task13_teardown.sh / task13_retrieve.sh.

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# ---- Sanity checks ----

VULTR_TOKEN=$(security find-generic-password -a "claude-cli" -s "vultr-api-token" -w)
if [ -z "$VULTR_TOKEN" ]; then
  echo "ERROR: vultr-api-token not found in Keychain" >&2
  exit 1
fi

if [ ! -d "data/simulation" ]; then
  echo "ERROR: data/simulation/ not found (run from worktree root)" >&2
  exit 1
fi

PROFILES_COUNT=$(ls data/simulation/profiles_seed*_season*.parquet 2>/dev/null | wc -l | tr -d ' ')
PA_COUNT=$(ls data/simulation/pa_predictions_seed*_season*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [ "$PROFILES_COUNT" -eq 0 ] || [ "$PA_COUNT" -eq 0 ]; then
  echo "ERROR: data/simulation/ parquets missing (profiles=$PROFILES_COUNT pa=$PA_COUNT)" >&2
  echo "  Expected: profiles_seed*_season*.parquet and pa_predictions_seed*_season*.parquet" >&2
  exit 1
fi
echo "Found $PROFILES_COUNT profile parquets and $PA_COUNT PA prediction parquets in data/simulation/"

SSH_PUBKEY_PATH=""
for candidate in ~/.ssh/id_ed25519.pub ~/.ssh/id_rsa.pub; do
  if [ -f "$candidate" ]; then
    SSH_PUBKEY_PATH="$candidate"
    break
  fi
done
if [ -z "$SSH_PUBKEY_PATH" ]; then
  echo "ERROR: No SSH pubkey found at ~/.ssh/id_ed25519.pub or ~/.ssh/id_rsa.pub" >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] Sanity checks passed. Would provision 4 Vultr boxes."
  echo "  SSH pubkey: $SSH_PUBKEY_PATH"
  echo "  Parquets: profiles=$PROFILES_COUNT pa=$PA_COUNT"
  echo "  instances.tsv would be written to: /tmp/v2.5/instances.tsv"
  exit 0
fi

# ---- Cloud-init payload ----
# Installs uv, creates /root/projects/bts/data/, signals readiness.
# Code + parquets are rsync'd from local after boxes are up.

CLOUD_INIT_B64=$(python3 -c "
import base64
payload = '''#cloud-config
packages: [rsync, git, libgomp1, python3, ca-certificates, curl]
runcmd:
  - curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
  - env HOME=/root sh /tmp/uv-install.sh
  - mkdir -p /root/projects/bts/data/simulation /root/projects/bts/data/validation
  - touch /root/cloud-init-done
'''
print(base64.b64encode(payload.encode()).decode())
")

# ---- Resolve Vultr OS ID and SSH key ID ----

echo "Resolving Vultr OS ID for 'Ubuntu 24.04 LTS x64'..."
OS_ID=$(curl -s "https://api.vultr.com/v2/os?per_page=500" \
  -H "Authorization: Bearer $VULTR_TOKEN" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for o in data.get('os', []):
    if o.get('name') == 'Ubuntu 24.04 LTS x64' and o.get('arch') == 'x64':
        print(o['id'])
        sys.exit(0)
print('NOT_FOUND', file=sys.stderr)
sys.exit(1)
")
echo "  OS ID: $OS_ID"

echo "Resolving SSH key ID..."
SSH_PUBKEY=$(cat "$SSH_PUBKEY_PATH")
SSH_KEY_ID=$(curl -s "https://api.vultr.com/v2/ssh-keys" \
  -H "Authorization: Bearer $VULTR_TOKEN" | python3 -c "
import json, sys
pubkey = '''$SSH_PUBKEY'''.strip()
data = json.load(sys.stdin)
for k in data.get('ssh_keys', []):
    if k.get('ssh_key', '').strip() == pubkey:
        print(k['id'])
        sys.exit(0)
print('NOT_FOUND')
")

if [ "$SSH_KEY_ID" = "NOT_FOUND" ]; then
  echo "  Uploading SSH pubkey to Vultr..."
  SSH_KEY_ID=$(curl -s -X POST "https://api.vultr.com/v2/ssh-keys" \
    -H "Authorization: Bearer $VULTR_TOKEN" \
    -H "Content-Type: application/json" \
    --data "{\"name\": \"bts-v2.5\", \"ssh_key\": $(python3 -c "import json,sys; print(json.dumps(open('$SSH_PUBKEY_PATH').read().strip()))")}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['ssh_key']['id'])")
fi
echo "  SSH key ID: $SSH_KEY_ID"

# ---- Provision 4 boxes ----

CELLS=(010 001 011 101)
mkdir -p /tmp/v2.5
> /tmp/v2.5/instances.tsv

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10"

for CELL in "${CELLS[@]}"; do
  LABEL="harness-v2.5-cell-${CELL}"
  echo "Creating $LABEL..."
  INSTANCE_ID=$(curl -s -X POST "https://api.vultr.com/v2/instances" \
    -H "Authorization: Bearer $VULTR_TOKEN" \
    -H "Content-Type: application/json" \
    --data "{
      \"region\": \"ewr\",
      \"plan\": \"vhp-12c-24gb-amd\",
      \"os_id\": $OS_ID,
      \"label\": \"$LABEL\",
      \"sshkey_id\": [\"$SSH_KEY_ID\"],
      \"user_data\": \"$CLOUD_INIT_B64\",
      \"backups\": \"disabled\",
      \"tag\": \"harness-v2.5\"
    }" | python3 -c "
import json, sys
data = json.load(sys.stdin)
inst = data.get('instance', {})
if not inst.get('id'):
    print(json.dumps(data), file=sys.stderr)
    sys.exit(1)
print(inst['id'])
")
  echo "  $LABEL: instance_id=$INSTANCE_ID"
  printf '%s\t%s\t\n' "$CELL" "$INSTANCE_ID" >> /tmp/v2.5/instances.tsv
done

# ---- Poll until all boxes are active with IPs ----

echo ""
echo "Polling for active status + IP assignment (timeout 5 min)..."
DEADLINE=$(python3 -c "import time; print(int(time.time()) + 300)")

while true; do
  NOW=$(python3 -c "import time; print(int(time.time()))")
  if [ "$NOW" -gt "$DEADLINE" ]; then
    echo "ERROR: Timeout waiting for boxes to become active" >&2
    exit 1
  fi

  ALL_READY=1
  # Rebuild TSV with IPs filled in
  > /tmp/v2.5/instances_new.tsv
  while IFS=$'\t' read -r CELL INST IP; do
    if [ -n "$IP" ]; then
      printf '%s\t%s\t%s\n' "$CELL" "$INST" "$IP" >> /tmp/v2.5/instances_new.tsv
      continue
    fi
    STATUS_JSON=$(curl -s "https://api.vultr.com/v2/instances/$INST" \
      -H "Authorization: Bearer $VULTR_TOKEN")
    BOX_STATUS=$(echo "$STATUS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin).get('instance',{}); print(d.get('status',''))")
    BOX_IP=$(echo "$STATUS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin).get('instance',{}); print(d.get('main_ip',''))")
    if [ "$BOX_STATUS" = "active" ] && [ -n "$BOX_IP" ] && [ "$BOX_IP" != "0.0.0.0" ]; then
      echo "  harness-v2.5-cell-${CELL} active ip=$BOX_IP"
      printf '%s\t%s\t%s\n' "$CELL" "$INST" "$BOX_IP" >> /tmp/v2.5/instances_new.tsv
    else
      printf '%s\t%s\t\n' "$CELL" "$INST" >> /tmp/v2.5/instances_new.tsv
      ALL_READY=0
    fi
  done < /tmp/v2.5/instances.tsv
  mv /tmp/v2.5/instances_new.tsv /tmp/v2.5/instances.tsv

  if [ "$ALL_READY" -eq 1 ]; then
    echo "All 4 boxes active."
    break
  fi
  sleep 10
done

# ---- Wait for cloud-init to finish on each box ----

echo ""
echo "Waiting for cloud-init to complete on all boxes..."
while IFS=$'\t' read -r CELL INST IP; do
  echo "  Waiting for harness-v2.5-cell-${CELL} ($IP)..."
  for i in $(seq 1 60); do
    if ssh $SSH_OPTS "root@$IP" "test -f /root/cloud-init-done" 2>/dev/null; then
      echo "    cloud-init done"
      break
    fi
    if [ "$i" -eq 60 ]; then
      echo "ERROR: cloud-init timeout on harness-v2.5-cell-${CELL} ($IP)" >&2
      exit 1
    fi
    sleep 10
  done
done < /tmp/v2.5/instances.tsv

# ---- Rsync code to each box ----

echo ""
echo "Rsyncing src/, pyproject.toml, uv.lock to each box..."
SSH_CMD="ssh $SSH_OPTS"
while IFS=$'\t' read -r CELL INST IP; do
  echo "  harness-v2.5-cell-${CELL} ($IP): code..."
  rsync -az -e "$SSH_CMD" \
    src/ \
    "root@${IP}:/root/projects/bts/src/" || { echo "ERROR: rsync src failed for cell $CELL" >&2; exit 1; }
  rsync -az -e "$SSH_CMD" \
    pyproject.toml uv.lock \
    "root@${IP}:/root/projects/bts/" || { echo "ERROR: rsync configs failed for cell $CELL" >&2; exit 1; }
done < /tmp/v2.5/instances.tsv

# ---- Rsync data/simulation/ parquets to each box ----

echo ""
echo "Rsyncing data/simulation/ parquets to each box (~239MB)..."
while IFS=$'\t' read -r CELL INST IP; do
  echo "  harness-v2.5-cell-${CELL} ($IP): parquets..."
  rsync -az -e "$SSH_CMD" \
    data/simulation/ \
    "root@${IP}:/root/projects/bts/data/simulation/" || { echo "ERROR: rsync simulation failed for cell $CELL" >&2; exit 1; }
done < /tmp/v2.5/instances.tsv

# ---- uv sync --extra model on each box ----

echo ""
echo "Running 'uv sync --extra model' on each box (parallel)..."
> /tmp/v2.5/uvsync_results.txt
while IFS=$'\t' read -r CELL INST IP; do
  (
    LOG=/tmp/v2.5/uvsync_cell${CELL}.log
    ssh $SSH_OPTS "root@$IP" \
      "cd /root/projects/bts && PATH=/root/.local/bin:\$PATH UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model 2>&1 | tail -5" \
      > "$LOG" 2>&1
    RC=$?
    echo -e "${CELL}\t${IP}\t${RC}" >> /tmp/v2.5/uvsync_results.txt
    echo "  harness-v2.5-cell-${CELL}: uv sync rc=$RC"
  ) &
done < /tmp/v2.5/instances.tsv
wait

FAILED=$(awk -F'\t' '$3 != "0"' /tmp/v2.5/uvsync_results.txt | wc -l | tr -d ' ')
if [ "$FAILED" -gt 0 ]; then
  echo "ERROR: uv sync failed on $FAILED box(es)" >&2
  awk -F'\t' '$3 != "0"' /tmp/v2.5/uvsync_results.txt >&2
  exit 1
fi

# ---- Final summary ----

echo ""
echo "===== v2.5 instances ready ====="
echo "cell_label	instance_id	ip"
cat /tmp/v2.5/instances.tsv
echo ""
echo "All 4 boxes provisioned and ready."
echo "Next: python scripts/v2_5_run_ablations.py"
