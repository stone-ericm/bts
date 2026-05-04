#!/bin/bash
# v2.5_teardown.sh — Delete all 4 harness-v2.5 Vultr instances.
#
# Mirrors task13_teardown.sh. Reads /tmp/v2.5/instances.tsv
# (cell_label TAB instance_id TAB ip) and deletes each instance in parallel.
#
# Usage: bash scripts/v2_5_teardown.sh

set -euo pipefail

if [ ! -f /tmp/v2.5/instances.tsv ]; then
  echo "ERROR: /tmp/v2.5/instances.tsv not found. Nothing to tear down." >&2
  exit 1
fi

VULTR_TOKEN=$(security find-generic-password -a "claude-cli" -s "vultr-api-token" -w)
if [ -z "$VULTR_TOKEN" ]; then
  echo "ERROR: vultr-api-token not found in Keychain" >&2
  exit 1
fi

> /tmp/v2.5/teardown_results.txt
while IFS=$'\t' read -r CELL INST IP; do
  (
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
      "https://api.vultr.com/v2/instances/$INST" \
      -H "Authorization: Bearer $VULTR_TOKEN")
    printf '%s\t%s\t%s\n' "$CELL" "$INST" "$HTTP_CODE" >> /tmp/v2.5/teardown_results.txt
  ) &
done < /tmp/v2.5/instances.tsv
wait

echo "===== teardown results ====="
sort /tmp/v2.5/teardown_results.txt
echo ""
SUCCEEDED=$(awk -F'\t' '$3 == "204"' /tmp/v2.5/teardown_results.txt | wc -l | tr -d ' ')
echo "Succeeded (HTTP 204): $SUCCEEDED / 4"

# Verify: confirm 0 v2.5 instances remain
echo ""
echo "===== remaining harness-v2.5 instances ====="
curl -s "https://api.vultr.com/v2/instances?per_page=100" \
  -H "Authorization: Bearer $VULTR_TOKEN" | python3 -c "
import json, sys
data = json.load(sys.stdin)
v25 = [i for i in data.get('instances', []) if 'harness-v2.5' in i.get('label', '')]
print(f'Remaining harness-v2.5 instances: {len(v25)}')
for i in v25[:10]:
    print(f'  {i[\"label\"]} {i[\"id\"]}')"
