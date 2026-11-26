#!/bin/bash
# Task 13: tear down all 17 Vultr boxes in parallel.
set -e

if [ ! -f /tmp/task13/instances.tsv ]; then
  echo "ERROR: /tmp/task13/instances.tsv not found" >&2
  exit 1
fi

VULTR_TOKEN=$(security find-generic-password -a "claude-cli" -s "vultr-api-token" -w)

> /tmp/task13/teardown_results.txt
while IFS=$'\t' read -r seed inst; do
  (
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE \
      "https://api.vultr.com/v2/instances/$inst" \
      -H "Authorization: Bearer $VULTR_TOKEN")
    echo -e "$seed\t$inst\t$HTTP_CODE" >> /tmp/task13/teardown_results.txt
  ) &
done < /tmp/task13/instances.tsv
wait

echo "===== teardown results ====="
sort -n /tmp/task13/teardown_results.txt
echo ""
SUCCEEDED=$(awk -F'\t' '$3 == "204"' /tmp/task13/teardown_results.txt | wc -l)
echo "Succeeded (HTTP 204): $SUCCEEDED / 17"

# Verify: confirm 0 instances remain
echo ""
echo "===== remaining task13 instances ====="
curl -s "https://api.vultr.com/v2/instances?per_page=100" \
  -H "Authorization: Bearer $VULTR_TOKEN" | python3 -c "
import json, sys
data = json.load(sys.stdin)
t13 = [i for i in data.get('instances', []) if 'task13' in i.get('label','')]
print(f'Remaining task13 instances: {len(t13)}')
for i in t13[:5]:
    print(f'  {i[\"label\"]} {i[\"id\"]}')"
