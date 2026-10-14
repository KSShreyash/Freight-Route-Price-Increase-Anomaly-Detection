#!/usr/bin/env bash
# Three untouched runs must produce identical output.
set -euo pipefail
cd "$(dirname "$0")/.."

for i in 1 2 3; do
  python -m ftanomaly.cli --output "output/run_${i}.csv" --token-log "reports/tokens_run_${i}.json"
done

echo
echo "--- diff run 1 vs run 2 ---"
diff output/run_1.csv output/run_2.csv && echo "identical"
echo "--- diff run 1 vs run 3 ---"
diff output/run_1.csv output/run_3.csv && echo "identical"

echo
echo "sha256:"
sha256sum output/run_1.csv output/run_2.csv output/run_3.csv

echo
echo "--- api calls per run (runs 2 and 3 must be 0) ---"
for i in 1 2 3; do
  printf 'run %d: ' "$i"
  python -c "import json,sys; d=json.load(open('reports/tokens_run_${i}.json')); print('llm_calls =', d['llm_calls'], ' cache_hits =', d['cache_hits'])"
done
