#!/bin/zsh
# One-off: the pre-fix opener (must fail), then the three dates on the pushed commit (must pass), each from a reset state.
cd ~/bts-runs/coldstart || exit 1
export UV_CACHE_DIR=/tmp/uv-cache TZ=America/New_York
reset_state() {
  rm -rf repo/data/raw
  rm -f repo/data/processed/pa_2026.parquet repo/data/processed/pa_2027.parquet repo/data/models/probable_pitcher_lookup.json
  find repo/data/processed -name ".refreshed_*" -delete
  find repo/data/models -name "blend_*.pkl" -delete
}
echo "### pre-fix commit e3edbf5, opener only $(date '+%F %T')"
git -C repo checkout -q --detach e3edbf58 || exit 1
git -C repo log --oneline -1
reset_state
(cd repo && .venv/bin/python ../rehearse.py 2026-03-25 --require-cold --expect rows > ../prefix-e3edbf5-rehearse-2026-03-25.log 2>&1; echo "pre-fix opener rc=$?")
echo "### pushed commit $(date '+%F %T')"
git -C repo checkout -q --detach 07780d3 || exit 1
reset_state
zsh ./run_coldstart.sh 2026-03-25 2026-03-26 2027-03-24
echo "run_coldstart rc=$?"
echo "RERUN DONE $(date '+%F %T')"
