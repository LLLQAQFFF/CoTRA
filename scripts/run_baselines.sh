#!/usr/bin/env bash
# Reproduce the deterministic / evidence baselines (B1, B3) under the shared
# output schema. B2 (ProcCtrlBench) lives in a separate package; see
# baselines/README.md. B4 (end-to-end judge) is produced by run_e2e.sh.
#
# B1 = Evidence + deterministic rules only: no calibration set (so the learned
#      scope gate is never trained -> rule scope) and --no-llm.
# B3 = Evidence + LLM, no fixed scope gate: an LLM judge runs but no calibration
#      set is supplied and scope is not frozen, so the model decides scope too.
#
# Env vars:
#   DATA_ROOT   dir containing the three splits (default: ./data/CoTRA-Bench)
#   REPO_CACHE  replay snapshot cache (default: ./.repo_cache)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src

DATA_ROOT="${DATA_ROOT:-./data/CoTRA-Bench}"
REPO_CACHE="${REPO_CACHE:-./.repo_cache}"
SPLITS=(set1_v2 set2_v2_batch03 set3_v2)

for s in "${SPLITS[@]}"; do
  # B1: Evidence + rules only (no calibrator, no LLM).
  python3 -m encoder_judge.cli prelabel-set "$DATA_ROOT/$s" \
    --no-llm \
    --repo-cache "$REPO_CACHE" --output-tag rules-only-nollm --skip-existing &

  # B3: Evidence + LLM with no fixed scope gate (no calibration set, not frozen).
  python3 -m encoder_judge.cli prelabel-set "$DATA_ROOT/$s" \
    --judge-model deepseek-v4-pro \
    --repo-cache "$REPO_CACHE" --output-tag abl-nogate --skip-existing &
done
wait
echo "BASELINE RUNS DONE"
