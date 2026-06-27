#!/usr/bin/env bash
# Cross-fitted main run (CoTRA). For each of the three splits, the scope
# calibrator is trained on the other two, so every evaluation trajectory is
# held out from its own scope gate (leakage-free, frozen scope). Produces the
# full LLM run and the matching zero-token (no-LLM) variant.
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

other () {  # other <split> -> the two training splits
  case "$1" in
    set1_v2)          echo "set2_v2_batch03 set3_v2";;
    set2_v2_batch03)  echo "set1_v2 set3_v2";;
    set3_v2)          echo "set1_v2 set2_v2_batch03";;
  esac
}

for s in "${SPLITS[@]}"; do
  read -r t1 t2 <<< "$(other "$s")"
  # Full CoTRA: evidence + learned scope gate (frozen) + LLM semantic scoring
  python3 -m encoder_judge.cli prelabel-set "$DATA_ROOT/$s" \
    --judge-model deepseek-v4-pro --freeze-scope \
    --calibration-set "$DATA_ROOT/$t1" --calibration-set "$DATA_ROOT/$t2" \
    --repo-cache "$REPO_CACHE" --output-tag crossfit-semantic-frozen --skip-existing &
  # Ablation: same pipeline without LLM semantic scoring (zero-token)
  python3 -m encoder_judge.cli prelabel-set "$DATA_ROOT/$s" \
    --no-llm \
    --calibration-set "$DATA_ROOT/$t1" --calibration-set "$DATA_ROOT/$t2" \
    --repo-cache "$REPO_CACHE" --output-tag crossfit-nollm --skip-existing &
done
wait
echo "CROSSFIT RUNS DONE"
