#!/usr/bin/env bash
# B4: end-to-end LLM judge. Each trajectory is judged in a single call; long
# trajectories are compressed to fit the model context window before judging.
# Uses a separate judge config (config/llm_config.e2e.example.py -> llm_config.e2e.py).
#
# Env vars:
#   DATA_ROOT     dir containing the three splits (default: ./data/CoTRA-Bench)
#   E2E_PARALLEL  number of parallel templates (default: 6)
#   LLM_CONFIG_EXTRA  e2e judge config (default: llm_config.e2e.py)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export LLM_CONFIG_EXTRA="${LLM_CONFIG_EXTRA:-llm_config.e2e.py}"

DATA_ROOT="${DATA_ROOT:-./data/CoTRA-Bench}"
OUT_ROOT="outputs/encoder_judge/e2e-judge"
TAG="e2e-judge"
PAR="${E2E_PARALLEL:-6}"
SPLITS=(set2_v2_batch03 set3_v2 set1_v2)

for set_name in "${SPLITS[@]}"; do
  for tpl in "$DATA_ROOT/$set_name"/*/*.target.template.json; do
    model=$(basename "$tpl" .target.template.json)
    inst=$(basename "$(dirname "$tpl")")
    out="$OUT_ROOT/$set_name/$inst/$model.$TAG.encoder_pre_label.json"
    [ -s "$out" ] && continue
    printf '%s\t%s\n' "$tpl" "$OUT_ROOT/$set_name"
  done
done | xargs -P "$PAR" -n 1 -d '\n' bash -c '
  IFS=$'\''\t'\'' read -r tpl outdir <<< "$0"
  python3 -m encoder_judge.cli prelabel "$tpl" --baseline e2e-llm \
    --output-tag '"$TAG"' --output-dir "$outdir" \
    || echo "FAILED $tpl"
'
echo "E2E DONE"
