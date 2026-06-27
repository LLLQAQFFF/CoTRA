# CoTRA: Contribution-Tracked Risk Auditing

Reference implementation of **CoTRA**, a framework for auditing the trajectories
of LLM coding agents. Instead of an opaque holistic judgment, CoTRA grounds every
score in repository evidence — replayed edits, affected code regions, and whether
each edit remains in the final code — and separates **action validity assessment**
(does the edit really contribute?) from **action risk assessment** (how risky is it?).

CoTRA has four sequential modules:

1. **Evidence extraction** — replay each edit on repository snapshots; build a
   per-action evidence record `e_i = ⟨kind, obs, persist, final, static⟩`.
2. **Scope adjudication** — a shallow decision tree assigns each action a
   contribution scope `s_i` from the evidence; only scorable scopes pass through.
3. **Semantic scoring** — one LLM call per scorable action scores predefined
   risk dimensions, taking the fixed scope as a premise.
4. **Score derivation** — fixed formulas combine the factual and semantic
   outputs into action-level (`y_i`) and trajectory-level (`Y`) short-sightedness.

The companion benchmark is **CoTRA-Bench**: 100 long-horizon SWE-bench Pro
trajectories with 4,303 human-annotated action labels across four agent families
(GPT, Claude, Gemini, GLM).

> The internal Python package is named `encoder_judge`; it is the CoTRA pipeline.
> See [`docs/method_to_code.md`](docs/method_to_code.md) for the full
> paper-section → code mapping.

## Install

```bash
pip install -r requirements.txt        # openai, scikit-learn
# or: pip install -e .
```

Python ≥ 3.11. All scripts expect `PYTHONPATH=src`.

## Configure the judge

CoTRA uses an OpenAI-compatible chat API (the paper uses DeepSeek-V4-pro at
temperature 0). **Never commit your API key.**

```bash
cp config/llm_config.example.py llm_config.local.py   # git-ignored
# edit llm_config.local.py and set LLM_API_KEY
# (or export LLM_API_KEY / LLM_BASE_URL / LLM_JUDGE_MODEL in the environment)
```

`llm_config.py` holds non-secret defaults; `llm_config.local.py` overrides it;
environment variables override both (see `src/llm_judge/config.py`).

## Data

CoTRA-Bench is provided separately and is not bundled here. Point `DATA_ROOT`
at your copy; the expected layout and label schema are documented in
[`docs/data_format.md`](docs/data_format.md), and the human annotation protocol
in [`docs/annotation_protocol.md`](docs/annotation_protocol.md).

```
data/CoTRA-Bench/{set1_v2,set2_v2_batch03,set3_v2}/<instance>/<model>.*.json
```

## Reproduce

```bash
export DATA_ROOT=data/CoTRA-Bench
export REPO_CACHE=.repo_cache          # replay snapshot cache

# Main CoTRA run (cross-fitted, frozen scope) + zero-token ablation
bash scripts/run_crossfit.sh

# Baselines B1 (Evidence+rules) and B3 (Evidence+LLM, no gate)
bash scripts/run_baselines.sh

# Baseline B4 (end-to-end judge); needs llm_config.e2e.py
cp config/llm_config.e2e.example.py llm_config.e2e.py   # edit key
bash scripts/run_e2e.sh

# Headline numbers: scope kappa, bootstrap CIs, per-family, token usage
python scripts/reproduce_tables.py \
    --data-root "$DATA_ROOT" --outputs-root outputs/encoder_judge
```

For a single method/run, `scripts/eval_predictions.py` writes a full
prediction-vs-gold report (scope kappa, exact agreement, per-dimension Spearman,
trajectory metrics, cost summary).

## Main result (CoTRA-Bench, 100 trajectories, 4,303 action labels)

Best quality value in each row is **bold**. M1–M3 are quality; M4 is cost.

| | B1 Evid+rules | B2 ProcCtrlBench | B3 Evid+LLM | B4 E2E judge | **CoTRA** |
|---|---:|---:|---:|---:|---:|
| **M1 Action scope agreement** | | | | | |
| &nbsp;&nbsp;Cohen's κ | 0.330 | 0.431 | 0.338 | 0.508 | **0.599** |
| &nbsp;&nbsp;Exact agreement | 0.459 | 0.613 | 0.468 | 0.644 | **0.744** |
| **M2 Severity ranking (Spearman ρ)** | | | | | |
| &nbsp;&nbsp;Task progress | 0.466 | 0.375 | 0.474 | **0.770** | 0.561 |
| &nbsp;&nbsp;Technical debt | 0.105 | 0.274 | 0.320 | **0.486** | 0.400 |
| &nbsp;&nbsp;Fragility | 0.225 | 0.220 | 0.375 | **0.572** | 0.391 |
| &nbsp;&nbsp;Regression reach | 0.290 | 0.272 | 0.294 | **0.646** | 0.555 |
| &nbsp;&nbsp;Action short-sightedness | 0.247 | 0.314 | 0.279 | 0.264 | **0.496** |
| &nbsp;&nbsp;Trajectory short-sightedness | 0.178 | 0.214 | 0.281 | 0.408 | **0.410** |
| **M3 Defect detection (F1)** | | | | | |
| &nbsp;&nbsp;Broad rewriting | 0.877 | 0.871 | 0.870 | 0.200 | **0.881** |
| &nbsp;&nbsp;Stray artifact | 0.732 | 0.411 | 0.732 | **0.865** | 0.838 |
| &nbsp;&nbsp;Error abstraction | 0.181 | 0.000 | 0.257 | 0.092 | **0.268** |
| **M4 Cost** | | | | | |
| &nbsp;&nbsp;API cost (USD) | 0 | 0 | 1.0 | 19.9 | 2.9 |

CoTRA gives the best action-scope agreement (M1) and the best aggregate
short-sightedness scores (M2, action/trajectory), and is competitive on defect
detection (M3). The end-to-end judge (B4) is stronger on isolated semantic
dimensions but costs ~7× more: CoTRA matches its scope agreement at roughly
**85% lower API cost** (M4), and stays stable on long trajectories where
holistic judging degrades.

## Repository layout

```
src/encoder_judge/     CoTRA pipeline (the four modules + CLI + baselines)
src/repo_env/          edit replay on repository snapshots
src/static_analysis/   lightweight static signals (syntax, lint, artifacts, ...)
src/llm_judge/         API client, cache, cost accounting, score derivation
src/core/              shared action / risk-vector / snapshot types
scripts/               run + evaluation scripts
docs/                  method-to-code map, data format, annotation protocol
baselines/             baseline notes (B2 ProcCtrlBench is external)
config/                example LLM configs (copy to git-ignored local files)
```

## License

Released under the MIT License (see [`LICENSE`](LICENSE)). Update the copyright
holder before release.
