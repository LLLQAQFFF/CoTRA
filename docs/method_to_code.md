# Method-to-code map

CoTRA's four modules (paper Section "Method") map to the `encoder_judge`
package as follows. The internal package keeps its development name
`encoder_judge`; it *is* the CoTRA pipeline.

| Paper module | What it does | Code |
|---|---|---|
| **Evidence extraction** (§ Evidence Extraction) | Replays each edit on repository snapshots; builds the action evidence record `e_i = ⟨kind, obs, persist, final, static⟩` | `src/encoder_judge/evidence.py` (`build_evidence_table`, `_build_replay_static_evidence`, `infer_patch_survival`, `infer_final_diff_contribution`); replay in `src/repo_env/replayer.py`; static signals in `src/static_analysis/` |
| **Scope adjudication** (§ Scope Adjudication) | Action validity assessment: a shallow decision tree assigns each action a contribution scope `s_i ∈ S` (6 classes); rule prelabel is one input feature | `src/encoder_judge/calibration.py` (`ScopeCalibrator`, `features_for`); rule prelabel + scope space in `src/encoder_judge/rules.py` (`classify_scope`) |
| **Semantic scoring** (§ Semantic Scoring) | Action risk assessment: one LLM call per scorable action returns the risk-dimension levels `r_i = Review(q, e_i, s_i)`; scope is a fixed premise | `src/encoder_judge/semantic.py` (`score_action_semantics`); error-abstraction review in `src/encoder_judge/wrong_abstraction.py`; trajectory-level review in `src/encoder_judge/review.py` |
| **Score derivation** (§ Score Derivation) | Maps discrete levels to numbers via `ℓ/η/ρ`, takes the max for `y_i`, and `Y = max{max y_i, σ_broad, σ_artifact}` | `src/llm_judge/derive.py`; trajectory waste baselines `σ_broad`, `σ_artifact` in `src/encoder_judge/rules.py` (`synthesize_trajectory`) |

## Key design points (as implemented)

- **Scope gate is frozen before the LLM.** With `--freeze-scope`, the
  calibrator's scope is fixed and semantic scoring cannot revise it
  (`apply_scope_overrides = not freeze_scope`).
- **Decision tree.** `DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
  random_state=0)`, no class weighting, features via `DictVectorizer`; trained
  on human consensus `risk_scope`.
- **Cross-fitting.** Trajectory-level, three folds (`set1_v2`,
  `set2_v2_batch03`, `set3_v2`), leave-two-train / predict-held-out, so every
  reported scope prediction is out-of-fold. See `scripts/run_crossfit.sh`.
- **Semantic model.** DeepSeek-V4-pro, temperature 0; only `substantive` and
  `uncertain` actions are sent to the model.

## Baselines

| Baseline | Code path / flag |
|---|---|
| **B1 Evidence+rules** | `encoder_judge.cli ... --no-llm` (no calibration set) → tag `rules-only-nollm` |
| **B2 ProcCtrlBench** | separate package, see `baselines/README.md` |
| **B3 Evidence+LLM** (no fixed scope gate) | `encoder_judge.cli ... --judge-model deepseek-v4-pro` (no calibration set, not frozen) → tag `abl-nogate` |
| **B4 End-to-end judge** | `encoder_judge.cli ... --baseline e2e-llm` (`src/encoder_judge/e2e_baseline.py`) → tag `e2e-judge` |
