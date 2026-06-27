# CoTRA-Bench data format

The benchmark is **not** bundled in this repository (see the top-level README,
"Data"). This document describes the layout the scripts expect so you can point
`DATA_ROOT` at your copy.

## Layout

```
<DATA_ROOT>/                         # e.g. data/CoTRA-Bench
  set1_v2/
    <instance_id>/
      <model>.candidate.json         # task spec q, eval outcome, metadata
      <model>.normalized_actions.json
      <model>.traj.json              # compact raw trajectory (actions + observations)
      <model>.target.template.json   # annotation template (inputs only)
      <model>.target.json            # consensus human gold labels
      <model>.HA.target.json         # annotator A (double-annotated subset)
      <model>.HB.target.json         # annotator B (double-annotated subset)
  set2_v2_batch03/ ...
  set3_v2/ ...
```

The three splits `set1_v2`, `set2_v2_batch03`, `set3_v2` are the cross-fitting
folds. The benchmark contains 100 trajectories and 4,303 action labels across
four agent families (GPT, Claude, Gemini, GLM).

## Gold label schema (`<model>.target.json`)

- `action_level[]`: one record per action, keyed by `action_id` /
  `action_index`, with:
  - `risk_scope` — one of `substantive`, `uncertain`, `noise_no_effect`,
    `noise_reverted`, `temporary_verification`, `artifact_only`;
  - `manual_risk_vector` — five scalars in `[0,1]` (`task_advancement`,
    `debt_density`, `fragility_delta`, `regression_surface`,
    `observability_loss`), anchors `{0, 0.3, 0.6, 0.9}`;
  - `wrong_abstraction` — `{present, severity, rationale}`;
  - derived: `action_myopia_score`, `is_myopic` (computed from the levels by
    the score-derivation formulas, **not** hand-entered).
- `trajectory_level.trajectory_penalties`:
  - `broad_rewrite` and `artifact_residue`, each `{present, severity, evidence}`;
  - `trajectory_myopia_score` (derived).

The full human annotation protocol is in
[`annotation_protocol.md`](annotation_protocol.md).

## Prediction files

Each method writes predictions mirroring the data layout under
`outputs/encoder_judge/<tag>/<split>/<instance>/<model>.<tag>.encoder_pre_label.json`,
with the same `action_level` / `trajectory_level` schema plus a `_run_meta`
block (token usage, cost, flags). `scripts/eval_predictions.py` aligns
predictions to gold by `action_id`; `scripts/reproduce_tables.py` aggregates
the headline numbers.
