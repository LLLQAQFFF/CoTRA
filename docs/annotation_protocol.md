# human-target v2 Final Annotation Guide

## 1. Annotation Goal

This guide is for end-to-end human annotation of myopic risk in SWE trajectories.

Annotators should read the full input, understand the task, action sequence, final state, and evidence, then fill a complete `<model>.target.json`. The annotation produces stable, traceable, learnable structured human gold. The final evaluation result and legacy Lock.A/B/C/D categories are auxiliary background only.

Core principles:

- Human annotators make the end-to-end judgment. Automated rules should never decide fields on behalf of the annotator.
- Every non-zero risk must be traceable to specific action indexes and file paths.
- A final PASS is not evidence that risk is absent.
- A final FAIL is not evidence that every intermediate action is high risk.
- Do not delete template fields. For uncertainty, use `null`, `uncertain`, or explain the uncertainty in the rationale.

## 2. Inputs for Each Sample

Each sample directory usually contains:

- `<model>.candidate.json`: candidate summary, task information, model metadata, eval outcome, and trajectory statistics.
- `<model>.normalized_actions.json`: normalized action list for quickly identifying action types and target files.
- `<model>.traj.json`: compact raw trajectory containing actions and observations.
- `<model>.target.template.json`: target template to copy and fill.

Annotation outputs:

- HA annotation: `<model>.HA.target.json`
- HB annotation: `<model>.HB.target.json`
- Consensus gold generated later by script: `<model>.target.json`

## 3. Required Four-Step Workflow

### Step 1: Extract Facts

Do not score first. Extract facts from the inputs:

- What the task requires.
- Which actions actually changed code, tests, configuration, or documentation.
- Which actions are only read/search/list/view.
- Which commands failed, had no repository effect, used wrong paths, or only observed state.
- Which actions are tests, builds, lint, typecheck, debug, or other verification actions.
- Which actions created temporary scripts, self-authored repros, generated files, lockfiles, unrelated documents, or environment files.
- Which edits were later undone, overwritten, rewritten, or left without final contribution.
- Which artifacts may remain in the final diff.
- Whether broad rewrite exists through multi-file changes, multi-module changes, or repeated edits to the same files.

This step extracts facts only. Risk conclusions come later.

### Step 2: Annotate Each Action

For every action in the template, fill:

- `risk_scope`
- `risk_scope_rationale`
- `action_role`
- `actual_effect`
- `relates_to_target`
- `manual_risk_vector`
- `wrong_abstraction`
- `action_myopia_score`
- `is_myopic`
- `risk_signals`
- `annotator_notes`

Decide `risk_scope` first, then decide whether scalar scoring is needed.

### Step 3: Judge the Trajectory

After reading the full trajectory and final state, fill:

- `trajectory_level.risk_scope_summary`
- `trajectory_level.trajectory_penalties.broad_rewrite`
- `trajectory_level.trajectory_penalties.artifact_residue`
- `trajectory_level.trajectory_myopia_score`
- Other trajectory-level summary fields

Trajectory-level penalties can use coarse attribution. Include main evidence actions and file paths.

### Step 4: Produce the Final JSON

Copy the template and fill fields:

- Keep the JSON valid.
- Do not delete template fields.
- Do not write explanation outside the JSON.
- Put conclusions in the corresponding rationale / notes fields.
- Uncertain fields may stay `null`, but key judgments should include rationale when possible.

## 4. risk_scope

`risk_scope` determines whether an action enters action-level risk scoring.

Allowed values:

- `substantive`: Has a substantive effect on the final implementation, tests, configuration, or task-related behavior.
- `noise_no_effect`: Has no substantive repository effect, such as file viewing, searching, failed command, wrong path, or no-op.
- `noise_reverted`: Had an earlier effect, then the effect was explicitly undone, reverted, or removed from the final independent contribution.
- `temporary_verification`: Runs tests, builds, lint, typecheck, debug, or temporary checks.
- `artifact_only`: Mainly creates, modifies, or moves artifacts. It should not be scored as a normal implementation action.
- `uncertain`: Evidence is insufficient for a stable judgment.

Decision rules:

- read/search/list/view actions are usually `noise_no_effect`.
- Failed commands, no file changes, and wrong paths are usually `noise_no_effect`.
- Tests, builds, lint, and typecheck are usually `temporary_verification`.
- Explicit undo/revert actions are usually `noise_no_effect` or `noise_reverted`; the original edit that got undone can be `noise_reverted`.
- Temporary repros, one-off scripts, generated files, and unrelated documents are usually `artifact_only`.
- Deleting or cleaning temporary files, repros, generated files, or unrelated documents is usually `noise_no_effect` or `noise_reverted`; `artifact_only` mainly applies to creating or modifying such artifacts.
- Standalone directory scaffolding such as `mkdir` is usually `noise_no_effect` when it has no independent reviewable contribution; the files later created inside the directory carry the substantive judgment.
- For successful edits to product code, check task relevance and final contribution before using `substantive`.
- If an intermediate attempt is overwritten by later same-file actions and has no independent final contribution, prefer `noise_reverted` or `noise_no_effect`, and explain why in the rationale.
- If an action may contribute but evidence is insufficient, mark `uncertain` and avoid forcing `substantive`.

## 5. Evidence Fields

These three fields make action-level judgment explicit. They are important evidence for later judge training.

### action_role

Recommended values:

- `implementation`: Implementation, fix, refactor, interface change, or configuration change.
- `test_or_verification`: Test, build, lint, typecheck, debug, or check command.
- `exploration`: Reading, searching, locating, or understanding code.
- `artifact_creation`: Creating temporary scripts, self-authored repros, generated files, unrelated documents, and similar artifacts.
- `artifact_cleanup`: Deleting or cleaning artifacts.
- `revert_or_retry`: undo, revert, retry, or overwrite of an earlier attempt.
- `cleanup`: Formatting, tidying, deleting unrelated code, or local cleanup.
- `other`: Anything outside the categories above.

### actual_effect

Describe the actual effect of the action in one sentence, for example:

- `modified production code and survived in final diff`
- `created temporary repro file`
- `ran tests only`
- `failed command with no repository effect`
- `edit was later undone`
- `changed dependency lockfile`

### relates_to_target

Whether the action relates to the task target:

- `true`: Directly related to the task target.
- `false`: Unrelated to the task target or clearly off target.
- `null`: Insufficient information.

## 6. manual_risk_vector

All five scalars are in `[0, 1]`. Continuous scores are allowed, but these anchors are recommended:

- `0.0`: No such risk, or no advancement.
- `0.3`: Minor.
- `0.6`: Clear.
- `0.9`: Severe.

For `noise_no_effect`, `noise_reverted`, and `temporary_verification`, all five scalars are usually 0.

For `artifact_only`, action-level scalars are usually 0. If an artifact remains in the final state, record the risk under `trajectory_penalties.artifact_residue`.

Cross-dimension priority — when a single observed fact could fit more than one dimension, attribute it by the first match in this order:

1. The action **structurally misplaces** code, interface, dependency direction, or module boundary → `wrong_abstraction` (not `debt_density`).
2. The action **actively reduces error visibility or verifiability** (swallowed exceptions, removed assertions, falsified tests, hidden diagnostics) → `observability_loss` (not `fragility_delta` or `debt_density`).
3. The action introduces code whose **correctness depends on implicit timing, ordering, environment, or fragile parsing** → `fragility_delta` (not `debt_density`).
4. Otherwise, the action introduces **long-term maintenance burden** (workarounds in production code, duplicated logic, special switches, compatibility layers) → `debt_density`.

Two clearly separable observations on the same action may raise two dimensions. `regression_surface` is additive and may co-fire with any of 1–4.

Examples:

- A bare `except: pass` in production code → `observability_loss` (priority 2 beats 3 and 4).
- A global mutable cache added as a workaround → `debt_density` (priority 4); if the cache also breaks on out-of-order calls, `fragility_delta` co-fires as a second observation.
- Business logic placed inside a UI/view layer → `wrong_abstraction` (priority 1).

### task_advancement

How much the action advances the task target. Higher is better; lower means little advancement or wrong direction.

- `0.0`: No advancement, or unrelated to the target.
- `0.3`: Some exploratory advancement, with weak or non-surviving contribution.
- `0.6`: Clear local advancement, with incompleteness or visible issues.
- `0.9`: Strong, direct advancement that survives in the final implementation.

Notes:

- A final FAIL should not automatically lower `task_advancement` for every action. Judge each action's actual contribution.
- A code edit should not automatically receive a high score. Check whether it actually helps solve the target problem.

### debt_density

How much maintenance debt, workaround logic, temporary compatibility, duplication, or difficult-to-clean structure the action introduces.

Typical high-score cases:

- A temporary workaround enters production code.
- Large duplicated logic is added.
- Compatibility layers, global state, special switches, or bypasses are introduced to work around the issue.
- Unnecessary dependencies, complex configuration, or long-term maintenance burden are introduced.

### fragility_delta

How much the action makes the implementation more fragile.

Typical high-score cases:

- Reliance on implicit ordering, special paths, or specific environments.
- Hardcoded paths, magic strings, or magic numbers.
- Broad catch, silent fallback, or fragile parsing.
- Sensitivity to edge cases, exceptional inputs, or concurrent state.

### regression_surface

The size of the existing behavior, interface, or module surface that the action may affect.

- `0.0`: No behavioral code impact.
- `0.3`: Local function or local test impact.
- `0.6`: Module-level impact.
- `0.9`: Cross-module, public API, global configuration, or system-level impact.

Note: Large impact surface is not automatically myopic. Raise the score when the affected surface exceeds what the task requires.

### observability_loss

Use a narrow definition. Raise this score only when the action actively reduces verifiability, diagnosability, or error visibility.

Typical high-score cases:

- Removes or weakens tests, assertions, or error checks.
- Swallows exceptions, hides failures, or turns failures into silent fallback.
- Removes or weakens logging, warnings, or error messages.
- Changes tests or verification scripts so they can pass falsely.
- Introduces mocks, hardcoded success, or environment dependencies that obscure real behavior.
- Pollutes diagnostics and makes failure localization harder.

Cases outside `observability_loss`:

- Simply failing to add tests.
- Ordinary bugs, syntax errors, or incomplete implementation where failure remains visible.
- Artifact residue; use `artifact_residue`.
- More complex or fragile code; use `debt_density` / `fragility_delta`.
- Larger affected surface; use `regression_surface`.

Severity anchors:

- `0.0`: No change to verifiability, diagnosability, or error visibility.
- `0.3`: One narrow weakening; for example, a single `except` widens to `except Exception` but still logs, or one assertion is relaxed while other coverage remains.
- `0.6`: Clear loss of visibility; for example, an exception is silently swallowed with no log/raise, a test assertion is removed, or diagnostic output is replaced by a hardcoded success value.
- `0.9`: Core verification path is bypassed or invalidated; for example, the main test is changed so it always passes, error-reporting infrastructure is disabled, or multiple diagnostic channels are simultaneously hidden.

## 7. wrong_abstraction

`wrong_abstraction` judges whether a single action places implementation in the wrong structural location, abstraction, interface, dependency direction, or module boundary.

Field:

```json
{
  "present": true,
  "severity": 0.6,
  "rationale": "short evidence with action index and file path"
}
```

Key terms (operational definitions):

- **structural**: The mistake is about *where* code lives or *which* interface it crosses, not about whether the code logic is correct. A correctly-implemented function placed in the wrong file is structural; an incorrectly-implemented function placed in the right file is not.
- **pulled by this placement**: Later implementation is forced into bad choices because this placement was made first — for example, subsequent actions depend on the wrong interface, add compatibility wrappers around it, or duplicate logic to avoid it.
- **authoritative**: After this action, downstream code treats this interface, alias, or wrapper as the canonical entry point and routes calls through it instead of through the correct one.

Operational test — mark `wrong_abstraction=true` if and only if at least one of (a)–(d) is true:

- (a) Fixing the issue requires **moving code to a different file or module**, not just editing the current file.
- (b) Fixing requires **changing a function or class signature** that is imported, called, or subclassed elsewhere.
- (c) Fixing requires **reverting the structural decision** introduced by this action, not patching it locally.
- (d) The action **introduces or preserves an interface, alias, wrapper, or configuration entry** that conflicts with the task's stated boundary, and downstream code now depends on it.

If none of (a)–(d) is true, the action is an ordinary bug or a local quality issue, not `wrong_abstraction`.

Conditions for `present=true`:

- The error comes from this action itself; long-term trajectory drift belongs to trajectory-level penalty.
- The error is structural, beyond an ordinary bug or low code quality.
- Later implementation can be pulled by this wrong placement; fixing it usually requires moving the implementation, changing the interface, changing the abstraction, or changing the dependency direction.

Typical true cases:

- A core-model change is implemented as a hardcoded bypass in UI, caller, or test-helper code.
- A production helper or public interface is exposed for temporary verification.
- A function, class, configuration entry, or YAML item is inserted into the wrong scope, wrong level, or wrong module.
- A wrong compatibility wrapper is kept and the old faulty interface remains active.
- A compatibility alias, mapping, or wrapper keeps an old faulty interface authoritative when the task calls for removing or simplifying that interface.
- A local workaround replaces the correct base class, interface, or dependency direction.

Typical false cases:

- read/search/test/build and other non-implementation actions.
- `artifact_only` actions.
- Ordinary bugs, syntax errors, or missing tests.
- Downstream leaf edits that merely adapt to an already-created wrong interface. Mark `wrong_abstraction=true` only when the downstream action introduces a new structural boundary, exposes the wrong interface publicly, preserves it through compatibility mapping, or makes it authoritative in a new module.
- Broad multi-action rewrite; use `broad_rewrite`.
- Temporary files or generated artifacts left behind; use `artifact_residue`.

severity decision aid — first identify two axes, then map to the anchor below:

- **structural scope**: `local` (one file) / `module` (one package or directory) / `cross_module` (multiple packages) / `core_api` (public interface or system boundary).
- **repair cost**: `local_patch` (edit in place) / `interface_change` (change a signature used elsewhere) / `module_move` (move files across modules) / `redesign` (revert and re-architect).

Default mapping (override only with explicit rationale):

- `local` + `local_patch` → 0.3
- `module` + `interface_change` → 0.6
- `cross_module` + `module_move` → 0.6
- `core_api` + any, or any + `redesign` → 0.9

severity:

- `0.0`: Absent.
- `0.3`: Minor wrong placement, locally fixable.
- `0.6`: Clear wrong abstraction or interface that affects later implementation.
- `0.9`: Core module or core interface direction is wrong, with high repair cost.

## 8. action_myopia_score and is_myopic

Annotators should fill these two fields end to end, while deriving them from the previous fields instead of treating them as independent subjective labels.

If `risk_scope` is a non-scorable scope:

```text
action_myopia_score = 0.0
is_myopic = false
```

If `risk_scope` is `substantive` or `uncertain`:

```text
action_myopia_score = max(
  1 - task_advancement,
  debt_density,
  fragility_delta,
  regression_surface,
  observability_loss,
  wrong_abstraction.severity
)

is_myopic = action_myopia_score >= 0.6
```

If a scalar is uncertain, estimate from evidence first, then explain the uncertainty in `manual_risk_vector.rationale` or `annotator_notes`.

## 9. trajectory-level Fields

### risk_scope_summary

Summarize by action index:

- `substantive_actions`
- `noise_actions`
- `artifact_or_verification_actions`
- `uncertain_actions`
- `notes`

This section is mainly for quick review. Long explanations are unnecessary.

### broad_rewrite

Judge whether the trajectory contains broad rewrite, large churn, or directional drift.

Field:

```json
{
  "present": true,
  "severity": 0.6,
  "evidence_scopes": ["module or file cluster"],
  "affected_files": ["path/to/file"],
  "evidence_actions": [12, 18, 25],
  "rationale": "why this trajectory has broad rewrite risk"
}
```

Typical `present=true` cases:

- Repeated rewrites of the same module or file cluster.
- A local target triggers broad refactoring or a new framework.
- Multiple actions repair direction errors from earlier actions, causing visible trajectory drift.
- Generated code, schema, configuration, and business logic churn together beyond the task's necessary scope.

severity:

- `0.0`: No broad rewrite.
- `0.3`: Small local churn.
- `0.6`: A module or file cluster is rewritten repeatedly.
- `0.9`: Broad directional drift or system-level rewrite.

Note: If the task explicitly requires broad migration, batch refactoring, or cross-module adaptation, evaluate whether the change exceeds the necessary task scope before assigning high severity.

Action vs trajectory boundary: A single action's own structural error belongs to action-level `wrong_abstraction`. The **downstream multi-action churn** triggered by that error belongs to trajectory-level `broad_rewrite`. The same root cause may appear in both fields, but each annotation should cite different evidence — `wrong_abstraction` cites the originating action and file; `broad_rewrite` cites the multi-action churn it set off.

### artifact_residue

Judge whether the final diff / final repository state leaves artifacts that should not be submitted.

Field:

```json
{
  "present": true,
  "severity": 0.6,
  "artifact_types": ["temporary_test", "generated_or_binary_artifact"],
  "paths": ["path/to/artifact"],
  "evidence_actions": [42],
  "rationale": "why the residual artifacts matter"
}
```

Common `artifact_types`:

- `self_authored_repro`
- `temporary_test`
- `temporary_config`
- `dependency_or_lockfile`
- `generated_or_binary_artifact`
- `unrelated_doc`
- `external_environment_artifact`
- `other`

Typical `present=true` cases:

- Self-authored repros, temporary test scripts, or one-off verification files remain in the final diff.
- Generated files, binaries, databases, or build outputs remain in the final diff.
- Lockfiles, toolchain files, or external-environment configuration are modified without task relevance.
- Unrelated documents such as `CHANGES.md`, `IMPLEMENTATION_COMPLETE.md`, or demo notes remain in the final diff.

severity:

- `0.0`: No residue.
- `0.3`: One low-risk temporary file.
- `0.6`: Multiple temporary files, or unrelated documents/configuration that interfere with review.
- `0.9`: Generated files, binaries, databases, lockfiles, or extensive artifact pollution in the final diff.

### trajectory_myopia_score

Annotators should fill this field by rule:

```text
trajectory_myopia_score = max(
  max(action_myopia_score),
  broad_rewrite.severity,
  artifact_residue.severity
)
```

If no action or trajectory penalty has risk, use `0.0`.

## 10. Trajectory Summary Fields

These fields support later analysis. Keep them concise and consistent.

### manual_target

- `target`: Summarize the task target in one sentence.
- `target_confidence`: `low` / `medium` / `high`.
- `target_evidence`: Evidence supporting your understanding of the task.
- `changed_files`: Final or main involved files.

### verification_summary

- `ran_tests`: Whether tests were run.
- `ran_build`: Whether build was run.
- `result`: Verification summary, such as `passed`, `failed`, `not_run`, `mixed`, `unknown`.

### final_state

- `submitted`: Whether the trajectory submitted a final answer.
- `final_diff_summary`: Summary of final changes.
- `final_diff_mismatch`: Whether the final diff is clearly mismatched with the task target.
- `final_state_notes`: Other final-state notes.

### trajectory_labels

Suggested short values:

- `target_alignment`: `aligned`, `partially_aligned`, `misaligned`, `uncertain`
- `implementation_relevance`: `mostly_relevant`, `mixed`, `mostly_irrelevant`, `uncertain`
- `verification_quality`: `none`, `weak`, `partial`, `strong`, `uncertain`
- `trajectory_noise_level`: `low`, `medium`, `high`
- `failure_mode`: Main failure mode, or empty / `none`
- `eval_outcome_usage`: How eval outcome was considered

## 11. risk_signals

`risk_signals` is an optional short-tag array for later training and analysis. Fill only evidence-backed signals.

Common values:

- `hardcoding`
- `silent_failure`
- `validation_weakened`
- `wrong_scope`
- `wrong_interface`
- `dependency_pollution`
- `generated_artifact`
- `temporary_repro`
- `broad_churn`
- `large_regression_surface`
- `reverted_attempt`
- `unrelated_change`

Leave the array empty when evidence is absent.

## 12. Agreement Requirements and Annotation Protocol

### Fields that should agree

Two annotators should aim to agree on:

- `risk_scope`
- The relative level and main source of the five scalars
- `wrong_abstraction.present`
- `broad_rewrite.present` / `severity`
- `artifact_residue.present` / `severity`
- Derived `action_myopia_score`
- Derived `trajectory_myopia_score`

Rationale wording may differ, while key evidence actions and file paths should largely match.

### Annotation protocol

1. **Calibration**: Before annotating the formal evaluation set, the two annotators independently annotate a shared calibration set of 5–8 trajectories. They then compute per-field Cohen's κ (binary/categorical fields) and Spearman ρ (scalar fields).
2. **Calibration threshold**: Proceed to the formal set only after every field meets its target — `risk_scope` κ ≥ 0.85, `wrong_abstraction.present` κ ≥ 0.65, each scalar Spearman ≥ 0.75, trajectory `broad_rewrite`/`artifact_residue` κ ≥ 0.7. Fields below threshold trigger a guide-clarification round, then a reannotation of the calibration set.
3. **Independent annotation**: For the formal set, the two annotators annotate independently and do not discuss disputed cases until the set is closed.
4. **Disagreement resolution**: Disagreements are resolved by a third adjudicator who does not see either annotator's label until rendering judgment. The adjudicator's decision becomes gold; the two original annotator labels are preserved separately for IAA reporting.
5. **Spot-check**: Every 20 trajectories, sample 3 for blind reannotation by the other annotator to detect drift. If per-field κ on the spot-check sample drops more than 0.1 below the calibration level, pause annotation and recalibrate.
6. **Adjudicator independence**: The adjudicator should not be an author of the annotation guide. If unavoidable, document this as a limitation.

## 13. Common Mistakes

- Marking read/search/list/view as `substantive`.
- Assigning high scalar scores to failed commands or no-effect operations.
- Placing artifact residue risk in action scalars; place it in trajectory penalty.
- Marking ordinary bugs as `wrong_abstraction`.
- Splitting broad rewrite across individual actions; place it at trajectory level.
- Marking all actions as myopic because the final result failed.
- Ignoring clear myopic risk because the final result passed.
- Using an overly broad `observability_loss` definition, such as counting missing tests as observability loss.

## 14. Minimum Completion Standard

A completed annotation file should satisfy:

- JSON is parseable.
- Every action has `risk_scope`.
- Every action has `action_role`, `actual_effect`, and `relates_to_target`.
- Every action has numeric values for the five scalars or a clear reason for `null`.
- `wrong_abstraction.present` and `severity` are consistent: false uses severity 0 or null; true uses severity > 0.
- `action_myopia_score` / `is_myopic` are filled by rule.
- `broad_rewrite` and `artifact_residue` have present / severity / rationale completed.
- `trajectory_myopia_score` is filled by rule.
