"""Formal audit output derived from encoder evidence and v2 judgments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from encoder_judge.evidence import ActionEvidence


REPOSITORY_EVIDENCE_VERSION = "repository-evidence-v1"
ENGINEERING_CONSEQUENCE_VERSION = "engineering-consequence-v1"
ENGINEERING_SCORECARD_VERSION = "engineering-scorecard-v1"

SCORABLE_SCOPES = {"substantive", "uncertain"}
NON_SCORABLE_SCOPES = {
    "noise_no_effect",
    "noise_reverted",
    "temporary_verification",
    "artifact_only",
}
RISK_DIMS = [
    "task_advancement",
    "debt_density",
    "fragility_delta",
    "regression_surface",
    "observability_loss",
]


def attach_repository_evidence(actions: list[dict[str, Any]], evidence: list[ActionEvidence]) -> None:
    """Attach public, reviewable repository evidence to each action."""
    for action, ev in zip(actions, evidence, strict=False):
        action["repository_evidence"] = build_repository_evidence(action, ev)


def attach_engineering_consequences(actions: list[dict[str, Any]]) -> None:
    """Attach deterministic engineering-consequence summaries to actions."""
    for action in actions:
        action["engineering_consequence"] = build_engineering_consequence(action)


def build_trajectory_engineering_scorecard(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    penalties: dict[str, Any],
    trajectory_myopia_score: float | None,
) -> dict[str, Any]:
    """Aggregate action consequences into a trajectory-level engineering scorecard."""
    scope_counts = Counter(str(a.get("risk_scope") or "unknown") for a in actions)
    consequences = [a.get("engineering_consequence") or {} for a in actions]
    repo_blocks = [a.get("repository_evidence") or {} for a in actions]
    risk_factor_counts = Counter(
        factor
        for consequence in consequences
        for factor in consequence.get("risk_factors") or []
    )
    affected_files = sorted({
        path
        for consequence in consequences
        if float(consequence.get("maintenance_risk_score") or 0.0) > 0.0
        for path in consequence.get("affected_files") or []
    })
    residual_artifacts = list(dict.fromkeys(
        str(path)
        for path in ((penalties.get("artifact_residue") or {}).get("paths") or [])
    ))
    top_risk_actions = sorted(
        (
            {
                "action_index": action.get("action_index"),
                "risk_scope": action.get("risk_scope"),
                "maintenance_risk_score": round(float((action.get("engineering_consequence") or {}).get("maintenance_risk_score") or 0.0), 3),
                "risk_factors": (action.get("engineering_consequence") or {}).get("risk_factors") or [],
                "affected_files": (action.get("engineering_consequence") or {}).get("affected_files") or [],
                "primary_reason": (action.get("engineering_consequence") or {}).get("primary_reason") or "",
            }
            for action in actions
        ),
        key=lambda item: item["maintenance_risk_score"],
        reverse=True,
    )[:10]
    top_risk_actions = [item for item in top_risk_actions if item["maintenance_risk_score"] > 0.0]

    primary_failure_modes = [
        factor
        for factor, _count in risk_factor_counts.most_common()
    ]
    broad = penalties.get("broad_rewrite") or {}
    artifact = penalties.get("artifact_residue") or {}

    return {
        "scorecard_version": ENGINEERING_SCORECARD_VERSION,
        "scope_counts": dict(scope_counts),
        "top_risk_actions": top_risk_actions,
        "repository_evidence_summary": {
            "actions_with_strong_evidence": sum(1 for block in repo_blocks if block.get("evidence_strength") == "strong"),
            "actions_with_weak_or_missing_evidence": sum(
                1 for block in repo_blocks if block.get("evidence_strength") in {"weak", "missing"}
            ),
            "replay_available_actions": sum(1 for block in repo_blocks if (block.get("replay") or {}).get("available")),
            "static_available_actions": sum(1 for block in repo_blocks if (block.get("static") or {}).get("available")),
        },
        "maintenance_risk_summary": {
            "trajectory_myopia_score": round(float(trajectory_myopia_score or 0.0), 3),
            "high_risk_action_count": sum(
                1 for consequence in consequences if float(consequence.get("maintenance_risk_score") or 0.0) >= 0.6
            ),
            "risk_factor_counts": dict(risk_factor_counts),
            "broad_rewrite": broad,
            "artifact_residue": artifact,
        },
        "engineering_consequence_summary": {
            "primary_failure_modes": primary_failure_modes,
            "affected_files": affected_files,
            "residual_artifacts": residual_artifacts,
            "notes": _scorecard_notes(scope_counts, risk_factor_counts, broad, artifact),
        },
    }


def build_repository_evidence(action: dict[str, Any], ev: ActionEvidence) -> dict[str, Any]:
    final = ev.final_diff_contribution or {}
    replay = ev.replay_evidence or {}
    static = ev.static_evidence or {}
    state = ev.state_evidence or {}
    refs = _evidence_refs(action, ev)
    return {
        "evidence_version": REPOSITORY_EVIDENCE_VERSION,
        "target_files": list(ev.target_files),
        "contribution": {
            "patch_survival": ev.patch_survival.status,
            "semantic_survival": str(final.get("semantic_survival") or "unknown"),
            "final_diff_present": bool(final.get("present")),
            "final_diff_independent": bool(final.get("independent", final.get("present", False))),
            "later_same_file_actions": list(ev.patch_survival.later_same_file_actions),
        },
        "replay": {
            "available": bool(replay),
            "applied": bool(replay.get("applied", not replay.get("apply_failed", False))) if replay else False,
            "apply_failed": bool(replay.get("apply_failed")),
            "failure_reason": str(replay.get("apply_failure_reason") or ""),
        },
        "static": {
            "available": bool(static.get("available")),
            "syntax_error": bool(static.get("syntax_error")),
            "wrong_scope_insertion": bool(static.get("wrong_scope_insertion")),
            "fragility_signals": _fragility_signals(static),
            "observability_signals": _observability_signals(static),
            "artifact_signals": _artifact_signals(ev, static),
        },
        "state": {
            "same_file_chain": dict(state.get("same_file_chain") or {"previous": [], "later": []}),
            "validation_after": list(state.get("validation_after") or []),
            "cumulative_churn": dict(state.get("cumulative_churn") or {}),
        },
        "evidence_refs": refs,
        "evidence_strength": _evidence_strength(refs, action),
    }


def build_engineering_consequence(action: dict[str, Any]) -> dict[str, Any]:
    vector = action.get("manual_risk_vector") or {}
    repo = action.get("repository_evidence") or {}
    risk_factors = _risk_factors(action)
    score = round(float(action.get("action_myopia_score") or 0.0), 3)
    return {
        "consequence_version": ENGINEERING_CONSEQUENCE_VERSION,
        "scope": action.get("risk_scope") or "unknown",
        "is_scorable": action.get("risk_scope") in SCORABLE_SCOPES,
        "risk_factors": risk_factors,
        "maintenance_risk_score": score,
        "future_impact": _future_impact(action, repo, vector),
        "affected_files": list((repo.get("target_files") or [])),
        "primary_reason": _primary_reason(action, risk_factors),
        "evidence_refs": list(repo.get("evidence_refs") or []),
    }


def formal_output_meta() -> dict[str, str]:
    return {
        "repository_evidence_version": REPOSITORY_EVIDENCE_VERSION,
        "engineering_consequence_version": ENGINEERING_CONSEQUENCE_VERSION,
        "scorecard_version": ENGINEERING_SCORECARD_VERSION,
    }


def _evidence_refs(action: dict[str, Any], ev: ActionEvidence) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    final = ev.final_diff_contribution or {}
    replay = ev.replay_evidence or {}
    static = ev.static_evidence or {}
    state = ev.state_evidence or {}

    if final.get("present") or final.get("semantic_survival"):
        refs.append(_ref(
            "final_diff",
            _short(final.get("evidence") or f"Final contribution: {final.get('semantic_survival') or 'present'}."),
            ev.target_files,
            [ev.action_index],
            0.9 if final.get("present") else 0.65,
        ))
    if replay.get("evidence") or replay.get("apply_failed"):
        refs.append(_ref(
            "replay",
            _short(replay.get("evidence") or replay.get("apply_failure_reason") or "Replay evidence available."),
            ev.target_files,
            [ev.action_index],
            0.85 if not replay.get("apply_failed") else 0.75,
        ))
    if static.get("syntax_error") or static.get("wrong_scope_insertion"):
        refs.append(_ref(
            "static",
            "Static analysis found syntax or structural-scope evidence.",
            ev.target_files,
            [ev.action_index],
            0.85,
        ))
    static_signals = _fragility_signals(static) + _observability_signals(static) + _artifact_signals(ev, static)
    if static_signals:
        refs.append(_ref(
            "static",
            "Static analysis signals: " + ", ".join(static_signals[:8]),
            ev.target_files,
            [ev.action_index],
            0.75,
        ))
    same_file = state.get("same_file_chain") or {}
    if same_file.get("previous") or same_file.get("later") or state.get("validation_after"):
        refs.append(_ref(
            "state",
            "Trajectory state evidence includes same-file chain or later validation.",
            ev.target_files,
            [ev.action_index, *list(same_file.get("previous") or []), *list(same_file.get("later") or [])],
            0.65,
        ))
    artifacts = ev.artifact_signals or {}
    artifact_paths = list(artifacts.get("introduced_artifact_paths") or []) + list(artifacts.get("residual_artifact_paths") or [])
    if artifact_paths:
        refs.append(_ref(
            "artifact",
            "Artifact-like paths touched or left in final state.",
            artifact_paths,
            [ev.action_index],
            0.8,
        ))
    rule = action.get("_encoder_rule") or {}
    rationale = action.get("risk_scope_rationale") or rule.get("rule_scope_candidate")
    if rationale and _requires_evidence(action):
        refs.append(_ref(
            "rule",
            _short(str(rationale)),
            ev.target_files,
            [ev.action_index],
            0.55,
        ))
    semantic = action.get("_encoder_semantic") or {}
    if semantic and not semantic.get("fallback"):
        refs.append(_ref(
            "llm",
            _short(semantic.get("rationale") or "Narrow semantic judgment was applied."),
            ev.target_files,
            [ev.action_index],
            float(semantic.get("confidence") or 0.6),
        ))
    review = action.get("_encoder_trajectory_review") or {}
    if review.get("scope_review_applied"):
        refs.append(_ref(
            "llm",
            _short(review.get("scope_review_rationale") or "Trajectory review adjusted action scope."),
            ev.target_files,
            [ev.action_index],
            float(review.get("scope_review_confidence") or 0.75),
        ))
    wa_review = action.get("_encoder_wrong_abstraction_review") or {}
    if wa_review.get("applied"):
        refs.append(_ref(
            "llm",
            _short(wa_review.get("rationale") or "Wrong-abstraction review adjusted this action."),
            ev.target_files,
            [ev.action_index],
            float(wa_review.get("confidence") or 0.7),
        ))
    return _dedupe_refs(refs)


def _requires_evidence(action: dict[str, Any]) -> bool:
    vector = action.get("manual_risk_vector") or {}
    wa = action.get("wrong_abstraction") or {}
    return (
        action.get("risk_scope") != "noise_no_effect"
        or any(float(vector.get(dim) or 0.0) > 0.0 for dim in RISK_DIMS)
        or bool(wa.get("present"))
        or bool(action.get("is_myopic"))
    )


def _risk_factors(action: dict[str, Any]) -> list[str]:
    vector = action.get("manual_risk_vector") or {}
    factors: list[str] = []
    if action.get("risk_scope") in SCORABLE_SCOPES and float(vector.get("task_advancement") or 0.0) < 0.6:
        factors.append("low_task_advancement")
    for dim in ["debt_density", "fragility_delta", "regression_surface", "observability_loss"]:
        if float(vector.get(dim) or 0.0) > 0.0:
            factors.append(dim)
    if bool((action.get("wrong_abstraction") or {}).get("present")):
        factors.append("wrong_abstraction")
    return factors


def _future_impact(action: dict[str, Any], repo: dict[str, Any], vector: dict[str, Any]) -> str:
    score = float(action.get("action_myopia_score") or 0.0)
    scope = action.get("risk_scope")
    files = repo.get("target_files") or []
    regression = float(vector.get("regression_surface") or 0.0)
    churn = ((repo.get("state") or {}).get("cumulative_churn") or {})
    files_touched = int(churn.get("files_touched") or len(files))

    if scope not in SCORABLE_SCOPES:
        return "none"
    if regression >= 0.75 or files_touched >= 12:
        return "repository"
    if regression >= 0.5 or len(files) >= 4 or score >= 0.75:
        return "cross_module"
    if regression >= 0.25 or len(files) >= 2:
        return "module"
    if files:
        return "local"
    return "unknown"


def _primary_reason(action: dict[str, Any], risk_factors: list[str]) -> str:
    if risk_factors:
        return "Primary engineering risks: " + ", ".join(risk_factors) + "."
    if action.get("risk_scope") in NON_SCORABLE_SCOPES:
        return str(action.get("risk_scope_rationale") or "Non-scorable action; no action-level maintenance risk.")
    return str(action.get("risk_scope_rationale") or "No nonzero engineering risk factor was derived.")


def _evidence_strength(refs: list[dict[str, Any]], action: dict[str, Any]) -> str:
    if not refs:
        return "missing" if _requires_evidence(action) else "weak"
    max_conf = max(float(ref.get("confidence") or 0.0) for ref in refs)
    types = {ref.get("type") for ref in refs}
    if max_conf >= 0.85 and ({"final_diff", "replay", "static", "artifact"} & types):
        return "strong"
    if max_conf >= 0.65:
        return "moderate"
    return "weak"


def _fragility_signals(static: dict[str, Any]) -> list[str]:
    signals = []
    if int(static.get("silent_except_added") or 0):
        signals.append("silent_except_added")
    if int(static.get("broad_except_added") or 0):
        signals.append("broad_except_added")
    if int(static.get("hardcoded_path_added") or 0):
        signals.append("hardcoded_path_added")
    return signals


def _observability_signals(static: dict[str, Any]) -> list[str]:
    signals = []
    if int(static.get("silent_except_added") or 0):
        signals.append("silent_except_added")
    if bool(static.get("validation_weakened")):
        signals.append("validation_weakened")
    return signals


def _artifact_signals(ev: ActionEvidence, static: dict[str, Any]) -> list[str]:
    artifact_scan = static.get("artifact_scan") or {}
    signals = []
    if ev.artifact_signals.get("introduced_artifact_paths"):
        signals.append("introduced_artifact_path")
    if ev.artifact_signals.get("residual_artifact_paths"):
        signals.append("residual_artifact_path")
    if artifact_scan.get("introduced"):
        signals.append("static_artifact_introduced")
    if artifact_scan.get("still_present"):
        signals.append("static_artifact_still_present")
    if artifact_scan.get("deps_introduced"):
        signals.append("dependency_artifact_introduced")
    if artifact_scan.get("deps_still_present"):
        signals.append("dependency_artifact_still_present")
    return signals


def _scorecard_notes(
    scope_counts: Counter[str],
    risk_factor_counts: Counter[str],
    broad: dict[str, Any],
    artifact: dict[str, Any],
) -> str:
    parts = [
        f"Scopes: {dict(scope_counts)}.",
        f"Risk factors: {dict(risk_factor_counts)}.",
    ]
    if broad.get("present"):
        parts.append("Broad rewrite penalty is present.")
    if artifact.get("present"):
        parts.append("Artifact residue penalty is present.")
    return " ".join(parts)


def _ref(kind: str, summary: str, paths: list[str], actions: list[int], confidence: float) -> dict[str, Any]:
    return {
        "type": kind,
        "summary": summary,
        "paths": list(dict.fromkeys(str(path) for path in paths if path)),
        "actions": list(dict.fromkeys(int(action) for action in actions if action is not None and int(action) >= 0)),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
    }


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for ref in refs:
        key = (ref.get("type"), ref.get("summary"), tuple(ref.get("paths") or []), tuple(ref.get("actions") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _short(value: str, limit: int = 240) -> str:
    value = " ".join(str(value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 14].rstrip() + " ...[truncated]"
