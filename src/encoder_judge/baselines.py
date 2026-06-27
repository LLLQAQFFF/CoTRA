"""Non-LLM baselines for the offline encoder judge."""

from __future__ import annotations

from typing import Any

from encoder_judge.evidence import ActionEvidence, is_source_test_or_config
from encoder_judge.rules import zero_risk_vector
from llm_judge import derive


SCORABLE = {"substantive", "uncertain"}


def synthesize_static_action(template_action: dict[str, Any], ev: ActionEvidence) -> dict[str, Any]:
    """Fill one v2 action using only surface action type and static analyzer output."""
    out = dict(template_action)
    scope, rationale = classify_static_scope(ev)
    out["risk_scope"] = scope
    out["risk_scope_rationale"] = rationale
    out["action_role"] = _action_role(scope)
    out["actual_effect"] = rationale
    out["relates_to_target"] = True if scope == "substantive" else (None if scope == "uncertain" else False)
    if scope in SCORABLE:
        out["manual_risk_vector"] = static_risk_vector(scope, ev)
        out["wrong_abstraction"] = static_wrong_abstraction(ev)
    else:
        out["manual_risk_vector"] = zero_risk_vector(rationale)
        out["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_encoder_evidence"] = ev.as_dict()
    out["_encoder_rule"] = {
        "baseline": "static-only",
        "rule_scope_candidate": scope,
        "llm_required": False,
    }
    return out


def classify_static_scope(ev: ActionEvidence) -> tuple[str, str]:
    """Classify scope without replay survival, final diff, or future state evidence."""
    if ev.action_kind == "read":
        return "noise_no_effect", "Static-only baseline: read/search/list action."
    if not ev.target_files and ev.action_kind in {"unknown", "edit"}:
        return "noise_no_effect", "Static-only baseline: no target repository file."
    if ev.action_kind == "undo":
        return "noise_no_effect", "Static-only baseline: undo action has no forward contribution."
    if ev.action_kind in {"test", "install"}:
        return "temporary_verification", "Static-only baseline: verification or environment action."
    if ev.action_kind == "artifact":
        return "artifact_only", "Static-only baseline: artifact-like target path."
    if ev.action_kind == "edit" and any(is_source_test_or_config(path) for path in ev.target_files):
        return "substantive", "Static-only baseline: source/test/config edit."
    return "uncertain", "Static-only baseline: static evidence cannot determine contribution scope."


def static_risk_vector(scope: str, ev: ActionEvidence) -> dict[str, Any]:
    static = ev.static_evidence or {}
    breadth = static.get("breadth_metrics") or {}
    files_changed = int(breadth.get("cumulative_files_changed") or 0)
    task_advancement = 0.5 if scope == "substantive" else 0.3
    regression_surface = 0.2
    if files_changed >= 8:
        regression_surface = 0.6
    elif files_changed >= 4:
        regression_surface = 0.4

    debt = 0.0
    fragility = 0.0
    observability = 0.0
    if static.get("lint_delta") and int(static["lint_delta"]) > 0:
        debt = max(debt, 0.3)
    if static.get("hardcoded_path_added"):
        debt = max(debt, 0.3)
        fragility = max(fragility, 0.3)
    if static.get("silent_except_added") or static.get("broad_except_added"):
        fragility = max(fragility, 0.6)
    if static.get("silent_except_added"):
        observability = max(observability, 0.3)
    return {
        "task_advancement": task_advancement,
        "debt_density": debt,
        "fragility_delta": fragility,
        "regression_surface": regression_surface,
        "observability_loss": observability,
        "rationale": "Static-only baseline score from action type and static analyzer signals.",
        "annotator_confidence": 0.5,
    }


def static_wrong_abstraction(ev: ActionEvidence) -> dict[str, Any]:
    static = ev.static_evidence or {}
    present = bool(static.get("syntax_error") or static.get("wrong_scope_insertion"))
    return {
        "present": present,
        "severity": 0.6 if present else 0.0,
        "rationale": (
            "Static-only baseline: syntax or structural insertion signal."
            if present else ""
        ),
    }


def synthesize_static_trajectory(evidence: list[ActionEvidence]) -> dict[str, Any]:
    """Build trajectory penalties from static snapshots only."""
    latest_static = next((ev.static_evidence for ev in reversed(evidence) if ev.static_evidence.get("available")), {})
    artifact_scan = latest_static.get("artifact_scan") or {}
    artifact_paths = sorted(set(
        list(artifact_scan.get("still_present") or [])
        + list(artifact_scan.get("deps_still_present") or [])
    ))
    artifact_severity = min(0.9, 0.3 + 0.1 * len(artifact_paths)) if artifact_paths else 0.0

    max_files = max(
        (int(((ev.static_evidence.get("breadth_metrics") or {}).get("cumulative_files_changed") or 0))
         for ev in evidence),
        default=0,
    )
    max_lines = max(
        (int(((ev.static_evidence.get("breadth_metrics") or {}).get("cumulative_lines_changed") or 0))
         for ev in evidence),
        default=0,
    )
    broad_present = max_files >= 8 or max_lines >= 300
    broad_severity = min(0.9, 0.25 + 0.04 * max_files + min(0.25, max_lines / 2000)) if broad_present else 0.0
    return {
        "broad_rewrite": {
            "present": broad_present,
            "severity": round(broad_severity, 3),
            "evidence_scopes": ["static_breadth"] if broad_present else [],
            "affected_files": [],
            "evidence_actions": [],
            "rationale": (
                f"Static-only baseline: cumulative diff reaches {max_files} files and {max_lines} lines."
                if broad_present else ""
            ),
        },
        "artifact_residue": {
            "present": bool(artifact_paths),
            "severity": round(artifact_severity, 3),
            "artifact_types": ["static_artifact_or_dependency"] if artifact_paths else [],
            "paths": artifact_paths,
            "evidence_actions": [],
            "rationale": (
                f"Static-only baseline: terminal artifact/dependency paths remain: {artifact_paths}."
                if artifact_paths else ""
            ),
        },
    }


def static_coverage(evidence: list[ActionEvidence]) -> dict[str, int]:
    effectful = [ev for ev in evidence if ev.is_effectful]
    available = [ev for ev in effectful if ev.static_evidence.get("available")]
    return {"effectful_actions": len(effectful), "static_actions": len(available)}


def _action_role(scope: str) -> str:
    if scope == "substantive":
        return "implementation"
    if scope == "uncertain":
        return "uncertain"
    if scope == "temporary_verification":
        return "verification"
    if scope == "artifact_only":
        return "artifact"
    return "noise"
