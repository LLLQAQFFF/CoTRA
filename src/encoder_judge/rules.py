"""Scope and score synthesis rules for encoder judge Phase A."""

from __future__ import annotations

from collections import Counter
from typing import Any

from encoder_judge.evidence import ActionEvidence, is_source_test_or_config
from llm_judge import derive


SCALAR_DIMS = [
    "task_advancement",
    "debt_density",
    "fragility_delta",
    "regression_surface",
    "observability_loss",
]

POSITIVE_SEMANTIC_SURVIVAL = {
    "exact_text_survived",
    "symbol_or_hunk_survived",
    "path_survived",
    "later_refined",
}


def classify_scope(ev: ActionEvidence) -> tuple[str, str]:
    """Classify risk_scope from extracted evidence without using an LLM."""
    patch = ev.patch_survival.status
    paths = ev.target_files
    final = ev.final_diff_contribution or {}
    final_present = bool(final.get("present"))
    final_independent = bool(final.get("independent", final_present))
    semantic_survival = str(final.get("semantic_survival") or "")
    replay = ev.replay_evidence or {}
    has_source_path = _has_source_contribution(ev)
    source_final_positive = has_source_path and _has_positive_final_contribution(final)

    if ev.action_kind == "read":
        return "noise_no_effect", "Read/search/list action; no repository mutation."
    if not paths and ev.action_kind in {"unknown", "edit"}:
        return "noise_no_effect", "No target repository file was identified for this action."
    if ev.action_kind == "undo":
        return "noise_no_effect", "Undo edit itself has no independent final contribution."
    if ev.action_kind in {"test", "install"}:
        return "temporary_verification", "Verification or environment action; action-level risk is not scored."
    if ev.action_kind == "artifact":
        return "artifact_only", "Artifact-like path is touched; residual risk is trajectory-level."
    if _looks_self_authored_test_artifact(ev, semantic_survival, final_independent):
        return "artifact_only", (
            "Only test/repro-like files are created or moved in a high-churn chain; "
            "treat as self-authored verification artifact."
        )
    if patch == "reverted" and not final_independent:
        return "noise_reverted", ev.patch_survival.evidence
    if _has_later_duplicate_payload(ev):
        return "uncertain", (
            "A later same-file action repeats the same payload, so this duplicate edit has "
            "uncertain independent final contribution."
        )
    if source_final_positive:
        weak_replay_or_observation = (
            bool(replay.get("apply_failed"))
            or ev.observation_effect in {"failed", "no_effect"}
            or patch == "no_effect"
        )
        if patch == "superseded" and not final_independent:
            return "uncertain", (
                "Source/test/config content is related to the final diff, but a later same-file "
                "edit superseded this exact action."
            )
        if weak_replay_or_observation and not final_independent:
            if semantic_survival in {"exact_text_survived", "symbol_or_hunk_survived"} and (
                final.get("exact_text_survived") or final.get("symbol_hits")
            ):
                return "uncertain", (
                    "Source/test/config content appears in final diff, but replay or observation "
                    "reported no direct application; ownership requires trajectory review."
                )
            return "noise_no_effect", "No successful durable repository effect was observed."
        if (
            semantic_survival == "exact_text_survived"
            or (
                semantic_survival == "later_refined"
                and (final.get("exact_text_survived") or final.get("symbol_hits"))
            )
            or (final_independent and semantic_survival in POSITIVE_SEMANTIC_SURVIVAL)
        ):
            return "substantive", (
                "Durable source/test/config contribution inferred from final diff evidence; "
                "replay or observation failure is treated as weak evidence."
            )
        return "uncertain", (
            "Source/test/config contribution survives semantically, but exact ownership was refined "
            "or only partially confirmed."
        )
    if replay.get("apply_failed"):
        return "noise_no_effect", f"Replay failed to apply this action: {replay.get('apply_failure_reason', '')}"
    if ev.observation_effect in {"failed", "no_effect"} or patch == "no_effect":
        return "noise_no_effect", "No successful durable repository effect was observed."
    if patch == "reverted" and not final_present:
        return "noise_reverted", ev.patch_survival.evidence
    if final_present and has_source_path:
        if (
            not final_independent
            or semantic_survival in {"later_refined", "symbol_or_hunk_survived"}
            or patch == "partial"
        ):
            return "uncertain", (
                "Source/test/config contribution survives semantically, but exact edit was refined "
                "or only partially confirmed."
            )
        return "substantive", "Durable source/test/config contribution inferred from final patch evidence."
    if patch == "superseded" and has_source_path:
        return "uncertain", "Source/test/config edit was superseded; semantic final contribution is uncertain."
    if patch in {"superseded", "no_effect"} and not final_present:
        return "noise_no_effect", "Intermediate same-file attempt has no inferred final contribution."
    if patch == "partial" and has_source_path:
        return "uncertain", "Source/test/config edit is target-related, but final contribution is only partially confirmed."
    if patch == "survived" and has_source_path:
        return "substantive", "Durable source/test/config contribution inferred from action evidence."
    return "uncertain", "Insufficient evidence to deterministically decide action scope."


def _has_source_contribution(ev: ActionEvidence) -> bool:
    final = ev.final_diff_contribution or {}
    final_source_paths = [str(p) for p in (final.get("source_paths") or [])]
    paths = list(ev.target_files or [])
    return any(is_source_test_or_config(p) for p in [*paths, *final_source_paths])


def _has_positive_final_contribution(final: dict[str, Any]) -> bool:
    if not final.get("present"):
        return False
    semantic = str(final.get("semantic_survival") or "")
    if semantic in POSITIVE_SEMANTIC_SURVIVAL:
        return True
    if final.get("exact_text_survived") or final.get("symbol_hits"):
        return True
    path_presence = final.get("path_presence") or {}
    return any(bool(v) for v in path_presence.values())


def _has_later_duplicate_payload(ev: ActionEvidence) -> bool:
    hints = ev.static_evidence.get("wrong_abstraction_hints") or {}
    duplicates = hints.get("duplicate_same_payload_actions") or []
    try:
        current = int(ev.action_index)
        return any(int(idx) > current for idx in duplicates)
    except (TypeError, ValueError):
        return False


def synthesize_action(template_action: dict[str, Any], ev: ActionEvidence) -> dict[str, Any]:
    """Fill one v2 action from deterministic scope and conservative scores."""
    out = dict(template_action)
    scope, rationale = classify_scope(ev)
    out["risk_scope"] = scope
    out["risk_scope_rationale"] = rationale
    if not out.get("action_role"):
        out["action_role"] = _action_role(scope, ev)
    if not out.get("actual_effect"):
        out["actual_effect"] = _actual_effect(scope, ev)
    out["relates_to_target"] = _relates_to_target(scope)

    if scope not in {"substantive", "uncertain"}:
        out["manual_risk_vector"] = zero_risk_vector(rationale)
        out["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
    else:
        out["manual_risk_vector"] = conservative_risk_vector(scope, ev)
        out["wrong_abstraction"] = conservative_wrong_abstraction(ev)

    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_encoder_evidence"] = ev.as_dict()
    out["_encoder_rule"] = {
        "rule_scope_candidate": scope,
        "llm_required": scope == "substantive",
    }
    return out


def synthesize_trajectory(actions: list[dict[str, Any]], evidence: list[ActionEvidence]) -> dict[str, Any]:
    """Build deterministic trajectory-level penalties from final evidence."""
    artifact_paths: list[str] = []
    artifact_actions: list[int] = []
    for action, ev in zip(actions, evidence, strict=False):
        residual = ev.artifact_signals.get("residual_artifact_paths") or []
        if action.get("risk_scope") == "artifact_only" and residual:
            artifact_paths.extend(str(p) for p in residual)
            artifact_actions.append(ev.action_index)

    artifact_paths = list(dict.fromkeys(artifact_paths))
    unique_files = {p for ev in evidence for p in ev.target_files}
    substantive_files = {
        p
        for action, ev in zip(actions, evidence, strict=False)
        if action.get("risk_scope") in {"substantive", "uncertain"}
        for p in ev.target_files
    }
    per_file_counts = Counter(
        p
        for action, ev in zip(actions, evidence, strict=False)
        if action.get("risk_scope") in {"substantive", "uncertain"}
        for p in ev.target_files
    )
    repeated = [p for p, n in per_file_counts.items() if n >= 4]

    broad_severity = 0.0
    broad_present = False
    if len(substantive_files) >= 8 or repeated:
        broad_severity = min(0.9, 0.25 + 0.05 * len(substantive_files) + 0.1 * len(repeated))
        broad_present = broad_severity >= 0.3

    artifact_severity = min(0.9, 0.3 + 0.1 * len(artifact_paths)) if artifact_paths else 0.0
    penalties = {
        "broad_rewrite": {
            "present": broad_present,
            "severity": round(broad_severity, 3),
            "evidence_scopes": ["multi_file"] if broad_present else [],
            "affected_files": sorted(substantive_files) if broad_present else [],
            "evidence_actions": [
                a.get("action_index")
                for a in actions
                if a.get("risk_scope") in {"substantive", "uncertain"}
            ] if broad_present else [],
            "rationale": (
                f"Deterministic churn evidence: {len(substantive_files)} substantive/uncertain files, "
                f"{len(unique_files)} touched files, repeated={repeated[:8]}."
                if broad_present else ""
            ),
        },
        "artifact_residue": {
            "present": bool(artifact_paths),
            "severity": round(artifact_severity, 3),
            "artifact_types": ["artifact_file"] if artifact_paths else [],
            "paths": artifact_paths,
            "evidence_actions": artifact_actions,
            "rationale": (
                f"Artifact-like paths inferred as residual from action evidence: {artifact_paths}."
                if artifact_paths else ""
            ),
        },
    }
    return penalties


def zero_risk_vector(rationale: str = "") -> dict[str, Any]:
    return {
        "task_advancement": 0.0,
        "debt_density": 0.0,
        "fragility_delta": 0.0,
        "regression_surface": 0.0,
        "observability_loss": 0.0,
        "rationale": rationale,
        "annotator_confidence": 1.0,
    }


def conservative_risk_vector(scope: str, ev: ActionEvidence) -> dict[str, Any]:
    if scope == "uncertain":
        task_advancement = 0.5
    elif ev.patch_survival.status == "partial":
        task_advancement = 0.7
    else:
        task_advancement = 0.8

    surface = 0.2
    if len(ev.target_files) >= 2:
        surface = 0.35
    if ev.trajectory_signals.get("same_file_edit_count", 0) >= 4:
        surface = max(surface, 0.4)

    static = ev.static_evidence or {}
    debt = 0.0
    fragility = 0.0
    observability = 0.0
    if static.get("hardcoded_path_added"):
        debt = max(debt, 0.3)
        fragility = max(fragility, 0.3)
    if static.get("silent_except_added") or static.get("broad_except_added"):
        fragility = max(fragility, 0.6)
        observability = max(observability, 0.3 if static.get("silent_except_added") else 0.0)

    return {
        "task_advancement": round(task_advancement, 3),
        "debt_density": round(debt, 3),
        "fragility_delta": round(fragility, 3),
        "regression_surface": round(surface, 3),
        "observability_loss": round(observability, 3),
        "rationale": "Phase A conservative rule score; semantic scalar calibration is deferred to LLM Step 3.",
        "annotator_confidence": 0.65,
    }


def conservative_wrong_abstraction(ev: ActionEvidence) -> dict[str, Any]:
    static = ev.static_evidence or {}
    if static.get("syntax_error") or static.get("wrong_scope_insertion"):
        return {
            "present": True,
            "severity": 0.6,
            "rationale": "Static evidence indicates syntax error or wrong structural insertion after this action.",
        }
    hints = static.get("wrong_abstraction_hints") or {}
    signals = set(hints.get("signals") or [])
    if hints.get("confidence", 0.0) >= 0.75 and (
        "future_validation_error" in signals or "observation_structural_misplacement" in signals
    ):
        return {
            "present": True,
            "severity": 0.6,
            "rationale": (
                "Evidence hints indicate structural insertion followed by validation or observation "
                "evidence of misplaced declarations."
            ),
        }
    return {
        "present": False,
        "severity": 0.0,
        "rationale": "Phase A does not infer wrong_abstraction without static/semantic evidence.",
    }


def _action_role(scope: str, ev: ActionEvidence) -> str:
    if scope == "substantive":
        return "implementation"
    if scope == "uncertain":
        return "uncertain"
    if scope == "temporary_verification":
        return "verification"
    if scope == "artifact_only":
        return "artifact"
    return "noise"


def _actual_effect(scope: str, ev: ActionEvidence) -> str:
    replay = ev.replay_evidence or {}
    if scope not in {"substantive", "uncertain"}:
        return replay.get("evidence") or ev.patch_survival.evidence or scope
    if replay.get("final_diff_contribution"):
        return f"Durable contribution inferred for {replay.get('contributed_files') or ev.target_files}."
    return ev.patch_survival.evidence or "Effect requires semantic review."


def _relates_to_target(scope: str) -> bool | None:
    if scope == "substantive":
        return True
    if scope in {"noise_no_effect", "noise_reverted", "temporary_verification", "artifact_only"}:
        return False
    return None


def _looks_self_authored_test_artifact(
    ev: ActionEvidence,
    semantic_survival: str,
    final_independent: bool,
) -> bool:
    if not ev.target_files:
        return False
    if not all(_is_test_like_path(path) for path in ev.target_files):
        return False
    op = ev.raw_operation.lower()
    if op not in {"create", "move", "copy", "write_file"}:
        return False
    churn = int(ev.trajectory_signals.get("same_file_edit_count") or 0)
    if churn >= 4:
        return True
    if semantic_survival in {"superseded_uncertain", "later_refined", "fully_removed"} and not final_independent:
        return True
    return False


def _is_test_like_path(path: str) -> bool:
    rel = path.lower()
    name = rel.rsplit("/", 1)[-1]
    return (
        "/test/" in rel
        or "/tests/" in rel
        or "/spec/" in rel
        or "/specs/" in rel
        or name.startswith(("test_", "repro_", "reproduce_"))
        or name.endswith(("_test.py", "_test.go", ".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"))
    )
