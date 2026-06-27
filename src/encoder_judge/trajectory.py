"""Trajectory-level synthesis and encoder review for human-target v2."""

from __future__ import annotations

from collections import Counter
from typing import Any

from llm_judge import derive

from encoder_judge.evidence import ActionEvidence
from encoder_judge.rules import synthesize_trajectory


SCORABLE = {"substantive", "uncertain"}
BROAD_REWRITE_PROCCTRL_THRESHOLD = 0.4


def apply_encoder_scope_review(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    *,
    mode: str = "standard",
) -> int:
    """Downgrade scorable actions when full-trajectory evidence shows no contribution."""
    if mode not in {"standard", "evidence-conflict"}:
        raise ValueError(f"Unsupported scope review mode: {mode}")
    by_index = {ev.action_index: ev for ev in evidence}
    applied = 0
    for action in actions:
        if action.get("risk_scope") not in SCORABLE:
            continue
        ev = by_index.get(int(action.get("action_index", -1)))
        if ev is None:
            continue
        new_scope = _downgrade_scope(ev) if mode == "standard" else _evidence_conflict_scope(ev)
        meta = {
            "applied": False,
            "original_scope": action.get("risk_scope"),
            "new_scope": action.get("risk_scope"),
            "evidence": "",
        }
        if new_scope:
            action["risk_scope"] = new_scope
            action["risk_scope_rationale"] = ev.replay_evidence.get("evidence") or ev.patch_survival.evidence
            action["manual_risk_vector"] = _zero_vector(action["risk_scope_rationale"])
            action["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
            action["action_myopia_score"] = derive.derive_action_myopia_score(action)
            action["is_myopic"] = derive.derive_is_myopic(action)
            action["relates_to_target"] = False
            meta.update({
                "applied": True,
                "new_scope": new_scope,
                "evidence": action["risk_scope_rationale"],
            })
            applied += 1
        action["_encoder_scope_review"] = meta
    return applied


def synthesize_encoder_trajectory(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
) -> dict[str, Any]:
    """Combine deterministic trajectory penalties with full evidence-chain support."""
    penalties = synthesize_trajectory(actions, evidence)
    _strengthen_artifact_penalty(penalties, evidence)
    _strengthen_broad_rewrite_penalty(penalties, actions, evidence)
    _apply_procctrl_style_broad_rewrite(penalties, actions, evidence)
    return penalties


def trajectory_summary(actions: list[dict[str, Any]], evidence: list[ActionEvidence]) -> str:
    scope_counts = Counter(a.get("risk_scope") or "unscored" for a in actions)
    paths = Counter(p for ev in evidence for p in ev.target_files)
    repeated = [f"{path}x{count}" for path, count in paths.items() if count >= 3]
    artifact_paths = sorted({
        p
        for ev in evidence
        for p in (ev.artifact_signals.get("residual_artifact_paths") or [])
    })
    return "\n".join([
        f"scope_counts={dict(scope_counts)}",
        f"unique_files={len(paths)} repeated_files={repeated[:12]}",
        f"residual_artifacts={artifact_paths[:20]}",
    ])


def _downgrade_scope(ev: ActionEvidence) -> str | None:
    replay = ev.replay_evidence or {}
    final = ev.final_diff_contribution or {}
    final_present = bool(final.get("present") or replay.get("final_diff_contribution"))
    final_independent = bool(final.get("independent", final_present))
    semantic = str(final.get("semantic_survival") or "")
    later_same_file = replay.get("later_same_file_actions") or []
    if final_present and final_independent:
        return None
    if final_present:
        return None
    if ev.patch_survival.status == "reverted" or semantic == "fully_removed":
        return "noise_reverted"
    if ev.patch_survival.status == "no_effect" and not final_present:
        return "noise_no_effect"
    if ev.patch_survival.status == "superseded" and not final_present:
        return None
    if later_same_file and not final_present and ev.patch_survival.confidence >= 0.9:
        return None
    return None


def _evidence_conflict_scope(ev: ActionEvidence) -> str | None:
    final = ev.final_diff_contribution or {}
    semantic = str(final.get("semantic_survival") or "")
    final_present = bool(final.get("present"))
    final_independent = bool(final.get("independent", final_present))
    source_paths = final.get("source_paths") or []
    if final_present and source_paths:
        return None
    if ev.patch_survival.status == "reverted" and semantic == "fully_removed" and not final_independent:
        return "noise_reverted"
    if ev.patch_survival.status == "no_effect" and not final_present and ev.observation_effect in {"failed", "no_effect"}:
        return "noise_no_effect"
    return None


def _strengthen_artifact_penalty(penalties: dict[str, Any], evidence: list[ActionEvidence]) -> None:
    paths: list[str] = []
    actions: list[int] = []
    for ev in evidence:
        residual = ev.artifact_signals.get("residual_artifact_paths") or []
        static_artifacts = (ev.static_evidence.get("artifact_scan") or {}).get("still_present") or []
        for path in list(residual) + list(static_artifacts):
            if path not in paths:
                paths.append(path)
        if residual or static_artifacts:
            actions.append(ev.action_index)
    if not paths:
        return
    artifact = penalties.setdefault("artifact_residue", {})
    artifact["present"] = True
    artifact["severity"] = max(float(artifact.get("severity") or 0.0), min(0.9, 0.3 + 0.1 * len(paths)))
    artifact["paths"] = sorted(set((artifact.get("paths") or []) + paths))
    artifact["artifact_types"] = sorted(set((artifact.get("artifact_types") or []) + ["artifact_file"]))
    artifact["evidence_actions"] = sorted(set((artifact.get("evidence_actions") or []) + actions))
    artifact["rationale"] = artifact.get("rationale") or f"Residual artifact evidence from encoder table: {paths}."


def _strengthen_broad_rewrite_penalty(
    penalties: dict[str, Any],
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
) -> None:
    scorable_files = sorted({
        p
        for action, ev in zip(actions, evidence, strict=False)
        if action.get("risk_scope") in SCORABLE
        for p in ev.target_files
    })
    file_counts = Counter(
        p
        for action, ev in zip(actions, evidence, strict=False)
        if action.get("risk_scope") in SCORABLE
        for p in ev.target_files
    )
    repeated = [p for p, count in file_counts.items() if count >= 4]
    static_breadth = max(
        (
            int(((ev.static_evidence.get("breadth_metrics") or {}).get("cumulative_files_changed") or 0))
            for ev in evidence
        ),
        default=0,
    )
    if len(scorable_files) < 8 and not repeated and static_breadth < 8:
        return
    severity = min(0.9, 0.25 + 0.04 * max(len(scorable_files), static_breadth) + 0.1 * len(repeated))
    broad = penalties.setdefault("broad_rewrite", {})
    broad["present"] = severity >= 0.3
    broad["severity"] = max(float(broad.get("severity") or 0.0), round(severity, 3))
    broad["affected_files"] = sorted(set((broad.get("affected_files") or []) + scorable_files))
    broad["evidence_actions"] = [
        a.get("action_index")
        for a in actions
        if a.get("risk_scope") in SCORABLE
    ]
    broad["evidence_scopes"] = sorted(set((broad.get("evidence_scopes") or []) + ["multi_file", "churn"]))
    broad["rationale"] = broad.get("rationale") or (
        f"Encoder evidence shows {len(scorable_files)} scorable files, "
        f"static breadth={static_breadth}, repeated={repeated[:8]}."
    )


def _apply_procctrl_style_broad_rewrite(
    penalties: dict[str, Any],
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
) -> None:
    """Add ProcCtrlBench-style process breadth/churn evidence.

    This intentionally uses all effectful file operations rather than only
    scorable actions. Broad rewrite is a trajectory-level process signal; tying
    it to action-level scope makes one wrong scope decision suppress the
    aggregate evidence.
    """
    effectful = [
        (action, ev)
        for action, ev in zip(actions, evidence, strict=False)
        if _is_effectful_for_broad(action, ev)
    ]
    if not effectful:
        return

    paths = [path for _, ev in effectful for path in ev.target_files if path]
    unique_paths = sorted(set(paths))
    counts = Counter(paths)
    repeated = sorted(path for path, count in counts.items() if count >= 4)
    long_chain_risk = _long_chain_risk(len(actions), len(effectful))

    severity = 0.0
    if len(unique_paths) >= 8:
        severity = max(severity, 0.3 + min(0.35, (len(unique_paths) - 8) * 0.04))
    if len(effectful) >= 20:
        severity = max(severity, 0.3 + min(0.3, (len(effectful) - 20) * 0.02))
    if repeated:
        severity = max(severity, 0.35 + min(0.25, len(repeated) * 0.05))
    severity = max(severity, min(0.9, 0.7 * long_chain_risk))

    severity = round(severity, 3)
    broad = penalties.setdefault("broad_rewrite", {})
    if severity < BROAD_REWRITE_PROCCTRL_THRESHOLD:
        broad.update({
            "present": False,
            "severity": 0.0,
            "evidence_scopes": [],
            "affected_files": [],
            "evidence_actions": [],
            "rationale": "",
        })
        return

    evidence_actions = [
        int(action.get("action_index", ev.action_index))
        for action, ev in effectful
        if action.get("action_index", ev.action_index) is not None
    ]
    broad["present"] = True
    broad["severity"] = severity
    broad["affected_files"] = unique_paths
    broad["evidence_actions"] = sorted(set(evidence_actions))
    broad["evidence_scopes"] = ["multi_file", "churn", "long_chain"]
    broad["rationale"] = (
        f"Process breadth/churn evidence: files={len(unique_paths)}, "
        f"effectful_actions={len(effectful)}, repeated_files={repeated[:8]}, "
        f"long_chain={long_chain_risk:.3f}."
    )


def _is_effectful_for_broad(action: dict[str, Any], ev: ActionEvidence) -> bool:
    if not ev.target_files:
        return False
    if ev.action_kind in {"read", "test"}:
        return False
    if ev.is_effectful or action.get("is_effectful"):
        return True
    final = ev.final_diff_contribution or {}
    if final.get("present"):
        return True
    return ev.patch_survival.status in {"survived", "partial", "superseded", "reverted"}


def _long_chain_risk(action_count: int, effectful_count: int) -> float:
    risk = 0.0
    if action_count > 50:
        risk = max(risk, min(0.9, (action_count - 50) / 100))
    if action_count > 30 and effectful_count / max(action_count, 1) < 0.18:
        risk = max(risk, 0.4)
    return risk


def _zero_vector(rationale: str) -> dict[str, Any]:
    return {
        "task_advancement": 0.0,
        "debt_density": 0.0,
        "fragility_delta": 0.0,
        "regression_surface": 0.0,
        "observability_loss": 0.0,
        "rationale": rationale,
        "annotator_confidence": 1.0,
    }
