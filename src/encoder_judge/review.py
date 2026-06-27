"""Trajectory-level encoder review for human-target v2.

The review stage gives the LLM a compressed full-trajectory evidence packet and
accepts only structured overrides.  It is deliberately not a replacement for
the deterministic evidence builder or action-level semantic scorer.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge import config, derive
from llm_judge.client import JudgeClient
from llm_judge.cost import CostTracker
from llm_judge.parser import _extract_json

from encoder_judge.evidence import ActionEvidence, PatchSurvival, load_final_patch, load_sidecar
from encoder_judge.rules import conservative_risk_vector, zero_risk_vector


VALID_RISK_SCOPES = {
    "substantive",
    "uncertain",
    "noise_no_effect",
    "noise_reverted",
    "artifact_only",
    "temporary_verification",
}
SCORABLE_SCOPES = {"substantive", "uncertain"}
SCOPE_OVERRIDE_THRESHOLD = 0.75
WRONG_ABSTRACTION_THRESHOLD = 0.65

MAX_REVIEW_ACTIONS = 30
MAX_RAW_ACTION_CHARS = 800
MAX_OBSERVATION_CHARS = 800
MAX_FINAL_DIFF_SNIPPETS_PER_FILE = 5
MAX_PACKET_CHARS = 14000


@dataclass
class TrajectoryReviewResult:
    scope_overrides: list[dict[str, Any]] = field(default_factory=list)
    wrong_abstraction_overrides: list[dict[str, Any]] = field(default_factory=list)
    trajectory_penalties: dict[str, dict[str, Any]] = field(default_factory=dict)
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


def should_run_trajectory_review(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    penalties: dict[str, Any],
) -> tuple[bool, str]:
    """Return whether the trajectory has enough signal to justify one review call."""
    if any(a.get("risk_scope") in SCORABLE_SCOPES for a in actions):
        return True, ""
    if any(a.get("risk_scope") == "uncertain" for a in actions):
        return True, ""
    if any(ev.action_kind == "artifact" or ev.artifact_signals.get("introduced_artifact_paths") for ev in evidence):
        return True, ""
    if any((ev.trajectory_signals.get("same_file_edit_count") or 0) >= 3 for ev in evidence):
        return True, ""
    if any(_static_has_review_signal(ev.static_evidence) for ev in evidence):
        return True, ""
    if any((penalties.get(name) or {}).get("present") for name in ("broad_rewrite", "artifact_residue")):
        return True, ""
    return False, "trivial_trajectory"


def review_trajectory(
    *,
    template: dict[str, Any],
    template_path: Path,
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    deterministic_penalties: dict[str, Any],
    task_description: str,
    client: JudgeClient,
    model: str,
    tracker: CostTracker | None = None,
) -> tuple[TrajectoryReviewResult, dict[str, Any]]:
    """Call the trajectory-level review LLM and return parsed overrides plus packet metadata."""
    packet = build_review_packet(
        template=template,
        template_path=template_path,
        actions=actions,
        evidence=evidence,
        deterministic_penalties=deterministic_penalties,
        task_description=task_description,
    )
    user = build_review_prompt(packet)
    call = client.call(
        model=model,
        system="Return valid JSON only. Do not include markdown or extra commentary.",
        user=user,
        max_tokens=config.MAX_TOKENS_TRAJECTORY,
    )
    if tracker is not None:
        tracker.record(call)
    result = parse_trajectory_review(call.text)
    return result, {
        "packet_action_count": len(packet.get("top_suspicious_actions") or []),
        "packet_truncated": bool(packet.get("packet_truncated")),
        "packet_chars": len(user),
        "response_chars": len(result.raw),
    }


def build_review_packet(
    *,
    template: dict[str, Any],
    template_path: Path,
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    deterministic_penalties: dict[str, Any],
    task_description: str,
) -> dict[str, Any]:
    """Build a cost-bounded full-trajectory evidence packet."""
    candidate = load_sidecar(template_path, ".candidate.json")
    traj = load_sidecar(template_path, ".traj.json")
    final_patch = load_final_patch(template_path, candidate)
    obs_by_index = {
        int(item.get("step_index")): str(item.get("observation") or "")
        for item in (traj.get("trajectory") or [])
        if item.get("step_index") is not None
    }
    raw_by_index = {
        int(item.get("step_index")): str(item.get("action") or item.get("raw_action") or "")
        for item in (traj.get("trajectory") or [])
        if item.get("step_index") is not None
    }

    by_index = {ev.action_index: ev for ev in evidence}
    action_by_index = {int(a.get("action_index", -1)): a for a in actions if a.get("action_index") is not None}
    selected = _select_review_actions(actions, evidence)
    top_actions = [
        _packet_action(action_by_index[idx], by_index[idx], raw_by_index, obs_by_index)
        for idx in selected
        if idx in action_by_index and idx in by_index
    ]

    packet = {
        "schema": "trajectory-encoder-review-v1",
        "truncation_notice": (
            "[truncated] means content was shortened only for cost control. It is not evidence "
            "that the command failed, output stopped, or the action had no effect."
        ),
        "task_description": _truncate_text(task_description, 1000)["text"],
        "candidate_summary": _candidate_summary(candidate),
        "scope_summary": dict(Counter(a.get("risk_scope") or "unscored" for a in actions)),
        "deterministic_penalties": deterministic_penalties,
        "same_file_chains": _same_file_chains(evidence),
        "ownership_chains": _ownership_chains(actions, evidence, raw_by_index, obs_by_index),
        "artifact_lifecycle": _artifact_lifecycle(evidence),
        "wrong_abstraction_candidates": _wrong_abstraction_candidates(actions, evidence),
        "validation_error_timeline": _validation_error_timeline(actions, evidence, obs_by_index),
        "final_diff_summary": _final_diff_summary(final_patch),
        "top_suspicious_actions": top_actions,
    }
    return _enforce_packet_budget(packet)


def build_review_prompt(packet: dict[str, Any]) -> str:
    return f"""Trajectory-level human-target v2 encoder review. Treat packet evidence as facts and output JSON only.

Focus on high-confidence corrections. wrong_abstraction means implementation placed in the wrong structural location,
abstraction, interface, dependency direction, schema parent, or module boundary. Candidate hints are not proof.
Truncation is not evidence. [truncated] only means cost-control truncation; do not treat it as failure/no-effect evidence.
Use final_diff_contribution.semantic_survival to distinguish exact-text overwrite from semantic contribution:
later_refined/path_survived/symbol_or_hunk_survived can still be substantive; fully_removed is reverted/no-effect.
Use ownership_chains to decide which same-file actions own the final implementation. An earlier action should be
noise_reverted only when later edits clearly replaced/removed its independent contribution. A later refinement can still
leave an earlier action substantive when the earlier target behavior remains part of the final implementation.
Artifact-like temporary scripts, repro files, local debug configs, generated files, and one-off docs should remain artifact_only
even if their execution failed.

Output shape:
{{
  "scope_overrides": [{{"action_index": 52, "risk_scope": "substantive|uncertain|noise_reverted|artifact_only|noise_no_effect|temporary_verification", "confidence": 0.0, "rationale": "..."}}],
  "wrong_abstraction_overrides": [{{"action_index": 55, "present": true, "severity": 0.0, "confidence": 0.0, "rationale": "..."}}],
  "trajectory_penalties": {{
    "broad_rewrite": {{"present": true, "severity": 0.0, "evidence_actions": [], "affected_files": [], "rationale": "..."}},
    "artifact_residue": {{"present": true, "severity": 0.0, "paths": [], "evidence_actions": [], "rationale": "..."}}
  }}
}}

{json.dumps(packet, ensure_ascii=False, indent=2)}
"""


def parse_trajectory_review(text: str) -> TrajectoryReviewResult:
    data = _extract_json(text)
    if data is None:
        return TrajectoryReviewResult(parse_failed=True, parse_failures=["json"], raw=text)

    failures: list[str] = []
    scope_overrides = _parse_scope_overrides(data.get("scope_overrides"), failures)
    wa_overrides = _parse_wrong_abstraction_overrides(data.get("wrong_abstraction_overrides"), failures)
    penalties = _parse_review_penalties(data.get("trajectory_penalties"), failures)
    return TrajectoryReviewResult(
        scope_overrides=scope_overrides,
        wrong_abstraction_overrides=wa_overrides,
        trajectory_penalties=penalties,
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def apply_trajectory_review(
    actions: list[dict[str, Any]],
    deterministic_penalties: dict[str, Any],
    result: TrajectoryReviewResult,
    *,
    apply_wrong_abstraction: bool = True,
    apply_scope_overrides: bool = True,
) -> dict[str, Any]:
    """Apply valid high-confidence review overrides and return metadata."""
    by_index = {int(a.get("action_index", -1)): a for a in actions if a.get("action_index") is not None}
    meta = {
        "scope_overrides_applied": 0,
        "wrong_abstraction_overrides_applied": 0,
        "trajectory_penalties_applied": False,
    }

    if apply_scope_overrides:
        for override in result.scope_overrides:
            action = by_index.get(override["action_index"])
            if action is None or override["confidence"] < SCOPE_OVERRIDE_THRESHOLD:
                continue
            old_scope = action.get("risk_scope")
            new_scope = override["risk_scope"]
            if not _allowed_scope_override(action, old_scope, new_scope):
                continue
            action["risk_scope"] = new_scope
            action["risk_scope_rationale"] = override["rationale"]
            if new_scope not in SCORABLE_SCOPES:
                action["manual_risk_vector"] = zero_risk_vector(override["rationale"])
                action["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
                action["relates_to_target"] = False
            elif old_scope not in SCORABLE_SCOPES:
                ev = (action.get("_encoder_evidence") or {})
                action["manual_risk_vector"] = _fallback_scorable_vector(new_scope, ev, override["rationale"])
                action["relates_to_target"] = True
            action["action_myopia_score"] = derive.derive_action_myopia_score(action)
            action["is_myopic"] = derive.derive_is_myopic(action)
            action["_encoder_trajectory_review"] = {
                "scope_override_applied": True,
                "old_scope": old_scope,
                "new_scope": new_scope,
                "confidence": override["confidence"],
                "rationale": override["rationale"],
            }
            meta["scope_overrides_applied"] += 1

    if apply_wrong_abstraction:
        for override in result.wrong_abstraction_overrides:
            action = by_index.get(override["action_index"])
            if action is None or override["confidence"] < WRONG_ABSTRACTION_THRESHOLD:
                continue
            if action.get("risk_scope") not in SCORABLE_SCOPES:
                continue
            action["wrong_abstraction"] = {
                "present": bool(override["present"]) and override["severity"] >= 0.3,
                "severity": override["severity"] if override["present"] else 0.0,
                "rationale": override["rationale"] if override["present"] else "",
            }
            action["action_myopia_score"] = derive.derive_action_myopia_score(action)
            action["is_myopic"] = derive.derive_is_myopic(action)
            review_meta = action.setdefault("_encoder_trajectory_review", {})
            review_meta["wrong_abstraction_override_applied"] = True
            review_meta["wrong_abstraction_confidence"] = override["confidence"]
            meta["wrong_abstraction_overrides_applied"] += 1

    penalties = deterministic_penalties
    if result.trajectory_penalties:
        penalties = _merge_review_penalties(deterministic_penalties, result.trajectory_penalties)
        meta["trajectory_penalties_applied"] = True
    return {"penalties": penalties, "meta": meta}


def _allowed_scope_override(action: dict[str, Any], old_scope: str, new_scope: str) -> bool:
    evidence = action.get("_encoder_evidence") or {}
    final = evidence.get("final_diff_contribution") or {}
    patch = evidence.get("patch_survival") or {}
    action_kind = str(evidence.get("action_kind") or action.get("action_kind") or "")
    if action_kind == "artifact" and new_scope in SCORABLE_SCOPES:
        return False
    if old_scope == "artifact_only" and new_scope in SCORABLE_SCOPES:
        return False
    if old_scope not in SCORABLE_SCOPES and new_scope in SCORABLE_SCOPES:
        if not (bool(final.get("present")) and bool(final.get("source_paths"))):
            return False
        if str(patch.get("status") or "") in {"reverted", "superseded"} and not bool(final.get("independent")):
            return False
        if _has_later_duplicate_payload(evidence):
            return False
        return True
    if new_scope == "substantive":
        if str(patch.get("status") or "") in {"reverted", "superseded"} and not bool(final.get("independent")):
            return False
        if _has_later_duplicate_payload(evidence):
            return False
    return True


def _has_later_duplicate_payload(evidence: dict[str, Any]) -> bool:
    hints = ((evidence.get("static_evidence") or {}).get("wrong_abstraction_hints") or {})
    duplicates = hints.get("duplicate_same_payload_actions") or []
    try:
        current = int(evidence.get("action_index", -1))
        return any(int(idx) > current for idx in duplicates)
    except (TypeError, ValueError):
        return False


def _fallback_scorable_vector(scope: str, evidence_dict: dict[str, Any], rationale: str) -> dict[str, Any]:
    """Provide conservative scores when review upgrades an unscored action."""
    try:
        patch = evidence_dict.get("patch_survival") or {}
        ev = ActionEvidence(
            action_index=int(evidence_dict.get("action_index", -1)),
            action_id=str(evidence_dict.get("action_id") or ""),
            action_kind=str(evidence_dict.get("action_kind") or "unknown"),
            target_files=list(evidence_dict.get("target_files") or []),
            is_effectful=bool(evidence_dict.get("is_effectful")),
            raw_operation=str(evidence_dict.get("raw_operation") or ""),
            observation_effect=str(evidence_dict.get("observation_effect") or "unknown"),
            patch_survival=PatchSurvival(
                status=str(patch.get("status") or "unknown"),
                confidence=float(patch.get("confidence") or 0.0),
                later_same_file_actions=list(patch.get("later_same_file_actions") or []),
                evidence=str(patch.get("evidence") or ""),
            ),
            final_diff_contribution=dict(evidence_dict.get("final_diff_contribution") or {}),
            artifact_signals=dict(evidence_dict.get("artifact_signals") or {}),
            trajectory_signals=dict(evidence_dict.get("trajectory_signals") or {}),
            raw_action=str(evidence_dict.get("raw_action") or ""),
            replay_evidence=dict(evidence_dict.get("replay_evidence") or {}),
            static_evidence=dict(evidence_dict.get("static_evidence") or {}),
            state_evidence=dict(evidence_dict.get("state_evidence") or {}),
        )
        vector = conservative_risk_vector(scope, ev)
    except Exception:
        vector = {
            "task_advancement": 0.5 if scope == "uncertain" else 0.7,
            "debt_density": 0.0,
            "fragility_delta": 0.0,
            "regression_surface": 0.2,
            "observability_loss": 0.0,
            "annotator_confidence": 0.55,
        }
    vector["rationale"] = f"Trajectory review upgraded scope; conservative fallback score. {rationale}"
    return vector


def _select_review_actions(actions: list[dict[str, Any]], evidence: list[ActionEvidence]) -> list[int]:
    by_idx = {ev.action_index: ev for ev in evidence}
    scores: list[tuple[int, int]] = []
    for action in actions:
        idx = int(action.get("action_index", -1))
        ev = by_idx.get(idx)
        if ev is None:
            continue
        score = 0
        if action.get("risk_scope") in SCORABLE_SCOPES:
            score += 3
        if action.get("risk_scope") == "uncertain":
            score += 3
        if (action.get("wrong_abstraction") or {}).get("present"):
            score += 4
        wa_hints = (ev.static_evidence.get("wrong_abstraction_hints") or {})
        if float(wa_hints.get("confidence") or 0.0) >= 0.65:
            score += 5
        if ev.action_kind == "artifact" or ev.artifact_signals.get("introduced_artifact_paths"):
            score += 5
        if ev.patch_survival.status in {"partial", "superseded", "reverted"}:
            score += 3
        semantic = (ev.final_diff_contribution or {}).get("semantic_survival")
        final = ev.final_diff_contribution or {}
        final_present = bool(final.get("present"))
        source_paths = final.get("source_paths") or []
        if semantic in {"later_refined", "superseded_uncertain", "fully_removed"}:
            score += 5
        if source_paths and final_present and ev.observation_effect in {"failed", "no_effect"}:
            score += 9
        if source_paths and final_present and ev.patch_survival.status == "no_effect":
            score += 9
        if (
            action.get("risk_scope") == "noise_no_effect"
            and source_paths
            and semantic in {"exact_text_survived", "symbol_or_hunk_survived", "path_survived", "later_refined"}
        ):
            score += 10
        if (
            action.get("risk_scope") == "noise_reverted"
            and source_paths
            and semantic != "fully_removed"
        ):
            score += 8
        if ev.action_kind == "edit" and ev.patch_survival.status == "superseded" and ev.target_files:
            score += 4
        if ev.action_kind == "artifact" and ev.observation_effect in {"failed", "no_effect"}:
            score += 4
        if (
            action.get("risk_scope") == "substantive"
            and semantic == "fully_removed"
        ):
            score += 5
        if (
            action.get("risk_scope") == "substantive"
            and ev.patch_survival.status == "survived"
            and len(ev.patch_survival.later_same_file_actions) >= 2
        ):
            score += 7
        if (
            action.get("risk_scope") in {"substantive", "noise_reverted"}
            and (ev.final_diff_contribution or {}).get("source_paths")
            and (
                len((ev.state_evidence.get("same_file_chain") or {}).get("previous") or []) >= 2
                or len((ev.state_evidence.get("same_file_chain") or {}).get("later") or []) >= 2
            )
        ):
            score += 6
        if ev.observation_effect in {"failed", "no_effect"}:
            score += 2
        if (ev.trajectory_signals.get("same_file_edit_count") or 0) >= 3:
            score += 2
        if _static_has_review_signal(ev.static_evidence):
            score += 4
        if score > 0:
            scores.append((score, idx))
    scores.sort(key=lambda item: (-item[0], item[1]))
    selected = [idx for _score, idx in scores[:MAX_REVIEW_ACTIONS]]
    if len(selected) < MAX_REVIEW_ACTIONS:
        for ev in evidence:
            if ev.action_index not in selected:
                selected.append(ev.action_index)
            if len(selected) >= MAX_REVIEW_ACTIONS:
                break
    return selected


def _packet_action(
    action: dict[str, Any],
    ev: ActionEvidence,
    raw_by_index: dict[int, str],
    obs_by_index: dict[int, str],
) -> dict[str, Any]:
    idx = ev.action_index
    raw = _truncate_text(raw_by_index.get(idx) or action.get("raw_action") or ev.raw_action, MAX_RAW_ACTION_CHARS)
    obs = _truncate_text(obs_by_index.get(idx) or "", MAX_OBSERVATION_CHARS)
    return {
        "action_index": idx,
        "risk_scope": action.get("risk_scope"),
        "risk_scope_rationale": _truncate_text(action.get("risk_scope_rationale") or "", 500)["text"],
        "action_myopia_score": action.get("action_myopia_score"),
        "wrong_abstraction": action.get("wrong_abstraction"),
        "target_files": ev.target_files,
        "action_kind": ev.action_kind,
        "observation_effect": ev.observation_effect,
        "patch_survival": ev.patch_survival.as_dict(),
        "final_diff_contribution": ev.final_diff_contribution,
        "artifact_signals": ev.artifact_signals,
        "state_evidence": ev.state_evidence,
        "static_evidence": ev.static_evidence,
        "raw_action": raw["text"],
        "raw_action_truncated": raw["truncated"],
        "observation": obs["text"],
        "observation_truncated": obs["truncated"],
    }


def _wrong_abstraction_candidates(actions: list[dict[str, Any]], evidence: list[ActionEvidence]) -> list[dict[str, Any]]:
    by_idx = {int(a.get("action_index", -1)): a for a in actions if a.get("action_index") is not None}
    candidates = []
    for ev in evidence:
        hints = ev.static_evidence.get("wrong_abstraction_hints") or {}
        signals = hints.get("signals") or []
        if not signals and not (by_idx.get(ev.action_index, {}).get("wrong_abstraction") or {}).get("present"):
            continue
        candidates.append({
            "action_index": ev.action_index,
            "risk_scope": by_idx.get(ev.action_index, {}).get("risk_scope"),
            "target_files": ev.target_files,
            "signals": signals,
            "confidence": hints.get("confidence", 0.0),
            "old_payload_snippet": hints.get("old_payload_snippet", ""),
            "payload_snippet": hints.get("payload_snippet", ""),
            "observation_snippet": hints.get("observation_snippet", ""),
            "duplicate_same_payload_actions": hints.get("duplicate_same_payload_actions") or [],
            "future_validation_errors": hints.get("future_validation_errors") or [],
        })
    return sorted(candidates, key=lambda item: (-float(item.get("confidence") or 0.0), item["action_index"]))[:20]


def _truncate_text(value: Any, max_chars: int) -> dict[str, Any]:
    text = str(value or "")
    if len(text) <= max_chars:
        return {"text": text, "truncated": False}
    return {"text": text[:max_chars] + "\n...[truncated]", "truncated": True}


def _enforce_packet_budget(packet: dict[str, Any]) -> dict[str, Any]:
    """Progressively shrink lower-priority evidence until the packet fits budget."""
    packet["packet_truncated"] = False
    if _packet_chars(packet) <= MAX_PACKET_CHARS:
        return packet

    packet["packet_truncated"] = True
    for action_limit, text_limit, timeline_limit in ((30, 400, 24), (20, 300, 18), (12, 220, 12)):
        packet["top_suspicious_actions"] = (packet.get("top_suspicious_actions") or [])[:action_limit]
        packet["validation_error_timeline"] = (packet.get("validation_error_timeline") or [])[:timeline_limit]
        packet["same_file_chains"] = [_compact_action_indices(item) for item in (packet.get("same_file_chains") or [])[:timeline_limit]]
        packet["ownership_chains"] = [_compact_ownership_chain(item, text_limit) for item in (packet.get("ownership_chains") or [])[:timeline_limit]]
        packet["artifact_lifecycle"] = [_compact_action_indices(item) for item in (packet.get("artifact_lifecycle") or [])[:timeline_limit]]
        packet["wrong_abstraction_candidates"] = [
            _compact_wa_candidate(item, text_limit)
            for item in (packet.get("wrong_abstraction_candidates") or [])[:timeline_limit]
        ]
        packet["final_diff_summary"] = _compact_final_diff_summary(packet.get("final_diff_summary") or {}, action_limit)
        for action in packet["top_suspicious_actions"]:
            _shrink_packet_action(action, text_limit)
        if _packet_chars(packet) <= MAX_PACKET_CHARS:
            return packet

    packet["final_diff_summary"] = _compact_final_diff_summary(packet.get("final_diff_summary") or {}, 2)
    packet["same_file_chains"] = [_compact_action_indices(item, max_items=8) for item in (packet.get("same_file_chains") or [])[:8]]
    packet["ownership_chains"] = [_compact_ownership_chain(item, 60, max_candidates=3) for item in (packet.get("ownership_chains") or [])[:3]]
    packet["artifact_lifecycle"] = [_compact_action_indices(item, max_items=8) for item in (packet.get("artifact_lifecycle") or [])[:8]]
    packet["wrong_abstraction_candidates"] = [
        _compact_wa_candidate(item, 80)
        for item in (packet.get("wrong_abstraction_candidates") or [])[:3]
    ]
    packet["validation_error_timeline"] = (packet.get("validation_error_timeline") or [])[:3]
    packet["top_suspicious_actions"] = (packet.get("top_suspicious_actions") or [])[:2]
    packet["top_suspicious_actions"] = [
        _minimal_packet_action(action, 80)
        for action in packet["top_suspicious_actions"]
    ]
    return packet


def _packet_chars(packet: dict[str, Any]) -> int:
    return len(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))


def _shrink_packet_action(action: dict[str, Any], max_chars: int) -> None:
    raw = _truncate_text(action.get("raw_action") or "", max_chars)
    obs = _truncate_text(action.get("observation") or "", max_chars)
    action["raw_action"] = raw["text"]
    action["observation"] = obs["text"]
    action["raw_action_truncated"] = bool(action.get("raw_action_truncated") or raw["truncated"])
    action["observation_truncated"] = bool(action.get("observation_truncated") or obs["truncated"])
    for key in ("risk_scope_rationale",):
        if key in action:
            action[key] = _truncate_text(action[key], max(200, max_chars))["text"]
    action["target_files"] = _limit_list(action.get("target_files"), 12)
    action["final_diff_contribution"] = _compact_mapping(action.get("final_diff_contribution"), max_items=8)
    action["artifact_signals"] = _compact_mapping(action.get("artifact_signals"), max_items=8)
    action["state_evidence"] = _compact_mapping(action.get("state_evidence"), max_items=8)
    action["static_evidence"] = _compact_mapping(action.get("static_evidence"), max_items=8)


def _minimal_packet_action(action: dict[str, Any], max_chars: int) -> dict[str, Any]:
    return {
        "action_index": action.get("action_index"),
        "risk_scope": action.get("risk_scope"),
        "risk_scope_rationale": _truncate_text(action.get("risk_scope_rationale") or "", max_chars)["text"],
        "action_myopia_score": action.get("action_myopia_score"),
        "wrong_abstraction": action.get("wrong_abstraction"),
        "target_files": _limit_list(action.get("target_files"), 8),
        "action_kind": action.get("action_kind"),
        "observation_effect": action.get("observation_effect"),
        "patch_survival": action.get("patch_survival"),
        "final_diff_contribution": _compact_mapping(action.get("final_diff_contribution"), max_items=4),
        "artifact_signals": _compact_mapping(action.get("artifact_signals"), max_items=4),
        "wrong_abstraction_hints": _compact_mapping(
            (action.get("static_evidence") or {}).get("wrong_abstraction_hints") or {},
            max_items=4,
        ),
        "raw_action": _truncate_text(action.get("raw_action") or "", max_chars)["text"],
        "raw_action_truncated": True,
        "observation": _truncate_text(action.get("observation") or "", max_chars)["text"],
        "observation_truncated": True,
    }


def _compact_action_indices(item: dict[str, Any], max_items: int = 12) -> dict[str, Any]:
    out = dict(item)
    actions = out.get("actions")
    if isinstance(actions, list) and len(actions) > max_items:
        out["actions"] = actions[: max_items // 2] + ["..."] + actions[-max_items // 2 :]
        out["actions_truncated"] = True
    return out


def _compact_ownership_chain(item: dict[str, Any], max_chars: int, max_candidates: int = 10) -> dict[str, Any]:
    out = {
        "path": item.get("path"),
        "action_count": item.get("action_count"),
        "candidate_count": item.get("candidate_count"),
    }
    candidates = []
    for candidate in item.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        compact = {
            "action_index": candidate.get("action_index"),
            "risk_scope": candidate.get("risk_scope"),
            "operation": candidate.get("operation"),
            "patch_status": candidate.get("patch_status"),
            "observation_effect": candidate.get("observation_effect"),
            "semantic_survival": candidate.get("semantic_survival"),
            "final_present": candidate.get("final_present"),
            "independent": candidate.get("independent"),
            "exact_text_survived": candidate.get("exact_text_survived"),
            "symbol_hits": _limit_list(candidate.get("symbol_hits"), 3),
            "previous_same_file": _limit_list(candidate.get("previous_same_file"), 6),
            "later_same_file": _limit_list(candidate.get("later_same_file"), 6),
        }
        compact["raw_action"] = _truncate_text(candidate.get("raw_action") or "", max_chars)["text"]
        compact["observation"] = _truncate_text(candidate.get("observation") or "", max_chars)["text"]
        candidates.append(compact)
    out["candidates"] = candidates[:max_candidates]
    if len(candidates) > max_candidates:
        out["candidates_truncated"] = True
    return out


def _compact_wa_candidate(item: dict[str, Any], max_chars: int) -> dict[str, Any]:
    out = dict(item)
    out["old_payload_snippet"] = _truncate_text(out.get("old_payload_snippet") or "", max_chars)["text"]
    out["payload_snippet"] = _truncate_text(out.get("payload_snippet") or "", max_chars)["text"]
    out["observation_snippet"] = _truncate_text(out.get("observation_snippet") or "", max_chars)["text"]
    errors = []
    for err in out.get("future_validation_errors") or []:
        if isinstance(err, dict):
            errors.append({
                "action_index": err.get("action_index"),
                "observation": _truncate_text(err.get("observation") or "", max_chars)["text"],
            })
    out["future_validation_errors"] = errors[:2]
    return out


def _compact_mapping(value: Any, max_items: int = 8) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, list):
            out[key] = _limit_list(item, max_items)
        elif isinstance(item, dict):
            out[key] = _compact_mapping(item, max_items=max_items)
        elif isinstance(item, str):
            out[key] = _truncate_text(item, 240)["text"]
        else:
            out[key] = item
    return out


def _limit_list(value: Any, max_items: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    if len(value) <= max_items:
        return value
    return value[: max_items // 2] + ["..."] + value[-max_items // 2 :]


def _compact_final_diff_summary(summary: dict[str, Any], max_files: int) -> dict[str, Any]:
    if not summary.get("available"):
        return summary
    snippets = []
    for item in (summary.get("snippets") or [])[:max_files]:
        snippets.append({
            "path": item.get("path"),
            "snippets": [str(s)[:160] for s in (item.get("snippets") or [])[:2]],
        })
    files = list(summary.get("files") or [])
    return {
        "available": True,
        "files": files[: max_files * 2],
        "files_truncated": len(files) > max_files * 2,
        "snippets": snippets,
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("model", "sample_id", "instance_id", "source_traj")
    return {k: candidate.get(k) for k in keys if candidate.get(k) is not None}


def _same_file_chains(evidence: list[ActionEvidence]) -> list[dict[str, Any]]:
    path_to_actions: dict[str, list[int]] = defaultdict(list)
    for ev in evidence:
        for path in ev.target_files:
            path_to_actions[path].append(ev.action_index)
    chains = [
        {"path": path, "actions": indices}
        for path, indices in sorted(path_to_actions.items())
        if len(indices) >= 2
    ]
    return sorted(chains, key=lambda item: (-len(item["actions"]), item["path"]))[:20]


def _ownership_chains(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    raw_by_index: dict[int, str],
    obs_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    """Build same-file ownership evidence for source/test/config edit chains."""
    action_by_index = {int(a.get("action_index", -1)): a for a in actions if a.get("action_index") is not None}
    path_to_evidence: dict[str, list[ActionEvidence]] = defaultdict(list)
    for ev in evidence:
        final = ev.final_diff_contribution or {}
        for path in final.get("source_paths") or []:
            path_to_evidence[path].append(ev)

    chains: list[dict[str, Any]] = []
    for path, items in path_to_evidence.items():
        items = sorted(items, key=lambda ev: ev.action_index)
        if len(items) < 2:
            continue
        candidates = []
        for ev in items:
            action = action_by_index.get(ev.action_index, {})
            same_file = ev.state_evidence.get("same_file_chain") or {}
            raw = _truncate_text(raw_by_index.get(ev.action_index) or ev.raw_action, 500)
            obs = _truncate_text(obs_by_index.get(ev.action_index) or "", 500)
            final = ev.final_diff_contribution or {}
            if (
                action.get("risk_scope") not in {"substantive", "uncertain", "noise_reverted", "noise_no_effect"}
                and final.get("semantic_survival") not in {"exact_text_survived", "later_refined", "symbol_or_hunk_survived", "fully_removed", "superseded_uncertain"}
            ):
                continue
            candidates.append({
                "action_index": ev.action_index,
                "risk_scope": action.get("risk_scope"),
                "risk_scope_rationale": action.get("risk_scope_rationale") or "",
                "operation": ev.raw_operation,
                "patch_status": ev.patch_survival.status,
                "observation_effect": ev.observation_effect,
                "semantic_survival": final.get("semantic_survival"),
                "final_present": final.get("present"),
                "independent": final.get("independent"),
                "exact_text_survived": final.get("exact_text_survived"),
                "symbol_hits": final.get("symbol_hits") or [],
                "previous_same_file": same_file.get("previous") or [],
                "later_same_file": same_file.get("later") or [],
                "raw_action": raw["text"],
                "raw_action_truncated": raw["truncated"],
                "observation": obs["text"],
                "observation_truncated": obs["truncated"],
            })
        if len(candidates) < 2:
            continue
        risky = [
            c for c in candidates
            if c.get("risk_scope") in {"substantive", "noise_reverted", "uncertain"}
            or c.get("semantic_survival") in {"fully_removed", "superseded_uncertain", "later_refined"}
            or len(c.get("later_same_file") or []) >= 2
        ]
        if not risky:
            continue
        chains.append({
            "path": path,
            "action_count": len(items),
            "candidate_count": len(candidates),
            "candidates": candidates,
        })
    return sorted(chains, key=lambda item: (-item["action_count"], item["path"]))[:12]


def _artifact_lifecycle(evidence: list[ActionEvidence]) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for ev in evidence:
        paths = set(ev.artifact_signals.get("introduced_artifact_paths") or [])
        paths.update(ev.artifact_signals.get("residual_artifact_paths") or [])
        for item in ev.state_evidence.get("artifact_lifecycle") or []:
            if isinstance(item, dict) and item.get("path"):
                paths.add(str(item["path"]))
        for path in paths:
            entry = by_path.setdefault(path, {"path": path, "actions": [], "residual": False})
            entry["actions"].append(ev.action_index)
            entry["residual"] = bool(entry["residual"] or path in (ev.artifact_signals.get("residual_artifact_paths") or []))
    return sorted(by_path.values(), key=lambda item: item["path"])[:30]


def _validation_error_timeline(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    obs_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    out = []
    by_idx = {ev.action_index: ev for ev in evidence}
    for action in actions:
        idx = int(action.get("action_index", -1))
        ev = by_idx.get(idx)
        if ev is None:
            continue
        obs = obs_by_index.get(idx, "")
        if ev.action_kind in {"test", "install"} or ev.observation_effect in {"failed", "no_effect"}:
            item = _truncate_text(obs, 500)
            out.append({
                "action_index": idx,
                "action_kind": ev.action_kind,
                "observation_effect": ev.observation_effect,
                "observation": item["text"],
                "observation_truncated": item["truncated"],
            })
    return out[:30]


def _final_diff_summary(final_patch: str) -> dict[str, Any]:
    if not final_patch:
        return {"available": False, "files": [], "snippets": []}
    files: list[str] = []
    snippets_by_file: dict[str, list[str]] = defaultdict(list)
    current_file = ""
    for line in final_patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3][2:] if parts[3].startswith("b/") else parts[3]
                files.append(current_file)
        elif current_file and line.startswith("@@"):
            snippets_by_file[current_file].append(line[:240])
        elif current_file and (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---")):
            snippets = snippets_by_file[current_file]
            if len(snippets) < MAX_FINAL_DIFF_SNIPPETS_PER_FILE:
                snippets.append(line[:240])
    return {
        "available": True,
        "files": list(dict.fromkeys(files))[:50],
        "snippets": [
            {"path": path, "snippets": snippets[:MAX_FINAL_DIFF_SNIPPETS_PER_FILE]}
            for path, snippets in list(snippets_by_file.items())[:30]
        ],
    }


def _parse_scope_overrides(raw: Any, failures: list[str]) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        failures.append("scope_overrides")
        return []
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.append(f"scope_overrides[{i}]")
            continue
        idx = _int_value(item.get("action_index"))
        scope = item.get("risk_scope")
        confidence = _score(item.get("confidence"))
        if idx is None:
            failures.append(f"scope_overrides[{i}].action_index")
        if scope not in VALID_RISK_SCOPES:
            failures.append(f"scope_overrides[{i}].risk_scope")
        if confidence is None:
            failures.append(f"scope_overrides[{i}].confidence")
        if idx is not None and scope in VALID_RISK_SCOPES and confidence is not None:
            out.append({
                "action_index": idx,
                "risk_scope": scope,
                "confidence": confidence,
                "rationale": str(item.get("rationale") or ""),
            })
    return out


def _parse_wrong_abstraction_overrides(raw: Any, failures: list[str]) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        failures.append("wrong_abstraction_overrides")
        return []
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            failures.append(f"wrong_abstraction_overrides[{i}]")
            continue
        idx = _int_value(item.get("action_index"))
        present = item.get("present")
        severity = _score(item.get("severity"))
        confidence = _score(item.get("confidence"))
        if idx is None:
            failures.append(f"wrong_abstraction_overrides[{i}].action_index")
        if not isinstance(present, bool):
            failures.append(f"wrong_abstraction_overrides[{i}].present")
        if severity is None:
            failures.append(f"wrong_abstraction_overrides[{i}].severity")
        if confidence is None:
            failures.append(f"wrong_abstraction_overrides[{i}].confidence")
        if idx is not None and isinstance(present, bool) and severity is not None and confidence is not None:
            out.append({
                "action_index": idx,
                "present": present,
                "severity": severity,
                "confidence": confidence,
                "rationale": str(item.get("rationale") or ""),
            })
    return out


def _parse_review_penalties(raw: Any, failures: list[str]) -> dict[str, dict[str, Any]]:
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        failures.append("trajectory_penalties")
        return {}
    out = {}
    for name in ("broad_rewrite", "artifact_residue"):
        item = raw.get(name)
        if item in (None, ""):
            continue
        if not isinstance(item, dict):
            failures.append(name)
            continue
        present = item.get("present")
        severity = _score(item.get("severity"))
        if not isinstance(present, bool):
            failures.append(f"{name}.present")
            continue
        if severity is None:
            failures.append(f"{name}.severity")
            continue
        penalty = {
            "present": present,
            "severity": severity,
            "evidence_actions": _int_list(item.get("evidence_actions")),
            "rationale": str(item.get("rationale") or ""),
        }
        if name == "broad_rewrite":
            penalty["affected_files"] = _string_list(item.get("affected_files"))
            penalty["evidence_scopes"] = ["trajectory_review"] if present else []
        else:
            penalty["paths"] = _string_list(item.get("paths"))
            penalty["artifact_types"] = ["artifact_file"] if present else []
        out[name] = penalty
    return out


def _merge_review_penalties(base: dict[str, Any], review: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base))
    for name, penalty in review.items():
        existing = merged.setdefault(name, {})
        existing.update(penalty)
    return merged


def _static_has_review_signal(static: dict[str, Any]) -> bool:
    if static.get("syntax_error") or static.get("wrong_scope_insertion"):
        return True
    artifact = static.get("artifact_scan") or {}
    if artifact.get("introduced") or artifact.get("still_present") or artifact.get("deps_introduced"):
        return True
    breadth = static.get("breadth_metrics") or {}
    return int(breadth.get("cumulative_files_changed") or 0) >= 8


def _score(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(max(0.0, min(1.0, f)), 3)


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        parsed = _int_value(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item not in (None, "")]
