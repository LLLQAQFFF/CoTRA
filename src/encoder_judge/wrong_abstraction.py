"""Dedicated wrong_abstraction review for the encoder judge.

This stage is intentionally narrow: it reviews only candidate scorable actions
and only patches the wrong_abstraction field plus derived myopia fields.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge import config, derive
from llm_judge.client import JudgeClient
from llm_judge.cost import CostTracker
from llm_judge.parser import _extract_json

from encoder_judge.evidence import ActionEvidence, extract_old_payload, extract_payload, load_sidecar


SCORABLE_SCOPES = {"substantive", "uncertain"}
VALID_EVIDENCE_TYPES = {
    "wrong_scope",
    "wrong_layer",
    "wrong_api_boundary",
    "wrong_schema_boundary",
    "misplaced_helper",
    "not_wrong_abstraction",
}

WA_PRESENT_CONFIDENCE_THRESHOLD = 0.70
WA_CLEAR_CONFIDENCE_THRESHOLD = 0.65
WA_PRESENT_SEVERITY_THRESHOLD = 0.30

MAX_WA_CANDIDATES = 24
MAX_WA_RAW_ACTION_CHARS = 800
MAX_WA_SNIPPET_CHARS = 1200
MAX_WA_PACKET_CHARS = 120000


@dataclass
class WAReviewResult:
    judgments: list[dict[str, Any]] = field(default_factory=list)
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


def build_wa_candidate_packet(
    *,
    template: dict[str, Any],
    template_path: Path,
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    task_description: str,
    max_candidates: int = MAX_WA_CANDIDATES,
) -> dict[str, Any]:
    """Build a cost-bounded evidence packet for WA-only review."""
    traj = load_sidecar(template_path, ".traj.json")
    candidate = load_sidecar(template_path, ".candidate.json")
    raw_by_index = {
        int(item.get("step_index")): str(item.get("action") or item.get("raw_action") or "")
        for item in (traj.get("trajectory") or [])
        if item.get("step_index") is not None
    }
    obs_by_index = {
        int(item.get("step_index")): str(item.get("observation") or "")
        for item in (traj.get("trajectory") or [])
        if item.get("step_index") is not None
    }

    action_by_index = {
        int(action.get("action_index", -1)): action
        for action in actions
        if action.get("action_index") is not None
    }
    selected = _select_candidates(actions, evidence, max_candidates=max_candidates)
    candidates = [
        _candidate_item(action_by_index[ev.action_index], ev, raw_by_index, obs_by_index)
        for ev in selected
        if ev.action_index in action_by_index
    ]
    packet = {
        "schema": "wrong-abstraction-candidate-review-v1",
        "truncation_notice": (
            "[truncated] means content was shortened only for cost control. It is not evidence "
            "that the command failed, output stopped, or the action had no effect."
        ),
        "task_description": _truncate_text(task_description, 1000)["text"],
        "candidate_summary": _candidate_summary(candidate),
        "scope_summary": dict(Counter(action.get("risk_scope") or "unscored" for action in actions)),
        "review_question": (
            "For each candidate, decide only whether the action placed implementation in the "
            "wrong structural location, abstraction layer, module boundary, API boundary, or "
            "schema/contract boundary. If present, state the better boundary."
        ),
        "candidates": candidates,
    }
    return _enforce_packet_budget(packet)


def should_run_wa_review(packet: dict[str, Any]) -> tuple[bool, str]:
    if packet.get("candidates"):
        return True, ""
    return False, "no_wrong_abstraction_candidates"


def review_wrong_abstraction_candidates(
    *,
    template: dict[str, Any],
    template_path: Path,
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    task_description: str,
    client: JudgeClient,
    model: str,
    tracker: CostTracker | None = None,
    max_candidates: int = MAX_WA_CANDIDATES,
) -> tuple[WAReviewResult, dict[str, Any]]:
    packet = build_wa_candidate_packet(
        template=template,
        template_path=template_path,
        actions=actions,
        evidence=evidence,
        task_description=task_description,
        max_candidates=max_candidates,
    )
    user = build_wa_review_prompt(packet)
    call = client.call(
        model=model,
        system="Return valid JSON only. Do not include markdown or extra commentary.",
        user=user,
        max_tokens=config.MAX_TOKENS_TRAJECTORY,
    )
    if tracker is not None:
        tracker.record(call)
    result = parse_wa_review(call.text)
    return result, {
        "packet_action_count": len(packet.get("candidates") or []),
        "packet_truncated": bool(packet.get("packet_truncated")),
        "packet_chars": len(user),
        "response_chars": len(result.raw),
    }


def build_wa_review_prompt(packet: dict[str, Any]) -> str:
    return f"""Dedicated human-target v2 wrong_abstraction review. Output JSON only.

Use a narrow definition. wrong_abstraction is present only when the action puts implementation in the wrong
structural location, abstraction layer, module boundary, dependency direction, API boundary, or schema/contract boundary.

Do not mark wrong_abstraction for incomplete implementation, failing tests, low task progress, broad rewrite, artifact
residue, or ordinary code quality issues. Those are handled by other fields.

Truncation is not evidence. [truncated] only means cost-control truncation; do not treat it as failure/no-effect evidence.

Return one JSON object exactly in this shape:
{{
  "wrong_abstraction_judgments": [
    {{
      "action_index": 12,
      "present": true,
      "severity": 0.6,
      "confidence": 0.82,
      "evidence_type": "wrong_scope|wrong_layer|wrong_api_boundary|wrong_schema_boundary|misplaced_helper|not_wrong_abstraction",
      "correct_boundary": "short description of where this logic should live",
      "rationale": "short evidence-based reason"
    }}
  ]
}}

{json.dumps(packet, ensure_ascii=False, indent=2)}
"""


def parse_wa_review(text: str) -> WAReviewResult:
    data = _extract_json(text)
    if data is None:
        return WAReviewResult(parse_failed=True, parse_failures=["json"], raw=text)
    raw_items = data.get("wrong_abstraction_judgments")
    failures: list[str] = []
    if raw_items in (None, ""):
        return WAReviewResult(raw=text)
    if not isinstance(raw_items, list):
        return WAReviewResult(parse_failed=True, parse_failures=["wrong_abstraction_judgments"], raw=text)

    judgments: list[dict[str, Any]] = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            failures.append(f"wrong_abstraction_judgments[{i}]")
            continue
        idx = _int_value(item.get("action_index"))
        present = item.get("present")
        severity = _score(item.get("severity"))
        confidence = _score(item.get("confidence"))
        evidence_type = item.get("evidence_type")
        if idx is None:
            failures.append(f"wrong_abstraction_judgments[{i}].action_index")
        if not isinstance(present, bool):
            failures.append(f"wrong_abstraction_judgments[{i}].present")
        if severity is None:
            failures.append(f"wrong_abstraction_judgments[{i}].severity")
        if confidence is None:
            failures.append(f"wrong_abstraction_judgments[{i}].confidence")
        if evidence_type not in VALID_EVIDENCE_TYPES:
            failures.append(f"wrong_abstraction_judgments[{i}].evidence_type")
        if (
            idx is not None
            and isinstance(present, bool)
            and severity is not None
            and confidence is not None
            and evidence_type in VALID_EVIDENCE_TYPES
        ):
            judgments.append({
                "action_index": idx,
                "present": present,
                "severity": severity,
                "confidence": confidence,
                "evidence_type": evidence_type,
                "correct_boundary": str(item.get("correct_boundary") or ""),
                "rationale": str(item.get("rationale") or ""),
            })
    return WAReviewResult(
        judgments=judgments,
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def apply_wa_review(actions: list[dict[str, Any]], result: WAReviewResult) -> dict[str, Any]:
    """Apply WA judgments without changing scope, scalar fields, or trajectory penalties."""
    by_index = {
        int(action.get("action_index", -1)): action
        for action in actions
        if action.get("action_index") is not None
    }
    meta = {
        "present_overrides_applied": 0,
        "clear_overrides_applied": 0,
        "ignored": 0,
    }
    for judgment in result.judgments:
        action = by_index.get(judgment["action_index"])
        if action is None:
            meta["ignored"] += 1
            continue
        original = dict(action.get("wrong_abstraction") or {})
        review_meta = {
            "applied": False,
            "original": original,
            "judgment": judgment,
        }
        if action.get("risk_scope") not in SCORABLE_SCOPES:
            review_meta["reason"] = "non_scorable_scope"
            action["_encoder_wrong_abstraction_review"] = review_meta
            meta["ignored"] += 1
            continue
        if judgment["present"]:
            if (
                judgment["confidence"] >= WA_PRESENT_CONFIDENCE_THRESHOLD
                and judgment["severity"] >= WA_PRESENT_SEVERITY_THRESHOLD
            ):
                action["wrong_abstraction"] = {
                    "present": True,
                    "severity": judgment["severity"],
                    "rationale": _format_rationale(judgment),
                }
                review_meta["applied"] = True
                meta["present_overrides_applied"] += 1
            else:
                review_meta["reason"] = "below_present_threshold"
                meta["ignored"] += 1
        else:
            if judgment["confidence"] >= WA_CLEAR_CONFIDENCE_THRESHOLD and not _protected_positive(action):
                action["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
                review_meta["applied"] = True
                meta["clear_overrides_applied"] += 1
            else:
                review_meta["reason"] = "below_clear_threshold_or_protected_positive"
                meta["ignored"] += 1
        if review_meta["applied"]:
            action["action_myopia_score"] = derive.derive_action_myopia_score(action)
            action["is_myopic"] = derive.derive_is_myopic(action)
        action["_encoder_wrong_abstraction_review"] = review_meta
    return meta


def _select_candidates(
    actions: list[dict[str, Any]],
    evidence: list[ActionEvidence],
    *,
    max_candidates: int,
) -> list[ActionEvidence]:
    action_by_index = {
        int(action.get("action_index", -1)): action
        for action in actions
        if action.get("action_index") is not None
    }
    scored: list[tuple[int, int, ActionEvidence]] = []
    for ev in evidence:
        action = action_by_index.get(ev.action_index)
        if not action or action.get("risk_scope") not in SCORABLE_SCOPES:
            continue
        score = _candidate_score(action, ev)
        if score > 0:
            scored.append((score, ev.action_index, ev))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [ev for _score, _idx, ev in scored[:max_candidates]]


def _candidate_score(action: dict[str, Any], ev: ActionEvidence) -> int:
    hints = ev.static_evidence.get("wrong_abstraction_hints") or {}
    hint_conf = float(hints.get("confidence") or 0.0)
    signals = set(hints.get("signals") or [])
    payload = extract_payload({"raw_action": action.get("raw_action") or ev.raw_action})
    old_payload = extract_old_payload({"raw_action": action.get("raw_action") or ev.raw_action})
    score = 0
    if (action.get("wrong_abstraction") or {}).get("present"):
        score += 8
    if hint_conf >= 0.4:
        score += int(hint_conf * 10)
    if signals:
        score += 2
    if "future_validation_error" in signals:
        score += 5
    if ev.final_diff_contribution.get("present"):
        score += 2
    if (ev.trajectory_signals.get("same_file_edit_count") or 0) >= 2:
        score += 2
    if _payload_kind(payload) != "ordinary_edit":
        score += 4
    if _boundary_change_hints(old_payload, payload, ev.target_files):
        score += 4
    return score


def _protected_positive(action: dict[str, Any]) -> bool:
    """Avoid clearing an existing WA positive when hard structural hints exist."""
    wa = action.get("wrong_abstraction") or {}
    if not wa.get("present"):
        return False
    evidence = action.get("_encoder_evidence") or {}
    static = evidence.get("static_evidence") or {}
    hints = static.get("wrong_abstraction_hints") or {}
    signals = set(hints.get("signals") or [])
    confidence = float(hints.get("confidence") or 0.0)
    hard_signals = {
        "structural_insert",
        "interface_boundary_change",
        "future_validation_error",
        "observation_structural_misplacement",
    }
    return confidence >= 0.4 and bool(signals & hard_signals)


def _candidate_item(
    action: dict[str, Any],
    ev: ActionEvidence,
    raw_by_index: dict[int, str],
    obs_by_index: dict[int, str],
) -> dict[str, Any]:
    raw_action = raw_by_index.get(ev.action_index) or action.get("raw_action") or ev.raw_action
    payload = extract_payload({"raw_action": raw_action})
    old_payload = extract_old_payload({"raw_action": raw_action})
    hints = ev.static_evidence.get("wrong_abstraction_hints") or {}
    path_role = _path_role(ev.target_files)
    inferred_layer = _inferred_layer(ev.target_files, payload)
    raw = _truncate_text(raw_action, MAX_WA_RAW_ACTION_CHARS)
    obs = _truncate_text(obs_by_index.get(ev.action_index) or "", MAX_WA_SNIPPET_CHARS)
    local_context = _truncate_text(
        hints.get("observation_snippet") or obs_by_index.get(ev.action_index) or "",
        MAX_WA_SNIPPET_CHARS,
    )
    payload_text = _truncate_text(payload or hints.get("payload_snippet") or "", MAX_WA_SNIPPET_CHARS)
    old_payload_text = _truncate_text(old_payload or hints.get("old_payload_snippet") or "", MAX_WA_SNIPPET_CHARS)
    return {
        "action_index": ev.action_index,
        "risk_scope": action.get("risk_scope"),
        "target_files": ev.target_files,
        "path_role": path_role,
        "inferred_layer": inferred_layer,
        "changed_symbol": _changed_symbol(payload) or _changed_symbol(old_payload),
        "local_context": local_context["text"],
        "local_context_truncated": local_context["truncated"],
        "payload_kind": _payload_kind(payload),
        "boundary_change_hints": _boundary_change_hints(old_payload, payload, ev.target_files),
        "same_file_fixup_chain": ev.state_evidence.get("same_file_chain") or {},
        "future_error_refs": hints.get("future_validation_errors") or [],
        "final_diff_contribution": ev.final_diff_contribution,
        "patch_survival": ev.patch_survival.as_dict(),
        "current_wrong_abstraction": action.get("wrong_abstraction") or {},
        "static_wrong_abstraction_hints": _compact_static_wa_hints(hints, MAX_WA_SNIPPET_CHARS),
        "raw_action": raw["text"],
        "raw_action_truncated": raw["truncated"],
        "observation": obs["text"],
        "observation_truncated": obs["truncated"],
        "edit_payload": payload_text["text"],
        "edit_payload_truncated": payload_text["truncated"],
        "old_payload": old_payload_text["text"],
        "old_payload_truncated": old_payload_text["truncated"],
    }


def _payload_kind(payload: str) -> str:
    text = payload.strip()
    if not text:
        return "empty"
    if re.search(r"^\s*(class|interface|type|struct|enum)\s+\w+", text, re.MULTILINE):
        return "type_or_class_definition"
    if re.search(r"^\s*(def|function|func)\s+\w+|^\s*(export\s+)?(async\s+)?function\s+\w+", text, re.MULTILINE):
        return "function_definition"
    if re.search(r"^\s*(import|from\s+\S+\s+import|export)\b", text, re.MULTILINE):
        return "import_or_export"
    if re.search(r"\b(api|schema|contract|route|handler|controller|service|repository|model|provider|adapter)\b", text, re.IGNORECASE):
        return "boundary_related_edit"
    if re.search(r"\b(Promise<|async\s+|await\s+|Observable<|Future<|Task<)\b", text):
        return "async_interface_edit"
    return "ordinary_edit"


def _boundary_change_hints(old_payload: str, payload: str, paths: list[str]) -> list[str]:
    combined = f"{old_payload}\n{payload}"
    hints: list[str] = []
    if re.search(r"\b(Promise<|async\s+|await\s+|Future<|Task<|Observable<)\b", combined):
        hints.append("async_or_promise_boundary")
    if re.search(r"\b(schema|contract|dto|request|response|query|config|setting)\b", combined, re.IGNORECASE):
        hints.append("schema_or_config_contract")
    if re.search(r"\b(api|route|handler|controller|service|repository|model|provider|adapter|client)\b", combined, re.IGNORECASE):
        hints.append("api_or_layer_boundary")
    path_text = "/".join(paths).lower()
    if re.search(r"/(api|routes?|controllers?|services?|repositories?|models?|schemas?|config|providers?|adapters?)/", path_text):
        hints.append("path_suggests_architectural_layer")
    return list(dict.fromkeys(hints))


def _path_role(paths: list[str]) -> str:
    if not paths:
        return "unknown"
    text = "/".join(paths).lower()
    if any(part in text for part in ("/test/", "/tests/", "/spec/", "/specs/")):
        return "test"
    if any(part in text for part in ("/docs/", ".md")):
        return "documentation"
    if any(part in text for part in ("/config/", ".json", ".yaml", ".yml", ".toml")):
        return "config_or_schema"
    return "source"


def _inferred_layer(paths: list[str], payload: str) -> str:
    text = f"{'/'.join(paths)}\n{payload}".lower()
    layer_terms = [
        ("controller", "controller_or_route"),
        ("route", "controller_or_route"),
        ("handler", "controller_or_route"),
        ("service", "service"),
        ("repository", "repository_or_storage"),
        ("store", "repository_or_storage"),
        ("model", "model_or_domain"),
        ("schema", "schema_or_contract"),
        ("contract", "schema_or_contract"),
        ("provider", "provider_or_adapter"),
        ("adapter", "provider_or_adapter"),
        ("client", "client_or_integration"),
        ("config", "config"),
    ]
    for term, layer in layer_terms:
        if term in text:
            return layer
    return "unknown"


def _changed_symbol(payload: str) -> str:
    patterns = [
        r"^\s*class\s+([A-Za-z_][\w]*)",
        r"^\s*interface\s+([A-Za-z_][\w]*)",
        r"^\s*type\s+([A-Za-z_][\w]*)",
        r"^\s*struct\s+([A-Za-z_][\w]*)",
        r"^\s*def\s+([A-Za-z_][\w]*)",
        r"^\s*func\s+([A-Za-z_][\w]*)",
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][\w]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, payload, re.MULTILINE)
        if match:
            return match.group(1)
    return ""


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = ("model", "sample_id", "instance_id", "source_traj")
    return {key: candidate.get(key) for key in keys if candidate.get(key) is not None}


def _format_rationale(judgment: dict[str, Any]) -> str:
    parts = [judgment.get("rationale") or ""]
    if judgment.get("correct_boundary"):
        parts.append(f"Correct boundary: {judgment['correct_boundary']}")
    if judgment.get("evidence_type"):
        parts.append(f"Evidence type: {judgment['evidence_type']}")
    return " ".join(part for part in parts if part).strip()


def _truncate_text(value: Any, max_chars: int) -> dict[str, Any]:
    text = str(value or "")
    if len(text) <= max_chars:
        return {"text": text, "truncated": False}
    return {"text": text[:max_chars] + "\n...[truncated]", "truncated": True}


def _enforce_packet_budget(packet: dict[str, Any]) -> dict[str, Any]:
    packet["packet_truncated"] = False
    if _packet_chars(packet) <= MAX_WA_PACKET_CHARS:
        return packet
    packet["packet_truncated"] = True
    max_candidates = len(packet.get("candidates") or [])
    for action_limit, snippet_limit in (
        (max_candidates, 600),
        (max_candidates, 360),
        (min(max_candidates, 16), 240),
        (min(max_candidates, 12), 180),
        (min(max_candidates, 8), 140),
    ):
        packet["candidates"] = (packet.get("candidates") or [])[:action_limit]
        for item in packet["candidates"]:
            _shrink_candidate(item, snippet_limit)
        if _packet_chars(packet) <= MAX_WA_PACKET_CHARS:
            return packet
    packet["candidates"] = [_minimal_candidate(item) for item in (packet.get("candidates") or [])[:3]]
    return packet


def _packet_chars(packet: dict[str, Any]) -> int:
    return len(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))


def _shrink_candidate(item: dict[str, Any], max_chars: int) -> None:
    for key in ("raw_action", "observation", "local_context", "edit_payload", "old_payload"):
        truncated = _truncate_text(item.get(key) or "", max_chars)
        item[key] = truncated["text"]
        item[f"{key}_truncated"] = bool(item.get(f"{key}_truncated") or truncated["truncated"])
    errors = []
    for err in item.get("future_error_refs") or []:
        if isinstance(err, dict):
            errors.append({
                "action_index": err.get("action_index"),
                "observation": _truncate_text(err.get("observation") or "", max_chars)["text"],
            })
    item["future_error_refs"] = errors[:2]
    item["static_wrong_abstraction_hints"] = _compact_static_wa_hints(
        item.get("static_wrong_abstraction_hints") or {},
        max_chars,
    )


def _minimal_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_index": item.get("action_index"),
        "risk_scope": item.get("risk_scope"),
        "target_files": item.get("target_files"),
        "path_role": item.get("path_role"),
        "inferred_layer": item.get("inferred_layer"),
        "changed_symbol": item.get("changed_symbol"),
        "local_context": _truncate_text(item.get("local_context") or "", 180)["text"],
        "local_context_truncated": True,
        "payload_kind": item.get("payload_kind"),
        "boundary_change_hints": item.get("boundary_change_hints"),
        "same_file_fixup_chain": item.get("same_file_fixup_chain"),
        "current_wrong_abstraction": item.get("current_wrong_abstraction"),
        "static_wrong_abstraction_hints": item.get("static_wrong_abstraction_hints"),
        "edit_payload": _truncate_text(item.get("edit_payload") or "", 180)["text"],
        "edit_payload_truncated": True,
    }


def _compact_static_wa_hints(hints: dict[str, Any], max_chars: int) -> dict[str, Any]:
    if not isinstance(hints, dict):
        return {}
    out = {
        "signals": hints.get("signals") or [],
        "confidence": hints.get("confidence", 0.0),
        "structural_insert": bool(hints.get("structural_insert")),
        "duplicate_same_payload_actions": hints.get("duplicate_same_payload_actions") or [],
        "observation_snippet": _truncate_text(hints.get("observation_snippet") or "", min(max_chars, 240))["text"],
    }
    errors = []
    for err in hints.get("future_validation_errors") or []:
        if isinstance(err, dict):
            errors.append({
                "action_index": err.get("action_index"),
                "observation": _truncate_text(err.get("observation") or "", max_chars)["text"],
            })
    out["future_validation_errors"] = errors[:2]
    return out


def _score(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return max(0.0, min(1.0, f))


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
