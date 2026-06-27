"""Narrow semantic GLM judgment for scorable encoder actions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from llm_judge import config, derive
from llm_judge.client import JudgeClient
from llm_judge.cost import CostTracker
from llm_judge.parser import _extract_json

from encoder_judge.evidence import ActionEvidence


LEVEL_SCORES = {
    "none": 0.0,
    "low": 0.3,
    "medium": 0.6,
    "high": 0.9,
}
OBSERVABILITY_SCORES = {
    "none": 0.0,
    "weak": 0.3,
    "clear": 0.6,
    "severe": 0.9,
}
SURFACE_SCORES = {
    "local": 0.2,
    "module": 0.5,
    "cross_module": 0.7,
    "system": 0.9,
}


@dataclass
class SemanticResult:
    manual_risk_vector: dict[str, Any]
    wrong_abstraction: dict[str, Any]
    relates_to_target: bool | None
    action_role: str
    actual_effect: str
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


def score_action_semantics(
    *,
    action: dict[str, Any],
    evidence: ActionEvidence,
    task_description: str,
    trajectory_summary: str,
    client: JudgeClient,
    model: str,
    tracker: CostTracker | None = None,
) -> dict[str, Any]:
    """Call GLM for bounded semantic labels and map them to v2 scalar fields."""
    user = build_user_prompt(
        task_description=task_description,
        action=action,
        evidence=evidence,
        trajectory_summary=trajectory_summary,
    )
    call = client.call(
        model=model,
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=config.MAX_TOKENS_ACTION,
    )
    if tracker is not None:
        tracker.record(call)
    parsed = parse_semantic_result(call.text)
    out = dict(action)
    out["manual_risk_vector"] = parsed.manual_risk_vector
    out["wrong_abstraction"] = parsed.wrong_abstraction
    out["relates_to_target"] = parsed.relates_to_target
    out["action_role"] = parsed.action_role
    out["actual_effect"] = parsed.actual_effect
    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_encoder_semantic"] = {
        "model": model,
        "parse_failed": parsed.parse_failed,
        "parse_failures": parsed.parse_failures,
        "raw": parsed.raw,
    }
    return out


def parse_semantic_result(text: str) -> SemanticResult:
    data = _extract_json(text)
    if data is None:
        return _fallback_result(["json"], text)

    failures: list[str] = []
    task_relevance = _level(data.get("task_relevance"), LEVEL_SCORES, "task_relevance", failures)
    debt = _level(data.get("debt_level"), LEVEL_SCORES, "debt_level", failures)
    fragility = _level(data.get("fragility_level"), LEVEL_SCORES, "fragility_level", failures)
    regression = _level(data.get("regression_surface_level"), SURFACE_SCORES, "regression_surface_level", failures)
    observability = _level(
        data.get("observability_loss_evidence"), OBSERVABILITY_SCORES, "observability_loss_evidence", failures
    )
    relates = _tri_bool(data.get("relates_to_target"), failures)

    wa_raw = data.get("wrong_abstraction") if isinstance(data.get("wrong_abstraction"), dict) else {}
    if not isinstance(data.get("wrong_abstraction"), dict):
        failures.append("wrong_abstraction")
    wa_present = wa_raw.get("present")
    if not isinstance(wa_present, bool):
        failures.append("wrong_abstraction.present")
        wa_present = False
    wa_severity = _level(wa_raw.get("severity_level"), LEVEL_SCORES, "wrong_abstraction.severity_level", failures)
    if not wa_present:
        wa_severity = 0.0

    rv = {
        "task_advancement": task_relevance,
        "debt_density": debt,
        "fragility_delta": fragility,
        "regression_surface": regression,
        "observability_loss": observability,
        "rationale": str(data.get("rationale") or ""),
        "annotator_confidence": _confidence(data.get("confidence", data.get("annotator_confidence"))),
    }
    return SemanticResult(
        manual_risk_vector=rv,
        wrong_abstraction={
            "present": bool(wa_present) and wa_severity >= 0.3,
            "severity": wa_severity if wa_present else 0.0,
            "rationale": str(wa_raw.get("rationale") or ""),
        },
        relates_to_target=relates,
        action_role=str(data.get("action_role") or "implementation"),
        actual_effect=str(data.get("actual_effect") or data.get("rationale") or ""),
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def build_user_prompt(
    *,
    task_description: str,
    action: dict[str, Any],
    evidence: ActionEvidence,
    trajectory_summary: str,
) -> str:
    compact_evidence = {
        "action_index": evidence.action_index,
        "action_kind": evidence.action_kind,
        "target_files": evidence.target_files,
        "replay_evidence": evidence.replay_evidence,
        "static_evidence": evidence.static_evidence,
        "state_evidence": evidence.state_evidence,
    }
    raw_action = str(action.get("raw_action") or evidence.raw_action or "")
    if len(raw_action) > config.ACTION_CONTENT_MAX_CHARS:
        raw_action = raw_action[: config.ACTION_CONTENT_MAX_CHARS] + "\n...[truncated]"
    return f"""You are a human-target v2 offline encoder judge.

The deterministic pipeline already extracted replay/static/state facts. Treat
those facts as fixed. Judge only semantic risk for this scorable action.

Use narrow definitions:
- observability_loss is non-zero only for swallowing errors, weakening validation,
  hiding failures, or polluting diagnostics.
- wrong_abstraction is true only when implementation is placed in the wrong
  structural location, abstraction, interface, dependency direction, or module boundary.
- If static_evidence.wrong_abstraction_hints is present, evaluate it carefully.
  Structural code inserted into the wrong local context, duplicate structural
  payloads, or a later validation error tied to the touched file are strong
  wrong_abstraction evidence when they show the implementation must be moved
  across a structural boundary.
- Interface-boundary changes are also wrong_abstraction when the action pushes
  responsibility across the wrong API/model boundary, for example converting a
  resolved dependency into a Promise in synchronous code, renaming a cache while
  still storing the old value model, or repurposing an existing query/config
  contract where the task requires a separate boundary.
- artifact residue and broad rewrite are trajectory-level issues unless this
  action also has direct semantic risk.

Task description:
{task_description}

Current action:
{raw_action}

Evidence table for this action:
{json.dumps(compact_evidence, ensure_ascii=False, indent=2)}

Compact trajectory summary:
{trajectory_summary}

Return one JSON object exactly in this shape:
{{
  "relates_to_target": "yes|no|uncertain",
  "action_role": "implementation|test_or_config|cleanup|uncertain",
  "actual_effect": "<short effect using action index and file path>",
  "task_relevance": "none|low|medium|high",
  "debt_level": "none|low|medium|high",
  "fragility_level": "none|low|medium|high",
  "regression_surface_level": "local|module|cross_module|system",
  "observability_loss_evidence": "none|weak|clear|severe",
  "wrong_abstraction": {{
    "present": <true|false>,
    "severity_level": "none|low|medium|high",
    "rationale": "<short rationale citing action index and file path>"
  }},
  "confidence": <0.0-1.0>,
  "rationale": "<short evidence-based rationale>"
}}
"""


SYSTEM_PROMPT = "Return valid JSON only. Do not include markdown or extra commentary."


def _level(value: Any, mapping: dict[str, float], field: str, failures: list[str]) -> float:
    if value not in mapping:
        failures.append(field)
        return 0.0
    return mapping[str(value)]


def _tri_bool(value: Any, failures: list[str]) -> bool | None:
    if value == "yes":
        return True
    if value == "no":
        return False
    if value == "uncertain":
        return None
    failures.append("relates_to_target")
    return None


def _confidence(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.6
    if f != f:
        return 0.6
    return round(max(0.0, min(1.0, f)), 3)


def _fallback_result(failures: list[str], raw: str) -> SemanticResult:
    return SemanticResult(
        manual_risk_vector={
            "task_advancement": 0.5,
            "debt_density": 0.0,
            "fragility_delta": 0.0,
            "regression_surface": 0.2,
            "observability_loss": 0.0,
            "rationale": "Semantic parse failed; conservative fallback.",
            "annotator_confidence": 0.3,
        },
        wrong_abstraction={"present": False, "severity": 0.0, "rationale": ""},
        relates_to_target=None,
        action_role="uncertain",
        actual_effect="Semantic parse failed; fallback judgment.",
        parse_failed=True,
        parse_failures=failures,
        raw=raw,
    )
