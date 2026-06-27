"""LLM output parsers for human-target v2 judgments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


VALID_RISK_SCOPES = {
    "substantive",
    "noise_no_effect",
    "noise_reverted",
    "temporary_verification",
    "artifact_only",
    "uncertain",
}

VALID_SCOPE_REVIEW_SCOPES = VALID_RISK_SCOPES - {"substantive"}

VALID_SCOPE_REVIEW_EVIDENCE = {
    "reverted_later",
    "superseded_later",
    "exploratory_or_dead_end",
    "artifact_or_verification",
    "insufficient_final_contribution",
}

VALID_SCOPE_EVIDENCE_TRI = {"yes", "no", "uncertain"}
VALID_SCOPE_EVIDENCE_LATER_STATUS = {"unchanged", "reverted", "superseded", "unknown"}

SCALAR_DIMS = [
    "task_advancement",
    "debt_density",
    "fragility_delta",
    "regression_surface",
    "observability_loss",
]


@dataclass
class V2ActionResult:
    risk_scope: str | None
    risk_scope_rationale: str = ""
    manual_risk_vector: dict[str, Any] = field(default_factory=dict)
    wrong_abstraction: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    annotator_confidence: float | None = None
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class V2TrajectoryResult:
    trajectory_penalties: dict[str, dict[str, Any]] = field(default_factory=dict)
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class ScopeReviewResult:
    scope_overrides: list[dict[str, Any]] = field(default_factory=list)
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class ScopeEvidenceResult:
    action_evidence: list[dict[str, Any]] = field(default_factory=list)
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


def _extract_json(text: str) -> dict | None:
    """Return the first JSON object embedded in model output."""
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _score(value: Any) -> float | None:
    """Parse a numeric 0-1 score, clamping numeric over/underflow."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(max(0.0, min(1.0, f)), 3)


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return None


def _list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def parse_action_v2(text: str) -> V2ActionResult:
    """Parse one human-target v2 action judgment."""
    data = _extract_json(text)
    if data is None:
        return V2ActionResult(
            risk_scope=None, parse_failed=True, parse_failures=["json"], raw=text
        )

    failures: list[str] = []
    risk_scope = data.get("risk_scope")
    if risk_scope not in VALID_RISK_SCOPES:
        risk_scope = None
        failures.append("risk_scope")

    raw_rv = data.get("manual_risk_vector")
    if not isinstance(raw_rv, dict):
        raw_rv = {}
        failures.append("manual_risk_vector")

    rv: dict[str, Any] = {}
    for dim in SCALAR_DIMS:
        parsed = _score(raw_rv.get(dim))
        if parsed is None:
            failures.append(dim)
        rv[dim] = parsed
    rv["rationale"] = str(raw_rv.get("rationale") or data.get("reasoning") or "")

    confidence = _score(raw_rv.get("annotator_confidence", data.get("confidence")))
    rv["annotator_confidence"] = confidence

    raw_wa = data.get("wrong_abstraction")
    if not isinstance(raw_wa, dict):
        raw_wa = {}
        failures.append("wrong_abstraction")
    wa_present = _bool_or_none(raw_wa.get("present"))
    if wa_present is None:
        failures.append("wrong_abstraction.present")
    wa_severity = _score(raw_wa.get("severity"))
    if wa_severity is None:
        failures.append("wrong_abstraction.severity")
    wrong_abstraction = {
        "present": wa_present,
        "severity": wa_severity,
        "rationale": str(raw_wa.get("rationale") or ""),
    }

    return V2ActionResult(
        risk_scope=risk_scope,
        risk_scope_rationale=str(data.get("risk_scope_rationale") or ""),
        manual_risk_vector=rv,
        wrong_abstraction=wrong_abstraction,
        reasoning=str(data.get("reasoning") or ""),
        annotator_confidence=confidence,
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def _parse_penalty(raw: Any, prefix: str, failures: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        failures.append(prefix)
        raw = {}
    present = _bool_or_none(raw.get("present"))
    if present is None:
        failures.append(f"{prefix}.present")
    severity = _score(raw.get("severity"))
    if severity is None:
        failures.append(f"{prefix}.severity")
    return {
        "present": present,
        "severity": severity,
        "evidence_scopes": _list(raw.get("evidence_scopes")),
        "affected_files": _list(raw.get("affected_files")),
        "artifact_types": _list(raw.get("artifact_types")),
        "paths": _list(raw.get("paths")),
        "evidence_actions": _list(raw.get("evidence_actions")),
        "rationale": str(raw.get("rationale") or ""),
    }


def parse_trajectory_v2(text: str) -> V2TrajectoryResult:
    """Parse one human-target v2 trajectory-level judgment."""
    data = _extract_json(text)
    if data is None:
        return V2TrajectoryResult(
            parse_failed=True, parse_failures=["json"], raw=text
        )

    failures: list[str] = []
    raw_penalties = data.get("trajectory_penalties")
    if not isinstance(raw_penalties, dict):
        raw_penalties = data

    penalties = {
        "broad_rewrite": _parse_penalty(
            raw_penalties.get("broad_rewrite"), "broad_rewrite", failures
        ),
        "artifact_residue": _parse_penalty(
            raw_penalties.get("artifact_residue"), "artifact_residue", failures
        ),
    }
    return V2TrajectoryResult(
        trajectory_penalties=penalties,
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def parse_scope_review_v2(text: str) -> ScopeReviewResult:
    """Parse trajectory-level scope review overrides."""
    data = _extract_json(text)
    if data is None:
        return ScopeReviewResult(
            parse_failed=True, parse_failures=["json"], raw=text
        )

    failures: list[str] = []
    raw_overrides = data.get("scope_overrides")
    if raw_overrides in (None, ""):
        raw_overrides = []
    if not isinstance(raw_overrides, list):
        failures.append("scope_overrides")
        raw_overrides = []

    overrides: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_overrides):
        prefix = f"scope_overrides[{i}]"
        if not isinstance(raw, dict):
            failures.append(prefix)
            continue

        action_index = raw.get("action_index")
        try:
            action_index = int(action_index)
        except (TypeError, ValueError):
            failures.append(f"{prefix}.action_index")
            continue

        risk_scope = raw.get("risk_scope")
        if risk_scope not in VALID_SCOPE_REVIEW_SCOPES:
            failures.append(f"{prefix}.risk_scope")
            risk_scope = None

        confidence = _score(raw.get("confidence"))
        if confidence is None:
            failures.append(f"{prefix}.confidence")

        evidence_type = raw.get("evidence_type")
        if evidence_type not in VALID_SCOPE_REVIEW_EVIDENCE:
            failures.append(f"{prefix}.evidence_type")
            evidence_type = None

        if risk_scope is None or confidence is None or evidence_type is None:
            continue

        overrides.append({
            "action_index": action_index,
            "risk_scope": risk_scope,
            "confidence": confidence,
            "evidence_type": evidence_type,
            "rationale": str(raw.get("rationale") or ""),
        })

    return ScopeReviewResult(
        scope_overrides=overrides,
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def parse_scope_evidence_v2(text: str) -> ScopeEvidenceResult:
    """Parse trajectory-level scope evidence without accepting direct labels."""
    data = _extract_json(text)
    if data is None:
        return ScopeEvidenceResult(
            parse_failed=True, parse_failures=["json"], raw=text
        )

    failures: list[str] = []
    raw_items = data.get("action_evidence")
    if raw_items in (None, ""):
        raw_items = []
    if not isinstance(raw_items, list):
        failures.append("action_evidence")
        raw_items = []

    items: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_items):
        prefix = f"action_evidence[{i}]"
        if not isinstance(raw, dict):
            failures.append(prefix)
            continue

        action_index = raw.get("action_index")
        try:
            action_index = int(action_index)
        except (TypeError, ValueError):
            failures.append(f"{prefix}.action_index")
            continue

        confidence = _score(raw.get("confidence"))
        if confidence is None:
            failures.append(f"{prefix}.confidence")

        durable = _tri_value(raw.get("durable_repo_effect"))
        contribution = _tri_value(raw.get("final_task_contribution"))
        artifact = _tri_value(raw.get("artifact_or_verification"))
        later_status = raw.get("later_status")
        if durable is None:
            failures.append(f"{prefix}.durable_repo_effect")
        if contribution is None:
            failures.append(f"{prefix}.final_task_contribution")
        if artifact is None:
            failures.append(f"{prefix}.artifact_or_verification")
        if later_status not in VALID_SCOPE_EVIDENCE_LATER_STATUS:
            failures.append(f"{prefix}.later_status")
            later_status = None

        evidence_actions = _int_list(raw.get("evidence_actions"))
        if confidence is None or durable is None or contribution is None or artifact is None or later_status is None:
            continue

        items.append({
            "action_index": action_index,
            "durable_repo_effect": durable,
            "final_task_contribution": contribution,
            "later_status": later_status,
            "artifact_or_verification": artifact,
            "confidence": confidence,
            "evidence_actions": evidence_actions,
            "rationale": str(raw.get("rationale") or ""),
        })

    return ScopeEvidenceResult(
        action_evidence=items,
        parse_failed=bool(failures),
        parse_failures=failures,
        raw=text,
    )


def _tri_value(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip().lower()
    return value if value in VALID_SCOPE_EVIDENCE_TRI else None


def _int_list(value: Any) -> list[int]:
    out: list[int] = []
    for item in _list(value):
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out
