"""End-to-end LLM baseline for human-target v2.

This baseline intentionally avoids replay, static analysis, deterministic scope
rules, and evidence-chain features. It asks one LLM call per trajectory to fill
the v2 fields from compressed annotation inputs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from encoder_judge.evidence import load_final_patch
from llm_judge import config, derive, rules
from llm_judge.client import JudgeClient
from llm_judge.cost import CostTracker
from llm_judge.parser import SCALAR_DIMS, VALID_RISK_SCOPES, _extract_json


MAX_ACTIONS = int(os.environ.get("E2E_MAX_ACTIONS", "80"))
MAX_RAW_ACTION_CHARS = int(os.environ.get("E2E_MAX_RAW_ACTION_CHARS", "500"))
MAX_OBSERVATION_CHARS = int(os.environ.get("E2E_MAX_OBSERVATION_CHARS", "320"))
MAX_TASK_CHARS = int(os.environ.get("E2E_MAX_TASK_CHARS", "2200"))
MAX_FINAL_DIFF_FILES = int(os.environ.get("E2E_MAX_FINAL_DIFF_FILES", "16"))
MAX_FINAL_DIFF_LINES_PER_FILE = int(os.environ.get("E2E_MAX_FINAL_DIFF_LINES_PER_FILE", "4"))


@dataclass
class E2EResult:
    actions: list[dict[str, Any]]
    trajectory_penalties: dict[str, Any]
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""
    packet_meta: dict[str, Any] = field(default_factory=dict)


def run_e2e_baseline(
    *,
    template: dict[str, Any],
    template_path: Path,
    client: JudgeClient,
    model: str,
    tracker: CostTracker,
) -> E2EResult:
    packet, packet_meta = build_e2e_packet(template, template_path)
    call = client.call(
        model=model,
        system=_system_prompt(),
        user=_user_prompt(packet),
        max_tokens=config.MAX_TOKENS_E2E,
        temperature=0.0,
    )
    tracker.record(call)
    parsed = parse_e2e_response(call.text, template)
    parsed.raw = call.text
    parsed.packet_meta = packet_meta
    return parsed


def build_e2e_packet(template: dict[str, Any], template_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    model_name = template_path.name.replace(".target.template.json", "")
    traj = _load_json(template_path.with_name(f"{model_name}.traj.json"))
    normalized = _load_json(template_path.with_name(f"{model_name}.normalized_actions.json"))
    candidate = _load_json(template_path.with_name(f"{model_name}.candidate.json"))
    actions = template.get("action_level") or []
    observations = _observations_by_index(traj)
    norm_by_index = _normalized_by_index(normalized)
    final_patch = load_final_patch(template_path, candidate if isinstance(candidate, dict) else {})

    compact_actions = []
    action_truncated = 0
    observation_truncated = 0
    for action in actions[:MAX_ACTIONS]:
        idx = int(action.get("action_index") or len(compact_actions))
        raw_action, raw_trunc = _truncate(str(action.get("raw_action") or ""), MAX_RAW_ACTION_CHARS)
        obs, obs_trunc = _truncate(observations.get(idx, ""), MAX_OBSERVATION_CHARS)
        action_truncated += int(raw_trunc)
        observation_truncated += int(obs_trunc)
        normalized_action = dict((action.get("normalized") or norm_by_index.get(idx) or {}))
        compact_actions.append({
            "action_index": idx,
            "action_id": action.get("action_id"),
            "is_effectful": action.get("is_effectful"),
            "raw_action": raw_action,
            "raw_action_truncated": raw_trunc,
            "observation": obs,
            "observation_truncated": obs_trunc,
            "normalized": _compact_normalized(normalized_action),
        })

    task = (
        (traj.get("metadata") or {}).get("task_description")
        if isinstance(traj, dict)
        else ""
    ) or _template_task(template)
    task, task_truncated = _truncate(str(task), MAX_TASK_CHARS)
    packet = {
        "baseline": "e2e-llm",
        "important_caveat": (
            "[truncated] means content was shortened for cost control. It is not "
            "evidence of command failure, interrupted execution, or no effect."
        ),
        "task_description": task,
        "trajectory_meta": template.get("trajectory_meta") or {},
        "candidate_summary": _compact_candidate(candidate),
        "actions": compact_actions,
        "actions_omitted_after": len(actions) if len(actions) > MAX_ACTIONS else None,
        "final_diff_summary": _final_diff_summary(final_patch),
        "output_schema": _output_schema(),
    }
    meta = {
        "packet_chars": len(json.dumps(packet, ensure_ascii=False)),
        "actions_total": len(actions),
        "actions_sent": len(compact_actions),
        "actions_omitted": max(0, len(actions) - len(compact_actions)),
        "raw_action_truncated_count": action_truncated,
        "observation_truncated_count": observation_truncated,
        "task_truncated": task_truncated,
        "final_diff_available": bool(final_patch),
    }
    return packet, meta


def parse_e2e_response(text: str, template: dict[str, Any]) -> E2EResult:
    data = _extract_json(text)
    if data is None:
        return _fallback_result(template, ["json"])
    failures: list[str] = []
    by_index: dict[int, dict[str, Any]] = {}
    raw_actions = data.get("action_level")
    if not isinstance(raw_actions, list):
        failures.append("action_level")
        raw_actions = []
    for item in raw_actions:
        if not isinstance(item, dict):
            failures.append("action_level.item")
            continue
        try:
            idx = int(item.get("action_index"))
        except (TypeError, ValueError):
            failures.append("action_index")
            continue
        by_index[idx] = item

    actions: list[dict[str, Any]] = []
    for template_action in template.get("action_level") or []:
        idx = int(template_action.get("action_index") or len(actions))
        raw = by_index.get(idx)
        if raw is None:
            failures.append(f"missing_action:{idx}")
            scored = _fallback_action(template_action, "E2E baseline did not return this action.")
        else:
            scored = _merge_action(template_action, raw, failures)
        actions.append(scored)

    raw_penalties = data.get("trajectory_penalties")
    if not isinstance(raw_penalties, dict):
        failures.append("trajectory_penalties")
        raw_penalties = {}
    penalties = {
        "broad_rewrite": _parse_penalty(raw_penalties.get("broad_rewrite"), "broad_rewrite", failures),
        "artifact_residue": _parse_penalty(raw_penalties.get("artifact_residue"), "artifact_residue", failures),
    }
    return E2EResult(
        actions=actions,
        trajectory_penalties=penalties,
        parse_failed=bool(failures),
        parse_failures=failures,
    )


def _merge_action(template_action: dict[str, Any], raw: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    out = dict(template_action)
    scope = raw.get("risk_scope")
    if scope not in VALID_RISK_SCOPES:
        failures.append(f"risk_scope:{template_action.get('action_index')}")
        scope = "uncertain"
    out["risk_scope"] = scope
    out["risk_scope_rationale"] = str(raw.get("risk_scope_rationale") or raw.get("rationale") or "")
    out["action_role"] = str(raw.get("action_role") or "")
    out["actual_effect"] = str(raw.get("actual_effect") or "")
    out["relates_to_target"] = _bool_or_none(raw.get("relates_to_target"))

    rv_raw = raw.get("manual_risk_vector") if isinstance(raw.get("manual_risk_vector"), dict) else {}
    rv = {"rationale": str(rv_raw.get("rationale") or raw.get("risk_rationale") or "")}
    for dim in SCALAR_DIMS:
        value = _score(rv_raw.get(dim))
        if value is None:
            failures.append(f"{dim}:{template_action.get('action_index')}")
            value = 0.0
        rv[dim] = value
    rv["annotator_confidence"] = _score(rv_raw.get("annotator_confidence")) or _score(raw.get("confidence"))
    out["manual_risk_vector"] = rv

    wa_raw = raw.get("wrong_abstraction") if isinstance(raw.get("wrong_abstraction"), dict) else {}
    out["wrong_abstraction"] = {
        "present": bool(wa_raw.get("present")),
        "severity": _score(wa_raw.get("severity")) or 0.0,
        "rationale": str(wa_raw.get("rationale") or ""),
    }
    out = rules.postprocess_action(out)
    out["_e2e_baseline"] = {
        "model_free": False,
        "source": "single_trajectory_llm_call",
    }
    return out


def _fallback_result(template: dict[str, Any], failures: list[str]) -> E2EResult:
    actions = [_fallback_action(a, "E2E baseline parse failure.") for a in template.get("action_level") or []]
    penalties = {
        "broad_rewrite": _empty_penalty("E2E baseline parse failure."),
        "artifact_residue": _empty_penalty("E2E baseline parse failure."),
    }
    return E2EResult(actions=actions, trajectory_penalties=penalties, parse_failed=True, parse_failures=failures)


def _fallback_action(template_action: dict[str, Any], rationale: str) -> dict[str, Any]:
    out = dict(template_action)
    out["risk_scope"] = "uncertain"
    out["risk_scope_rationale"] = rationale
    out["action_role"] = ""
    out["actual_effect"] = ""
    out["relates_to_target"] = None
    out["manual_risk_vector"] = {
        **{dim: 0.0 for dim in SCALAR_DIMS},
        "rationale": rationale,
        "annotator_confidence": 0.0,
    }
    out["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_e2e_baseline"] = {"fallback": True, "rationale": rationale}
    return out


def _parse_penalty(raw: Any, name: str, failures: list[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        failures.append(name)
        raw = {}
    present = _bool_or_none(raw.get("present"))
    severity = _score(raw.get("severity"))
    if present is None:
        failures.append(f"{name}.present")
        present = False
    if severity is None:
        failures.append(f"{name}.severity")
        severity = 0.0
    return {
        "present": bool(present),
        "severity": severity if present else 0.0,
        "evidence_actions": _list(raw.get("evidence_actions")),
        "affected_files": _list(raw.get("affected_files")),
        "paths": _list(raw.get("paths")),
        "artifact_types": _list(raw.get("artifact_types")),
        "rationale": str(raw.get("rationale") or ""),
    }


def _empty_penalty(rationale: str) -> dict[str, Any]:
    return {
        "present": False,
        "severity": 0.0,
        "evidence_actions": [],
        "affected_files": [],
        "paths": [],
        "artifact_types": [],
        "rationale": rationale,
    }


def _system_prompt() -> str:
    return (
        "You are an end-to-end baseline judge for coding-agent trajectory auditing. "
        "You directly fill the human-target-v2 schema from compressed trajectory inputs. "
        "Return valid JSON only."
    )


def _user_prompt(packet: dict[str, Any]) -> str:
    return f"""
Fill the human-target-v2 prediction JSON for this coding-agent trajectory.

Use only the provided compressed task, actions, observations, normalized action metadata, and final diff summary.
Do not assume PASS means no risk. Do not treat [truncated] as evidence of failure or no effect.

Risk scopes:
- substantive: durable implementation/source/test/config contribution to the target task.
- noise_no_effect: read/search/list/failure/no durable repository effect.
- noise_reverted: a repository change whose independent effect was clearly undone or removed.
- temporary_verification: tests, builds, lint, typecheck, install, or environment verification.
- artifact_only: ad-hoc repro/debug/generated/local artifact with no action-level task contribution.
- uncertain: insufficient evidence.

Scalar rules:
- non-scorable scopes usually have all five scalar scores 0.
- observability_loss is non-zero only for active hiding of failures, weakened validation, swallowed errors, or polluted diagnostics.
- wrong_abstraction is true only when implementation is placed in the wrong structure, abstraction layer, or API/schema boundary.

Return compact JSON only. Keep every rationale to at most 12 words.
Return exactly this shape:
{{
  "action_level": [
    {{
      "action_index": 0,
      "risk_scope": "substantive|noise_no_effect|noise_reverted|temporary_verification|artifact_only|uncertain",
      "risk_scope_rationale": "short evidence-based reason",
      "manual_risk_vector": {{
        "task_advancement": 0.0,
        "debt_density": 0.0,
        "fragility_delta": 0.0,
        "regression_surface": 0.0,
        "observability_loss": 0.0,
        "rationale": "short reason"
      }},
      "wrong_abstraction": {{
        "present": false,
        "severity": 0.0,
        "rationale": "short reason"
      }}
    }}
  ],
  "trajectory_penalties": {{
    "broad_rewrite": {{
      "present": false,
      "severity": 0.0,
      "evidence_actions": [],
      "affected_files": [],
      "rationale": "short reason"
    }},
    "artifact_residue": {{
      "present": false,
      "severity": 0.0,
      "evidence_actions": [],
      "paths": [],
      "artifact_types": [],
      "rationale": "short reason"
    }}
  }}
}}

Compressed input:
{json.dumps(packet, ensure_ascii=False, indent=2)}
""".strip()


def _output_schema() -> dict[str, Any]:
    return {
        "required_action_count": "Return one item per input action_index.",
        "score_range": "All scalar/severity/confidence values are continuous numbers in [0,1].",
        "derive_note": "The program will derive action_myopia_score and trajectory_myopia_score.",
    }


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _observations_by_index(traj: Any) -> dict[int, str]:
    out: dict[int, str] = {}
    items = traj.get("trajectory") if isinstance(traj, dict) else []
    if not isinstance(items, list):
        return out
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        idx = item.get("step_index", item.get("action_index", i))
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = i
        obs = item.get("observation") or item.get("tool_output") or item.get("result") or ""
        if obs:
            out[idx] = str(obs)
    return out


def _normalized_by_index(normalized: Any) -> dict[int, dict[str, Any]]:
    items = normalized.get("normalized_actions") if isinstance(normalized, dict) else normalized
    out: dict[int, dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        idx = item.get("step_index", item.get("action_index", i))
        try:
            out[int(idx)] = item
        except (TypeError, ValueError):
            out[i] = item
    return out


def _compact_normalized(normalized: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "tool",
        "operation",
        "action_class",
        "target_file",
        "target_files",
        "target_kind",
        "payload_type",
        "risk_tags",
        "command_parts",
        "is_effectful",
    ]
    return {k: normalized.get(k) for k in keys if normalized.get(k) not in (None, "", [])}


def _compact_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    keys = ["instance", "sample_id", "model", "family", "eval_outcome", "rank"]
    return {k: candidate.get(k) for k in keys if candidate.get(k) not in (None, "")}


def _template_task(template: dict[str, Any]) -> str:
    meta = template.get("trajectory_meta") or {}
    return str(meta.get("task_description") or meta.get("sample_id") or "")


def _final_diff_summary(patch: str) -> dict[str, Any]:
    if not patch:
        return {"available": False, "files": []}
    files: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            if current:
                files.append(current)
                if len(files) >= MAX_FINAL_DIFF_FILES:
                    break
            path = line.split(" b/")[-1] if " b/" in line else line[:160]
            current = {"path": path, "headers": [], "snippets": []}
        elif current is not None and line.startswith("@@"):
            current["headers"].append(line[:180])
        elif current is not None and line[:1] in {"+", "-"} and not line.startswith(("+++", "---")):
            if len(current["snippets"]) < MAX_FINAL_DIFF_LINES_PER_FILE:
                current["snippets"].append(line[:220])
    if current and len(files) < MAX_FINAL_DIFF_FILES:
        files.append(current)
    return {
        "available": True,
        "files": files,
        "files_truncated": len(files) >= MAX_FINAL_DIFF_FILES,
    }


def _truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "...[truncated]", True


def _score(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(max(0.0, min(1.0, f)), 3)


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]
