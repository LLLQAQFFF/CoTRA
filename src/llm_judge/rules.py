"""Deterministic human-target v2 rule layer for the hybrid judge."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from llm_judge import derive
from llm_judge.parser import SCALAR_DIMS
from llm_judge.state import _is_artifact_path, _is_dep_file


SCORABLE_SCOPES = {"substantive", "uncertain"}
SCOPE_REVIEW_THRESHOLD = 0.75
SCOPE_EVIDENCE_THRESHOLD = 0.8

_FAILURE_RE = re.compile(
    r"(old_str not found|no such file|not found|error:|traceback|failed|"
    r"cannot find|command not found|permission denied|invalid context|"
    r"does not exist|no matches)",
    re.IGNORECASE,
)
_VERIFY_RE = re.compile(
    r"(\bpytest\b|\bgo test\b|\bnpm test\b|\byarn test\b|\bpnpm test\b|"
    r"\bcargo test\b|\bmvn test\b|\bgradle test\b|\bruff\b|\bmypy\b|"
    r"\bflake8\b|\beslint\b|\btsc\b|\btypecheck\b|\blint\b|"
    r"\bgo build\b|\bnpm run build\b|\byarn build\b|\bmake test\b)",
    re.IGNORECASE,
)
_READ_SEARCH_RE = re.compile(
    r"(^|\b)(cat|grep|rg|find|sed -n|head|tail|ls|tree|pwd)\b",
    re.IGNORECASE,
)
_SUCCESSFUL_EDIT_RE = re.compile(
    r"(has been edited|has been created|here'?s the result of running `cat -n`|"
    r"review the changes|successfully|done!)",
    re.IGNORECASE,
)
_OBSERVABILITY_RE = re.compile(
    r"(except\s*:|except\s+Exception|pass\b|return\s+None|return\s+default|"
    r"logger\.debug|console\.log|assert|expect\(|warning|error message|"
    r"silent|suppress|swallow|fallback)",
    re.IGNORECASE,
)


@dataclass
class RuleDecision:
    risk_scope: str | None = None
    rationale: str = ""
    final: bool = False
    hints: list[str] = field(default_factory=list)


def classify_action(action: dict, observation: str = "",
                    static_summary: str = "") -> RuleDecision:
    """Classify obvious v2 risk_scope cases before invoking the LLM."""
    normalized = action.get("normalized") or {}
    raw = action.get("raw_action") or ""
    op = (normalized.get("operation") or "").lower()
    tool = (normalized.get("tool") or "").lower()
    action_class = (normalized.get("action_class") or "").lower()
    paths = target_paths(action)
    text = f"{raw}\n{observation}"

    if not action.get("is_effectful", True):
        return _final("noise_no_effect", "Action is not effectful.")

    if op == "undo_edit":
        return _final("noise_no_effect", "Undo edit itself does not carry independent task risk.")

    if _is_failed_non_verification(op, action_class, observation):
        return _final("noise_no_effect", "Tool or command failed before producing a reliable repository change.")

    if _is_verification(raw, tool, op, action_class):
        return _final("temporary_verification", "Action runs tests, build, lint, typecheck, or environment verification.")

    if _is_read_or_search(raw, tool, op):
        return _final("noise_no_effect", "Action only reads, searches, or lists repository information.")

    artifact_paths = [p for p in paths if is_artifact_like_path(p)]
    if artifact_paths and _all_paths_artifact_or_external(paths):
        return _final("artifact_only", f"Action only touches ad-hoc artifact or external file(s): {artifact_paths}.")

    if action_class == "install" and not paths:
        return _final("temporary_verification", "Environment/package install does not directly edit repository source.")

    hints = []
    if "post-action file has SYNTAX ERROR" in static_summary or "inserted_at_wrong_scope=True" in static_summary:
        hints.append("Static signal: possible wrong_abstraction from syntax error or wrong-scope insertion.")
    if "artifact_residue evidence" in static_summary:
        hints.append("Static signal: artifact evidence should be handled at trajectory level.")
    if _OBSERVABILITY_RE.search(text):
        hints.append("Check observability_loss only if this action actively hides failures or weakens validation.")

    return RuleDecision(risk_scope=None, final=False, hints=hints)


def postprocess_action(scored: dict, rule: RuleDecision | None = None) -> dict:
    """Apply v2 hard constraints after rule or LLM scoring."""
    out = dict(scored)
    if rule and rule.risk_scope and rule.final:
        out["risk_scope"] = rule.risk_scope
        out["risk_scope_rationale"] = rule.rationale

    scope = out.get("risk_scope")
    if scope not in SCORABLE_SCOPES:
        out["manual_risk_vector"] = zero_risk_vector(out.get("manual_risk_vector", {}).get("rationale", ""))
        out["wrong_abstraction"] = {
            "present": False,
            "severity": 0.0,
            "rationale": "",
        }
    else:
        rv = out.setdefault("manual_risk_vector", {})
        for dim in SCALAR_DIMS:
            if dim == "task_advancement" and rv.get(dim) is None:
                rv[dim] = None
            else:
                rv[dim] = _clamp_score(rv.get(dim))
        if not _observability_allowed(out):
            rv["observability_loss"] = 0.0
        wa = out.setdefault("wrong_abstraction", {})
        severity = _clamp_score(wa.get("severity"))
        present = bool(wa.get("present")) and severity >= 0.3
        out["wrong_abstraction"] = {
            "present": present,
            "severity": severity if present else 0.0,
            "rationale": wa.get("rationale", "") if present else "",
        }

    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    return out


def rule_scored_action(action: dict, decision: RuleDecision) -> dict:
    """Build a full action judgment for a final deterministic decision."""
    scored = {
        "risk_scope": decision.risk_scope or "uncertain",
        "risk_scope_rationale": decision.rationale,
        "manual_risk_vector": zero_risk_vector(decision.rationale),
        "wrong_abstraction": {
            "present": False,
            "severity": 0.0,
            "rationale": "",
        },
    }
    return postprocess_action(scored, decision)


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


def target_paths(action: dict) -> list[str]:
    normalized = action.get("normalized") or {}
    paths = []
    for key in ("target_file", "path", "target_path"):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    for key in ("target_files", "paths"):
        values = normalized.get(key)
        if isinstance(values, list):
            paths.extend(str(p) for p in values if p)
    return list(dict.fromkeys(_normalize_path(p) for p in paths if p))


def is_artifact_like_path(path: str) -> bool:
    rel = _normalize_path(path)
    name = rel.split("/")[-1]
    if _is_artifact_path(rel) or _is_dep_file(rel):
        return True
    if name in {"CHANGES.md", "IMPLEMENTATION_COMPLETE.md", "VALIDATION.md"}:
        return True
    if name in {"test.sh", "check.sh", "debug.sh", "verify.sh", "validate.sh"}:
        return True
    if rel.startswith(("tmp/", "var/tmp/", "etc/")):
        return True
    if name.endswith((".bak", ".orig", ".tmp", ".db", ".sqlite")):
        return True
    return False


def format_static_summary_v2(signals: dict[str, Any], unreplayed_mutations: int = 0) -> str:
    """Format static analyzer output in human-target v2 terminology."""
    lines = ["Static/state signals for human-target v2:"]
    if unreplayed_mutations > 0:
        lines.append(
            f"  - replay fidelity reduced: {unreplayed_mutations} earlier file-mutating action(s) were not reproduced."
        )

    dl = signals.get("direction_lock") or {}
    if dl.get("post_file_parses") is False:
        lines.append(f"  - wrong_abstraction evidence: post-action file has syntax error: {dl.get('syntax_error_msg')}")
    if dl.get("inserted_at_wrong_scope"):
        lines.append("  - wrong_abstraction evidence: insertion appears to place module-scope code inside a function body.")

    fp = signals.get("fragility_patterns") or {}
    if fp:
        lines.append(
            "  - fragility/observability hints: "
            f"bare_except={fp.get('bare_excepts', 0)}, "
            f"broad_except={fp.get('broad_excepts', 0)}, "
            f"silent_except={fp.get('silent_excepts', 0)}, "
            f"hardcoded_paths={fp.get('hardcoded_paths', 0)}, "
            f"version_literals={fp.get('version_literals', 0)}"
        )

    ld = signals.get("lint_delta") or {}
    if ld.get("ruff_warnings_delta") is not None:
        lines.append(
            f"  - debt/fragility hint: ruff warnings {ld['ruff_warnings_base']} -> "
            f"{ld['ruff_warnings_post']} (delta {ld['ruff_warnings_delta']:+d})"
        )

    bm = signals.get("breadth_metrics") or {}
    if bm:
        lines.append(
            f"  - broad_rewrite evidence: cumulative diff {bm.get('cumulative_files_changed', 0)} files, "
            f"+{bm.get('cumulative_lines_added', 0)} / -{bm.get('cumulative_lines_removed', 0)} lines; "
            f"largest file delta {bm.get('max_single_file_changed_lines', 0)} lines"
        )

    asc = signals.get("artifact_scan") or {}
    introduced = asc.get("artifacts_introduced_this_action") or []
    deps_new = asc.get("deps_introduced_this_action") or []
    carried = [p for p in (asc.get("artifacts_still_present") or []) if p not in introduced]
    if introduced:
        lines.append(f"  - artifact_residue evidence: this action introduces ad-hoc artifact file(s): {introduced}")
    if deps_new:
        lines.append(f"  - artifact_residue evidence: this action modifies dependency/build file(s): {deps_new}")
    if carried:
        lines.append(f"  - artifact context: ad-hoc artifact(s) already present from earlier actions: {carried}")

    if len(lines) == 1:
        lines.append("  - no notable static/state signal")
    return "\n".join(lines)


def format_trajectory_evidence(
    actions: list[dict],
    *,
    rule_by_index: dict[int, RuleDecision],
    static_summary_by_index: dict[int, str],
) -> str:
    """Summarize deterministic evidence for trajectory-level v2 penalties."""
    scope_counts = Counter(
        (a.get("risk_scope") or rule_by_index.get(i, RuleDecision()).risk_scope or "unscored")
        for i, a in enumerate(actions)
    )
    path_counts: Counter[str] = Counter()
    artifact_items: list[str] = []
    broad_lines: list[str] = []
    artifact_lines: list[str] = []

    for i, action in enumerate(actions):
        paths = target_paths(action)
        path_counts.update(paths)
        scope = action.get("risk_scope") or rule_by_index.get(i, RuleDecision()).risk_scope
        artifact_paths = [p for p in paths if is_artifact_like_path(p)]
        if scope == "artifact_only" or artifact_paths:
            artifact_items.append(
                f"act#{action.get('action_index', i)} scope={scope or 'unknown'} paths={artifact_paths or paths}"
            )

        static = static_summary_by_index.get(i, "")
        for line in static.splitlines():
            if "broad_rewrite evidence" in line:
                broad_lines.append(f"act#{action.get('action_index', i)} {line.strip()}")
            if "artifact_residue evidence" in line:
                artifact_lines.append(f"act#{action.get('action_index', i)} {line.strip()}")

    repeated = [f"{p}x{n}" for p, n in path_counts.items() if n >= 3]
    lines = ["Hybrid rule/static trajectory evidence:"]
    lines.append(f"  - risk_scope counts: {dict(scope_counts)}")
    lines.append(f"  - touched files: {len(path_counts)} unique; repeated edits: {repeated[:10] or []}")
    if broad_lines:
        lines.append("  - broad_rewrite evidence from static/state:")
        lines.extend(f"    {line}" for line in broad_lines[:20])
    if artifact_items or artifact_lines:
        lines.append("  - artifact_residue evidence from rules/static:")
        lines.extend(f"    {line}" for line in artifact_items[:20])
        lines.extend(f"    {line}" for line in artifact_lines[:20])
    if not broad_lines and not artifact_items and not artifact_lines:
        lines.append("  - no deterministic broad_rewrite or artifact_residue evidence")
    return "\n".join(lines)


def postprocess_trajectory_penalties(
    penalties: dict[str, Any],
    actions: list[dict],
    *,
    static_summary_by_index: dict[int, str],
) -> dict[str, Any]:
    """Clamp trajectory-level penalties to evidence-supported v2 ranges."""
    out = {
        "broad_rewrite": dict((penalties or {}).get("broad_rewrite") or {}),
        "artifact_residue": dict((penalties or {}).get("artifact_residue") or {}),
    }

    broad = out["broad_rewrite"]
    artifact = out["artifact_residue"]

    broad_evidence = _has_broad_rewrite_evidence(actions, static_summary_by_index)
    artifact_evidence = _has_artifact_residue_evidence(actions, static_summary_by_index, artifact)

    broad["severity"] = _clamp_score(broad.get("severity"))
    artifact["severity"] = _clamp_score(artifact.get("severity"))

    if not broad_evidence and broad["severity"] > 0.3:
        broad["severity"] = 0.3
        broad["rationale"] = _append_note(
            broad.get("rationale", ""),
            "Capped because hybrid state/static evidence does not show breadth or repeated churn.",
        )
    broad["present"] = bool(broad.get("present")) and broad["severity"] >= 0.3

    if not artifact_evidence:
        artifact["present"] = False
        artifact["severity"] = 0.0
        artifact["rationale"] = _append_note(
            artifact.get("rationale", ""),
            "Cleared because no artifact-like path or static artifact evidence was found.",
        )
    else:
        artifact["present"] = bool(artifact.get("present")) and artifact["severity"] >= 0.3

    return out


def apply_scope_review_overrides(
    actions: list[dict],
    overrides: list[dict[str, Any]],
    *,
    review_model: str,
    parse_failures: list[str],
    threshold: float = SCOPE_REVIEW_THRESHOLD,
) -> int:
    """Safely apply trajectory-level scope downgrades to action judgments."""
    reviewed_indices = _review_candidate_indices(actions)
    by_action: dict[int, dict[str, Any]] = {}
    for override in overrides:
        idx = override.get("action_index")
        if idx in reviewed_indices:
            by_action[idx] = override

    applied = 0
    for action in actions:
        idx = action.get("action_index")
        if idx not in reviewed_indices:
            continue

        original_scope = action.get("risk_scope")
        meta = {
            "applied": False,
            "original_scope": original_scope,
            "new_scope": original_scope,
            "confidence": None,
            "evidence_type": None,
            "rationale": "",
            "model": review_model,
            "parse_failures": parse_failures,
        }
        override = by_action.get(idx)
        if override:
            confidence = _clamp_score(override.get("confidence"))
            new_scope = override.get("risk_scope")
            meta.update({
                "new_scope": new_scope,
                "confidence": confidence,
                "evidence_type": override.get("evidence_type"),
                "rationale": override.get("rationale", ""),
            })
            if (
                confidence >= threshold
                and new_scope in _review_allowed_targets(original_scope)
                and _scope_review_override_compatible(action, actions, new_scope, override.get("evidence_type"))
                and not _is_rule_final(action)
            ):
                action["risk_scope"] = new_scope
                action["risk_scope_rationale"] = override.get("rationale", "")
                reviewed = postprocess_action(action)
                for key in ("risk_scope", "risk_scope_rationale", "manual_risk_vector",
                            "wrong_abstraction", "action_myopia_score", "is_myopic"):
                    action[key] = reviewed[key]
                meta["applied"] = True
                applied += 1
        action["_scope_review"] = meta

    return applied


def apply_scope_evidence(
    actions: list[dict],
    evidence_items: list[dict[str, Any]],
    *,
    review_model: str,
    parse_failures: list[str],
    threshold: float = SCOPE_EVIDENCE_THRESHOLD,
) -> int:
    """Apply conservative scope downgrades derived from extracted evidence."""
    reviewed_indices = _review_candidate_indices(actions)
    by_action: dict[int, dict[str, Any]] = {}
    for item in evidence_items:
        idx = item.get("action_index")
        if idx in reviewed_indices:
            by_action[idx] = item

    applied = 0
    for action in actions:
        idx = action.get("action_index")
        if idx not in reviewed_indices:
            continue

        original_scope = action.get("risk_scope")
        meta = {
            "applied": False,
            "original_scope": original_scope,
            "new_scope": original_scope,
            "derived_scope": None,
            "confidence": None,
            "evidence": None,
            "rationale": "",
            "model": review_model,
            "parse_failures": parse_failures,
        }
        item = by_action.get(idx)
        if item:
            confidence = _clamp_score(item.get("confidence"))
            derived_scope = derive_scope_from_evidence(
                item, action, actions, threshold=threshold
            )
            meta.update({
                "derived_scope": derived_scope,
                "confidence": confidence,
                "evidence": {
                    "durable_repo_effect": item.get("durable_repo_effect"),
                    "final_task_contribution": item.get("final_task_contribution"),
                    "later_status": item.get("later_status"),
                    "artifact_or_verification": item.get("artifact_or_verification"),
                    "evidence_actions": item.get("evidence_actions") or [],
                },
                "rationale": item.get("rationale", ""),
            })
            if (
                derived_scope in _review_allowed_targets(original_scope)
                and not _is_rule_final(action)
            ):
                action["risk_scope"] = derived_scope
                action["risk_scope_rationale"] = item.get("rationale", "")
                reviewed = postprocess_action(action)
                for key in ("risk_scope", "risk_scope_rationale", "manual_risk_vector",
                            "wrong_abstraction", "action_myopia_score", "is_myopic"):
                    action[key] = reviewed[key]
                meta["applied"] = True
                meta["new_scope"] = derived_scope
                applied += 1
        action["_scope_evidence"] = meta

    return applied


def derive_scope_from_evidence(
    evidence: dict[str, Any],
    action: dict,
    actions: list[dict],
    *,
    threshold: float = SCOPE_EVIDENCE_THRESHOLD,
) -> str | None:
    """Convert extracted evidence into a conservative risk_scope downgrade."""
    confidence = _clamp_score(evidence.get("confidence"))
    if confidence < 0.55:
        return None

    artifact_or_verification = evidence.get("artifact_or_verification")
    durable = evidence.get("durable_repo_effect")
    contribution = evidence.get("final_task_contribution")
    later_status = evidence.get("later_status")
    evidence_actions = evidence.get("evidence_actions") or []

    if later_status == "reverted" and confidence >= 0.85:
        if _has_later_same_file_undo(action, actions) or _has_later_same_file_action(
            action, actions, evidence_actions=evidence_actions
        ):
            return "noise_reverted"

    if later_status == "superseded" and confidence >= 0.55:
        if (
            durable != "yes"
            and contribution != "yes"
            and _has_later_same_file_action(action, actions, evidence_actions=evidence_actions)
        ):
            return "noise_reverted"

    if artifact_or_verification == "yes" and confidence >= 0.85:
        paths = target_paths(action)
        if paths and _all_paths_artifact_or_external(paths):
            return "artifact_only"

    if durable == "no" and confidence >= threshold:
        return "noise_no_effect"

    if contribution == "no" and durable != "yes" and confidence >= 0.9:
        return "noise_no_effect"

    return None


def _review_candidate_indices(actions: list[dict]) -> set[int]:
    return {
        a.get("action_index")
        for a in actions
        if a.get("risk_scope") in SCORABLE_SCOPES and not _is_rule_final(a)
    }


def _review_allowed_targets(original_scope: str | None) -> set[str]:
    if original_scope == "substantive":
        return {"noise_no_effect", "noise_reverted", "temporary_verification", "artifact_only", "uncertain"}
    if original_scope == "uncertain":
        return {"noise_no_effect", "noise_reverted", "temporary_verification", "artifact_only", "uncertain"}
    return set()


def _scope_review_override_compatible(
    action: dict,
    actions: list[dict],
    scope: str | None,
    evidence_type: str | None,
) -> bool:
    if scope == "noise_reverted":
        return evidence_type == "reverted_later" and _has_later_same_file_undo(action, actions)
    if scope in {"artifact_only", "temporary_verification"}:
        return evidence_type == "artifact_or_verification"
    if scope == "noise_no_effect":
        return evidence_type in {"exploratory_or_dead_end", "insufficient_final_contribution"}
    if scope == "uncertain":
        return evidence_type in {"superseded_later", "exploratory_or_dead_end", "insufficient_final_contribution"}
    return False


def _has_later_same_file_undo(action: dict, actions: list[dict]) -> bool:
    idx = action.get("action_index")
    files = set(target_paths(action))
    if idx is None or not files:
        return False
    for later in actions:
        later_idx = later.get("action_index")
        if later_idx is None or later_idx <= idx:
            continue
        op = ((later.get("normalized") or {}).get("operation") or "").lower()
        if op != "undo_edit":
            continue
        if files & set(target_paths(later)):
            return True
    return False


def _has_later_same_file_action(
    action: dict,
    actions: list[dict],
    *,
    evidence_actions: list[int] | None = None,
) -> bool:
    idx = action.get("action_index")
    files = set(target_paths(action))
    allowed = set(evidence_actions or [])
    if idx is None or not files or not allowed:
        return False
    for later in actions:
        later_idx = later.get("action_index")
        if later_idx is None or later_idx <= idx or later_idx not in allowed:
            continue
        if files & set(target_paths(later)):
            return True
    return False


def _is_rule_final(action: dict) -> bool:
    return bool((action.get("_llm_judge") or {}).get("hybrid_rule_final"))


def _final(scope: str, rationale: str) -> RuleDecision:
    return RuleDecision(risk_scope=scope, rationale=rationale, final=True)


def _is_failed_non_verification(op: str, action_class: str, observation: str) -> bool:
    if not observation or not _FAILURE_RE.search(observation):
        return False
    if _SUCCESSFUL_EDIT_RE.search(observation):
        return False
    if action_class in {"install"}:
        return False
    if op in {"str_replace", "insert", "create", "apply", "delete", "move", "copy", "write_file", "touch", "chmod", "mkdir"}:
        return True
    return False


def _is_verification(raw: str, tool: str, op: str, action_class: str) -> bool:
    if action_class == "install" or op == "install":
        return True
    text = f"{tool} {op} {raw}"
    return bool(_VERIFY_RE.search(text))


def _is_read_or_search(raw: str, tool: str, op: str) -> bool:
    if op in {"view", "view_range", "list_dir", "read", "search"}:
        return True
    if tool in {"grep", "rg", "find", "ls"}:
        return True
    if op in {"str_replace", "insert", "create", "apply", "delete", "move", "copy", "write_file", "touch", "chmod", "mkdir"}:
        return False
    return bool(_READ_SEARCH_RE.search(raw))


def _all_paths_artifact_or_external(paths: list[str]) -> bool:
    if not paths:
        return False
    return all(is_artifact_like_path(p) for p in paths)


def _observability_allowed(action: dict) -> bool:
    rv = action.get("manual_risk_vector") or {}
    if not rv.get("observability_loss"):
        return True
    text = "\n".join([
        str(rv.get("rationale") or ""),
        str(action.get("risk_scope_rationale") or ""),
        str((action.get("wrong_abstraction") or {}).get("rationale") or ""),
    ])
    raw = str(action.get("raw_action") or "")
    return bool(_OBSERVABILITY_RE.search(text + "\n" + raw))


def _clamp_score(value) -> float:
    try:
        f = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if f != f:
        return 0.0
    return round(max(0.0, min(1.0, f)), 3)


def _normalize_path(path: str) -> str:
    for prefix in ("/app/", "/workspace/", "/repo/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path.lstrip("/")


def _has_broad_rewrite_evidence(actions: list[dict], static_summary_by_index: dict[int, str]) -> bool:
    paths: list[str] = []
    for action in actions:
        scope = action.get("risk_scope")
        if scope in {"noise_no_effect", "temporary_verification"}:
            continue
        paths.extend(target_paths(action))
    counts = Counter(paths)
    if len(counts) >= 4:
        return True
    if any(n >= 5 for n in counts.values()):
        return True
    return any("broad_rewrite evidence" in s for s in static_summary_by_index.values())


def _has_artifact_residue_evidence(
    actions: list[dict],
    static_summary_by_index: dict[int, str],
    artifact_penalty: dict[str, Any],
) -> bool:
    for action in actions:
        if action.get("risk_scope") == "artifact_only":
            return True
        if any(is_artifact_like_path(p) for p in target_paths(action)):
            return True
    paths = artifact_penalty.get("paths") or []
    if any(is_artifact_like_path(str(p)) for p in paths):
        return True
    return any("artifact_residue evidence" in s for s in static_summary_by_index.values())


def _append_note(text: str, note: str) -> str:
    if not text:
        return note
    if note in text:
        return text
    return f"{text} {note}"
