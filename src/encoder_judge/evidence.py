"""Programmatic evidence extraction for the offline encoder judge.

The evidence table is the shared interface between replay, static analysis,
trajectory state, deterministic rules, and the narrow semantic LLM.  It is not a
replacement for those mechanisms; it is where their outputs are normalized.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge.rules import is_artifact_like_path, target_paths

try:
    from llm_judge.state import TrajectoryState
    from repo_env.metadata import resolve_trajectory_env
    from repo_env.replayer import TrajectoryReplayer
    from static_analysis.base import AnalyzerInput
    from static_analysis.runner import run_analyzers
    from llm_judge.rules import format_static_summary_v2

    _REPLAY_STATIC_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without optional deps.
    _REPLAY_STATIC_AVAILABLE = False


FAILED_RE = re.compile(
    r"(old_str not found|no such file|not found|traceback|failed|"
    r"cannot find|command not found|permission denied|invalid context|"
    r"does not exist|no matches|parameter .* required|"
    r"argument .* expected|error: argument|"
    r"no replacement was performed|syntax error)",
    re.IGNORECASE,
)
EDIT_SUCCESS_RE = re.compile(
    r"(has been edited|has been created|successfully|review the changes)",
    re.IGNORECASE,
)
VERIFY_RE = re.compile(
    r"(\bpytest\b|\bgo test\b|\bnpm test\b|\byarn test\b|\bpnpm test\b|"
    r"\bcargo test\b|\bmvn test\b|\bgradle test\b|\bruff\b|\bmypy\b|"
    r"\bflake8\b|\beslint\b|\btsc\b|\btypecheck\b|\blint\b|"
    r"\bgo build\b|\bnpm run build\b|\byarn build\b|\bmake test\b)",
    re.IGNORECASE,
)
READ_OPS = {"view", "view_range", "list_dir", "read", "search"}
EDIT_OPS = {
    "str_replace",
    "insert",
    "create",
    "apply",
    "delete",
    "move",
    "copy",
    "write_file",
    "touch",
    "chmod",
    "mkdir",
}
SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".rb",
    ".rs",
    ".php",
    ".cs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".scala",
    ".kt",
}
CONFIG_SUFFIXES = {
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
    ".gradle",
}


@dataclass
class PatchSurvival:
    status: str
    confidence: float
    later_same_file_actions: list[int] = field(default_factory=list)
    evidence: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "later_same_file_actions": self.later_same_file_actions,
            "evidence": self.evidence,
        }


@dataclass
class ActionEvidence:
    action_index: int
    action_id: str
    action_kind: str
    target_files: list[str]
    is_effectful: bool
    raw_operation: str
    observation_effect: str
    patch_survival: PatchSurvival
    final_diff_contribution: dict[str, Any]
    artifact_signals: dict[str, Any]
    trajectory_signals: dict[str, Any]
    raw_action: str = ""
    replay_evidence: dict[str, Any] = field(default_factory=dict)
    static_evidence: dict[str, Any] = field(default_factory=dict)
    state_evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "target_files": self.target_files,
            "is_effectful": self.is_effectful,
            "raw_operation": self.raw_operation,
            "observation_effect": self.observation_effect,
            "patch_survival": self.patch_survival.as_dict(),
            "final_diff_contribution": self.final_diff_contribution,
            "artifact_signals": self.artifact_signals,
            "trajectory_signals": self.trajectory_signals,
            "replay_evidence": self.replay_evidence,
            "static_evidence": self.static_evidence,
            "state_evidence": self.state_evidence,
        }


def load_sidecar(template_path: Path, suffix: str) -> dict[str, Any]:
    base = template_path.name.replace(".target.template.json", suffix)
    path = template_path.with_name(base)
    if not path.exists():
        return {}
    return _load_json(path)


def build_evidence_table(
    template: dict[str, Any],
    template_path: Path,
    repo_cache_root: Path | None = None,
) -> list[ActionEvidence]:
    """Build action evidence aligned to the template action_level list."""
    normalized_data = load_sidecar(template_path, ".normalized_actions.json")
    traj_data = load_sidecar(template_path, ".traj.json")
    candidate_data = load_sidecar(template_path, ".candidate.json")
    final_patch = load_final_patch(template_path, candidate_data)
    normalized = normalized_data.get("normalized_actions") or []
    trajectory = traj_data.get("trajectory") or []

    norm_by_index = {
        int(a.get("step_index")): a
        for a in normalized
        if a.get("step_index") is not None
    }
    obs_by_index = {
        int(a.get("step_index")): str(a.get("observation") or "")
        for a in trajectory
        if a.get("step_index") is not None
    }
    template_actions = template.get("action_level") or []
    aligned = [_merge_action(a, norm_by_index.get(int(a.get("action_index", -1)), {})) for a in template_actions]
    for action in aligned:
        idx = int(action.get("action_index", -1))
        action["_observation"] = obs_by_index.get(idx, "")

    replay_static = _build_replay_static_evidence(
        template_path=template_path,
        aligned_actions=aligned,
        repo_cache_root=repo_cache_root,
    )
    path_to_indices: dict[str, list[int]] = {}
    for action in aligned:
        for path in target_paths(action):
            path_to_indices.setdefault(path, []).append(int(action.get("action_index", -1)))

    validation_indices = [
        int(a.get("action_index", -1))
        for a in aligned
        if classify_action_kind(a) == "test" and a.get("action_index") is not None
    ]
    all_paths = list(path_to_indices)
    revisited = [p for p, indices in path_to_indices.items() if len(indices) >= 2]
    lines_by_index = {
        int(a.get("action_index", -1)): len(extract_payload(a).splitlines())
        for a in aligned
        if a.get("action_index") is not None
    }

    evidence: list[ActionEvidence] = []
    for action in aligned:
        idx = int(action.get("action_index", -1))
        paths = target_paths(action)
        observation = obs_by_index.get(idx, "")
        kind = classify_action_kind(action)
        obs_effect = observation_effect(action, observation)
        later_same_file = sorted({
            later
            for path in paths
            for later in path_to_indices.get(path, [])
            if later > idx
        })
        previous_same_file = sorted({
            previous
            for path in paths
            for previous in path_to_indices.get(path, [])
            if previous < idx
        })
        later_undo = [
            later.get("action_index")
            for later in aligned
            if later.get("action_index") is not None
            and int(later["action_index"]) > idx
            and str((later.get("normalized") or {}).get("operation") or "").lower() == "undo_edit"
            and set(paths) & set(target_paths(later))
        ]
        patch = infer_patch_survival(action, kind, obs_effect, later_same_file, aligned, final_patch)
        replay = replay_static.get(idx, {}).get("replay", {})
        static = replay_static.get(idx, {}).get("static", {})
        if replay.get("apply_failed"):
            obs_effect = "failed"
            if patch.status not in {"reverted", "superseded"}:
                patch = PatchSurvival("no_effect", 1.0, later_same_file, replay.get("apply_failure_reason", "Replay failed."))
        artifact_paths = [p for p in paths if is_artifact_like_path(p)]
        final_contribution = infer_final_diff_contribution(
            action=action,
            patch=patch,
            paths=paths,
            final_patch=final_patch,
            later_same_file=later_same_file,
            aligned_actions=aligned,
        )
        final_present = bool(final_contribution.get("present"))
        residual_artifacts = [
            p for p in artifact_paths
            if final_contribution.get("path_presence", {}).get(p)
            or (kind == "artifact" and patch.status not in {"reverted", "no_effect"})
        ]
        static_evidence = _normalize_static_evidence(static)
        static_evidence["wrong_abstraction_hints"] = _wrong_abstraction_hints(
            action=action,
            aligned_actions=aligned,
            idx=idx,
            observation=observation,
            obs_by_index=obs_by_index,
            paths=paths,
        )
        evidence.append(ActionEvidence(
            action_index=idx,
            action_id=str(action.get("action_id") or f"{template_path.parent.name}:{idx}"),
            action_kind=kind,
            target_files=paths,
            is_effectful=bool(action.get("is_effectful", True)),
            raw_operation=str((action.get("normalized") or {}).get("operation") or ""),
            observation_effect=obs_effect,
            patch_survival=patch,
            final_diff_contribution=final_contribution,
            artifact_signals={
                "introduced_artifact_paths": artifact_paths,
                "residual_artifact_paths": residual_artifacts,
            },
            trajectory_signals={
                "same_file_edit_count": max((len(path_to_indices.get(path, [])) for path in paths), default=0),
                "later_same_file_action_count": len(later_same_file),
                "unique_files_touched": len(path_to_indices),
                "cumulative_lines_changed": sum(lines_by_index.values()),
            },
            raw_action=str(action.get("raw_action") or ""),
            replay_evidence={
                "applied": not bool(replay.get("apply_failed")),
                "apply_failed": bool(replay.get("apply_failed")),
                "apply_failure_reason": replay.get("apply_failure_reason", ""),
                "patch_survival": patch.status,
                "patch_survival_confidence": round(patch.confidence, 3),
                "final_diff_contribution": final_present,
                "contributed_files": paths if final_present else [],
                "later_same_file_actions": later_same_file,
                "later_undo_actions": later_undo,
                "replay_fidelity_reduced": bool(replay.get("unreplayed_mutations", 0)),
                "unreplayed_mutations": int(replay.get("unreplayed_mutations", 0) or 0),
                "evidence": patch.evidence,
            },
            static_evidence=static_evidence,
            state_evidence={
                "phase": _phase_for_index(len(evidence), len(aligned)),
                "same_file_chain": {
                    "previous": previous_same_file,
                    "current": idx,
                    "later": later_same_file,
                },
                "artifact_lifecycle": _artifact_lifecycle(idx, paths, aligned, final_present),
                "validation_after": [v for v in validation_indices if v > idx],
                "cumulative_churn": {
                    "files_touched": len(all_paths),
                    "revisited_files": len(revisited),
                    "lines_changed": sum(lines_by_index.values()),
                },
                "static_summary": static.get("summary", ""),
            },
        ))
    return evidence


def _build_replay_static_evidence(
    *,
    template_path: Path,
    aligned_actions: list[dict[str, Any]],
    repo_cache_root: Path | None,
) -> dict[int, dict[str, Any]]:
    if repo_cache_root is None or not _REPLAY_STATIC_AVAILABLE:
        return {}

    out: dict[int, dict[str, Any]] = {}
    state = TrajectoryState(total_actions_in_trajectory=sum(1 for a in aligned_actions if a.get("is_effectful", True)))
    replayer = None
    try:
        env = resolve_trajectory_env(template_path)
        replayer = TrajectoryReplayer(env, cache_root=Path(repo_cache_root))
        replayer.setup()
        for action in aligned_actions:
            idx = int(action.get("action_index", -1))
            if not action.get("is_effectful", True):
                out[idx] = {
                    "replay": {
                        "apply_failed": False,
                        "apply_failure_reason": "",
                        "unreplayed_mutations": 0,
                    },
                    "static": {},
                }
                continue
            snap = replayer.step(action)
            signals: dict[str, Any] = {}
            try:
                signals = run_analyzers(AnalyzerInput(
                    snapshot=snap,
                    action=action,
                    state=state,
                    workdir=replayer.workdir,
                    base_commit=replayer.env.base_commit,
                ))
                summary = format_static_summary_v2(signals, snap.unreplayed_mutations)
            except Exception as exc:  # Static signals are advisory.
                signals = {"error": f"{type(exc).__name__}: {exc}"}
                summary = ""
            out[idx] = {
                "replay": {
                    "apply_failed": snap.apply_failed,
                    "apply_failure_reason": snap.apply_failure_reason,
                    "changed_files": list(snap.changed_files),
                    "cumulative_changed_files": sorted(snap.cumulative_changed_files),
                    "unreplayed_mutations": snap.unreplayed_mutations,
                },
                "static": {
                    "raw_signals": signals,
                    "summary": summary,
                },
            }
            state.update(action)
    except Exception as exc:
        return {"_error": {"replay": {"error": f"{type(exc).__name__}: {exc}"}, "static": {}}}
    finally:
        if replayer is not None:
            try:
                replayer.teardown()
            except Exception:
                pass
    return out


def _normalize_static_evidence(static: dict[str, Any]) -> dict[str, Any]:
    raw = static.get("raw_signals") or {}
    direction = raw.get("direction_lock") or {}
    fragility = raw.get("fragility_patterns") or {}
    lint = raw.get("lint_delta") or {}
    breadth = raw.get("breadth_metrics") or {}
    artifact = raw.get("artifact_scan") or {}
    return {
        "available": bool(raw),
        "syntax_error": direction.get("post_file_parses") is False,
        "wrong_scope_insertion": bool(direction.get("inserted_at_wrong_scope")),
        "silent_except_added": int(fragility.get("silent_excepts") or 0),
        "broad_except_added": int(fragility.get("broad_excepts") or 0),
        "hardcoded_path_added": int(fragility.get("hardcoded_paths") or 0),
        "validation_weakened": False,
        "lint_delta": lint.get("ruff_warnings_delta"),
        "breadth_metrics": {
            "cumulative_files_changed": breadth.get("cumulative_files_changed", 0),
            "cumulative_lines_changed": int(breadth.get("cumulative_lines_added") or 0)
            + int(breadth.get("cumulative_lines_removed") or 0),
            "max_single_file_changed_lines": breadth.get("max_single_file_changed_lines", 0),
        },
        "artifact_scan": {
            "introduced": artifact.get("artifacts_introduced_this_action") or [],
            "still_present": artifact.get("artifacts_still_present") or [],
            "deps_introduced": artifact.get("deps_introduced_this_action") or [],
            "deps_still_present": artifact.get("deps_files_modified_still_present") or [],
        },
        "summary": static.get("summary", ""),
    }


def _phase_for_index(position: int, total: int) -> str:
    if total <= 0:
        return "implementation"
    ratio = position / total
    if ratio < 0.25:
        return "exploration"
    if ratio < 0.75:
        return "implementation"
    return "cleanup"


def _artifact_lifecycle(idx: int, paths: list[str], actions: list[dict[str, Any]], final_present: bool) -> list[dict[str, Any]]:
    artifact_paths = [p for p in paths if is_artifact_like_path(p)]
    lifecycle = []
    for path in artifact_paths:
        related = [
            int(a.get("action_index", -1))
            for a in actions
            if path in target_paths(a) and a.get("action_index") is not None
        ]
        lifecycle.append({
            "path": path,
            "introduced_at": min(related) if related else idx,
            "related_actions": related,
            "residual": final_present,
        })
    return lifecycle


def _wrong_abstraction_hints(
    *,
    action: dict[str, Any],
    aligned_actions: list[dict[str, Any]],
    idx: int,
    observation: str,
    obs_by_index: dict[int, str],
    paths: list[str],
) -> dict[str, Any]:
    """Extract project-agnostic hints for misplaced implementation structure."""
    payload = extract_payload(action)
    old_payload = extract_old_payload(action)
    structural_insert = _looks_structural_payload(payload)
    interface_boundary_change = _looks_interface_boundary_change(old_payload, payload)
    same_payload = _same_payload_actions(action, aligned_actions)
    future_errors = _future_validation_errors(idx, paths, aligned_actions, obs_by_index)
    observation_snippet = ""
    if structural_insert or future_errors or same_payload:
        observation_snippet = _compact_observation_snippet(observation)

    signals: list[str] = []
    if structural_insert:
        signals.append("structural_insert")
    if interface_boundary_change:
        signals.append("interface_boundary_change")
    if same_payload:
        signals.append("duplicate_same_payload")
    if future_errors:
        signals.append("future_validation_error")
    if _observation_shows_structural_misplacement(observation):
        signals.append("observation_structural_misplacement")

    confidence = 0.0
    if "future_validation_error" in signals and (
        "structural_insert" in signals or "observation_structural_misplacement" in signals
    ):
        confidence = 0.75
    elif "duplicate_same_payload" in signals and "structural_insert" in signals:
        confidence = 0.65
    elif "observation_structural_misplacement" in signals:
        confidence = 0.65
    elif "interface_boundary_change" in signals:
        confidence = 0.55
    elif signals:
        confidence = 0.4

    return {
        "signals": signals,
        "confidence": round(confidence, 3),
        "structural_insert": structural_insert,
        "duplicate_same_payload_actions": same_payload,
        "future_validation_errors": future_errors,
        "observation_snippet": observation_snippet,
        "old_payload_snippet": _truncate(old_payload.strip(), 600),
        "payload_snippet": _truncate(payload.strip(), 600),
    }


def classify_action_kind(action: dict[str, Any]) -> str:
    normalized = action.get("normalized") or {}
    op = str(normalized.get("operation") or "").lower()
    tool = str(normalized.get("tool") or "").lower()
    action_class = str(normalized.get("action_class") or "").lower()
    raw = str(action.get("raw_action") or "")
    paths = target_paths(action)
    primary = primary_target_path(action)

    if op == "undo_edit":
        return "undo"
    if op in READ_OPS or action_class in {"read", "search"} or tool in {"grep", "rg", "find", "ls"}:
        return "read"
    artifact_paths = [p for p in paths if is_encoder_artifact_path(p)]
    if primary and is_encoder_artifact_path(primary):
        return "artifact"
    if paths and all(is_encoder_artifact_path(p) for p in paths):
        return "artifact"
    if artifact_paths and op in {"create", "str_replace", "insert", "write_file", "delete", "chmod", "mkdir", "copy", "move"}:
        return "artifact"
    if action_class == "install" or op == "install":
        return "install"
    if VERIFY_RE.search(raw) or action_class in {"test", "build", "lint", "typecheck"}:
        return "test"
    if op in EDIT_OPS or action_class in {"patch", "edit", "write"}:
        return "edit"
    return "unknown"


def observation_effect(action: dict[str, Any], observation: str) -> str:
    if not action.get("is_effectful", True):
        return "no_effect"
    if observation and FAILED_RE.search(observation) and not EDIT_SUCCESS_RE.search(observation):
        return "failed"
    if EDIT_SUCCESS_RE.search(observation):
        return "success"
    if observation.strip() and classify_action_kind(action) in {"edit", "artifact"}:
        return "no_effect"
    return "unknown"


def infer_patch_survival(
    action: dict[str, Any],
    kind: str,
    obs_effect: str,
    later_same_file: list[int],
    aligned_actions: list[dict[str, Any]],
    final_patch: str = "",
) -> PatchSurvival:
    if kind == "read" or not action.get("is_effectful", True):
        return PatchSurvival("no_effect", 1.0, later_same_file, "Read/search/list action has no repository mutation.")
    if obs_effect in {"failed", "no_effect"}:
        return PatchSurvival("no_effect", 1.0, later_same_file, "Observed no successful repository mutation.")
    if kind in {"test", "install"}:
        return PatchSurvival("no_effect", 0.9, later_same_file, "Verification/environment action is not a durable source edit.")
    if kind == "undo":
        return PatchSurvival("no_effect", 1.0, later_same_file, "Undo action itself has no independent final contribution.")

    if _has_later_undo(action, aligned_actions):
        return PatchSurvival("reverted", 0.95, later_same_file, "Later same-file undo/revert action removes this edit.")
    if _has_later_delete_or_rm(action, aligned_actions):
        return PatchSurvival("reverted", 0.9, later_same_file, "Later same-path delete/remove action removes this edit.")
    if _has_later_overwrite(action, aligned_actions):
        return PatchSurvival("superseded", 0.9, later_same_file, "Later same-path create/delete/write action supersedes this edit.")

    final_match = final_patch_contribution(action, final_patch)
    if final_match:
        if obs_effect == "success":
            return PatchSurvival("survived", 0.9, later_same_file, final_match)
        return PatchSurvival(
            "partial",
            0.55,
            later_same_file,
            f"{final_match} The current action has no explicit success observation.",
        )
    repeated_payload_actions = _earlier_same_payload_actions(action, aligned_actions)
    if repeated_payload_actions:
        return PatchSurvival(
            "superseded",
            0.9,
            repeated_payload_actions,
            "Earlier same-file edit already introduced the same payload; this repeated attempt is treated as no final contribution.",
        )
    if later_same_file and kind == "edit":
        if obs_effect == "unknown":
            return PatchSurvival(
                "partial",
                0.45,
                later_same_file,
                "Target-related edit is unconfirmed and followed by later same-file edits.",
            )
        if final_patch:
            return PatchSurvival("superseded", 0.75, later_same_file, "Final patch does not contain this action's concrete inserted text.")
        return PatchSurvival("partial", 0.45, later_same_file, "Later same-file edit exists; without final patch, contribution is uncertain.")
    op = str((action.get("normalized") or {}).get("operation") or "").lower()
    if op == "delete" and final_patch:
        return PatchSurvival("no_effect", 0.8, later_same_file, "Delete action has no matching final patch contribution.")
    if kind == "artifact":
        return PatchSurvival("survived", 0.7, later_same_file, "Artifact-like path was touched; final residue is approximated from action evidence.")
    if kind == "edit" and obs_effect in {"success", "unknown"}:
        return PatchSurvival("survived", 0.7, later_same_file, "No later same-file overwrite was observed in the available trajectory evidence.")
    return PatchSurvival("unknown", 0.0, later_same_file, "Insufficient evidence.")


def infer_final_diff_contribution(
    *,
    action: dict[str, Any],
    patch: PatchSurvival,
    paths: list[str],
    final_patch: str,
    later_same_file: list[int],
    aligned_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Infer whether an action has a durable semantic contribution.

    `patch_survival` tracks exact edit survival.  This function tracks broader
    action contribution: final path presence, symbol/payload survival, and
    same-file refinement.  This avoids treating every later same-file rewrite as
    no contribution.
    """
    path_presence = {p: bool(final_patch and _path_in_patch(p, final_patch)) for p in paths}
    exact = final_patch_contribution(action, final_patch)
    symbols = changed_symbols(extract_payload(action))
    symbol_hits = [
        symbol for symbol in symbols
        if final_patch and re.search(rf"\b{re.escape(symbol)}\b", final_patch)
    ]
    artifact_paths = [p for p in paths if is_encoder_artifact_path(p)]
    source_paths = [p for p in paths if is_source_test_or_config(p) and p not in artifact_paths]

    semantic_survival = "unknown"
    present = False
    evidence = ""
    independent = True
    if patch.status == "no_effect" and source_paths and (exact or symbol_hits or any(path_presence.values())):
        present = True
        independent = False
        if exact:
            semantic_survival = "exact_text_survived"
            evidence = (
                f"{exact} Observation/replay reported no effect, so independent ownership "
                "requires trajectory review."
            )
        elif symbol_hits:
            semantic_survival = "symbol_or_hunk_survived"
            evidence = (
                f"Changed symbol(s) appear in final patch despite no-effect observation: "
                f"{symbol_hits[:5]}."
            )
        else:
            semantic_survival = "path_survived"
            evidence = "Target source/test/config path appears in final patch despite no-effect observation."
    elif patch.status == "no_effect":
        semantic_survival = "no_final_evidence"
        evidence = _final_contribution_evidence(patch, paths)
    elif exact and later_same_file and patch.status in {"superseded", "reverted"}:
        semantic_survival = "later_refined"
        present = True
        independent = False
        evidence = (
            f"{exact} Later same-file actions also changed the target, so independent "
            "contribution requires trajectory review."
        )
    elif exact:
        semantic_survival = "exact_text_survived"
        present = True
        evidence = exact
    elif symbol_hits:
        semantic_survival = "symbol_or_hunk_survived"
        present = True
        evidence = f"Changed symbol(s) appear in final patch: {symbol_hits[:5]}."
    elif final_patch and any(path_presence.values()):
        if later_same_file:
            semantic_survival = "later_refined"
            independent = False
            evidence = "Target path appears in final patch after later same-file refinement."
        else:
            semantic_survival = "path_survived"
            evidence = "Target path appears in final patch."
        present = True
    elif patch.status == "reverted" or _has_later_delete_or_rm(action, aligned_actions):
        semantic_survival = "fully_removed"
        evidence = patch.evidence or "Later undo/delete evidence indicates removal."
    elif patch.status == "superseded":
        semantic_survival = "superseded_uncertain"
        evidence = patch.evidence or "Exact payload was superseded by later same-file action."
    elif patch.status in {"survived", "partial"} and not final_patch:
        semantic_survival = "path_survived"
        present = True
        evidence = _final_contribution_evidence(patch, paths)
    else:
        semantic_survival = "no_final_evidence"
        evidence = _final_contribution_evidence(patch, paths)

    return {
        "present": present,
        "files": paths if present else [],
        "evidence": evidence,
        "semantic_survival": semantic_survival,
        "exact_text_survived": bool(exact),
        "symbol_hits": symbol_hits,
        "path_presence": path_presence,
        "later_same_file_actions": later_same_file,
        "source_paths": source_paths,
        "artifact_paths": artifact_paths,
        "independent": independent and present,
    }


def is_source_test_or_config(path: str) -> bool:
    rel = path.lower()
    suffix = Path(rel).suffix
    if suffix in SOURCE_SUFFIXES or suffix in CONFIG_SUFFIXES:
        return True
    return any(part in rel for part in ("/test/", "/tests/", "/spec/", "/specs/"))


def is_encoder_artifact_path(path: str) -> bool:
    rel = path.lstrip("/")
    name = Path(rel).name.lower()
    if is_artifact_like_path(rel):
        return True
    parts = [p.lower() for p in Path(rel).parts]
    if any(p in {"tmp", "temp", "scratch", "repro", "debug"} for p in parts):
        return True
    if name.startswith(("verify_", "validate_", "validation_", "check_", "debug_")):
        return True
    if name.startswith("run_") and "test" in name:
        return True
    if name in {"test.sh", "check.sh", "debug.sh", "verify.sh", "validate.sh"}:
        return True
    if name.startswith(("test_config", "debug_config", "repro_config", "local_config")):
        return True
    if name.startswith(("simple_test", "performance_comparison", "consolidation_summary")):
        return True
    if name.endswith((".bak", ".orig", ".tmp", ".log", ".db", ".sqlite")):
        return True
    if name in {"consolidation_summary.md", "implementation_complete.md", "validation.md"}:
        return True
    return False


def primary_target_path(action: dict[str, Any]) -> str:
    normalized = action.get("normalized") or {}
    for key in ("target_file", "path", "target_path"):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            return _normalize_path(value)
    paths = target_paths(action)
    return paths[0] if paths else ""


def final_patch_contribution(action: dict[str, Any], final_patch: str) -> str:
    if not final_patch:
        return ""
    primary = primary_target_path(action)
    op = str((action.get("normalized") or {}).get("operation") or "").lower()
    if primary and _path_in_patch(primary, final_patch) and op == "create":
        return f"Created path appears in final patch: {primary}."
    payload = extract_payload(action)
    snippets = meaningful_snippets(payload)
    for snippet in snippets:
        if snippet in final_patch:
            return "Inserted text appears in final patch."
    return ""


def changed_symbols(payload: str) -> list[str]:
    symbols: list[str] = []
    patterns = [
        r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][\w]*)",
        r"^\s*(?:def|func|class|interface|type|struct|enum)\s+([A-Za-z_][\w]*)",
        r"^\s*(?:const|let|var)\s+([A-Za-z_][\w]*)\s*[=:]",
    ]
    for line in payload.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                symbols.append(match.group(1))
    return list(dict.fromkeys(symbols))


def extract_payload(action: dict[str, Any]) -> str:
    raw = str(action.get("raw_action") or "")
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()
    for flag in ("--new_str", "--file_text"):
        if flag in parts:
            idx = parts.index(flag)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def extract_old_payload(action: dict[str, Any]) -> str:
    raw = str(action.get("raw_action") or "")
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()
    for flag in ("--old_str", "--old_text"):
        if flag in parts:
            idx = parts.index(flag)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def meaningful_snippets(payload: str) -> list[str]:
    out: list[str] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if len(stripped) < 8:
            continue
        if stripped in {"}", "{", "else:", "return nil", "return None"}:
            continue
        out.append(stripped)
    if not out and len(payload.strip()) >= 12:
        out.append(payload.strip())
    return out[:12]


def load_final_patch(template_path: Path, candidate: dict[str, Any]) -> str:
    model = candidate.get("model")
    sample_id = candidate.get("sample_id")
    candidates: list[Path] = []
    if model and sample_id:
        candidates.append(template_path.parents[3] / str(model) / "eval" / str(sample_id) / "_patch.diff")
        candidates.append(template_path.parents[2] / str(model) / "eval" / str(sample_id) / "_patch.diff")
        candidates.append(Path("../data") / str(model) / "eval" / str(sample_id) / "_patch.diff")
        candidates.append(Path("data") / str(model) / "eval" / str(sample_id) / "_patch.diff")
    source_traj = candidate.get("source_traj")
    if source_traj:
        source = Path(str(source_traj))
        candidates.append(source.parents[2] / "eval" / source.parent.name / "_patch.diff")
    for path in candidates:
        if path.exists():
            return path.read_text(errors="ignore")
    return ""


def _merge_action(template_action: dict[str, Any], normalized: dict[str, Any]) -> dict[str, Any]:
    merged = dict(template_action)
    existing = dict(merged.get("normalized") or {})
    if normalized:
        existing.update({
            k: v for k, v in normalized.items()
            if k not in {"step_index", "raw_action"} and v is not None
        })
        merged["normalized"] = existing
        merged["raw_action"] = merged.get("raw_action") or normalized.get("raw_action")
        if "is_effectful" not in merged:
            merged["is_effectful"] = normalized.get("is_effectful")
    return merged


def _has_later_undo(action: dict[str, Any], aligned_actions: list[dict[str, Any]]) -> bool:
    idx = action.get("action_index")
    paths = set(target_paths(action))
    if idx is None or not paths:
        return False
    for later in aligned_actions:
        if later.get("action_index") is None or int(later["action_index"]) <= int(idx):
            continue
        if not _later_action_has_effect(later):
            continue
        op = str((later.get("normalized") or {}).get("operation") or "").lower()
        raw = str(later.get("raw_action") or "").lower()
        if op == "undo_edit" or "undo_edit" in raw:
            if paths & set(target_paths(later)):
                return True
    return False


def _has_later_overwrite(action: dict[str, Any], aligned_actions: list[dict[str, Any]]) -> bool:
    idx = action.get("action_index")
    paths = set(target_paths(action))
    if idx is None or not paths:
        return False
    overwrite_ops = {"create", "delete", "write_file", "move", "copy"}
    for later in aligned_actions:
        later_idx = later.get("action_index")
        if later_idx is None or int(later_idx) <= int(idx):
            continue
        if not (paths & set(target_paths(later))):
            continue
        if not _later_action_has_effect(later):
            continue
        op = str((later.get("normalized") or {}).get("operation") or "").lower()
        if op in overwrite_ops:
            return True
    return False


def _has_later_delete_or_rm(action: dict[str, Any], aligned_actions: list[dict[str, Any]]) -> bool:
    idx = action.get("action_index")
    paths = set(target_paths(action))
    if idx is None or not paths:
        return False
    op = str((action.get("normalized") or {}).get("operation") or "").lower()
    if op != "create":
        return False
    for later in aligned_actions:
        later_idx = later.get("action_index")
        if later_idx is None or int(later_idx) <= int(idx):
            continue
        if not (paths & set(target_paths(later))):
            continue
        if not _later_action_has_effect(later):
            continue
        later_op = str((later.get("normalized") or {}).get("operation") or "").lower()
        later_raw = str(later.get("raw_action") or "").lower()
        if later_op == "delete" or re.search(r"\brm\s+-[^\n;]*f\b|\brm\s+", later_raw):
            return True
    return False


def _later_action_has_effect(action: dict[str, Any]) -> bool:
    observation = str(action.get("_observation") or "")
    effect = observation_effect(action, observation)
    return effect not in {"failed", "no_effect"}


def _earlier_same_payload_actions(action: dict[str, Any], aligned_actions: list[dict[str, Any]]) -> list[int]:
    idx = action.get("action_index")
    paths = set(target_paths(action))
    payload = _canonical_payload(extract_payload(action))
    if idx is None or not paths or not payload:
        return []
    matches: list[int] = []
    for later in aligned_actions:
        later_idx = later.get("action_index")
        if later_idx is None or int(later_idx) >= int(idx):
            continue
        if not (paths & set(target_paths(later))):
            continue
        later_payload = _canonical_payload(extract_payload(later))
        if payload == later_payload:
            matches.append(int(later_idx))
    return matches


def _same_payload_actions(action: dict[str, Any], aligned_actions: list[dict[str, Any]]) -> list[int]:
    idx = action.get("action_index")
    paths = set(target_paths(action))
    payload = _canonical_payload(extract_payload(action))
    if idx is None or not paths or not payload:
        return []
    matches: list[int] = []
    for other in aligned_actions:
        other_idx = other.get("action_index")
        if other_idx is None or int(other_idx) == int(idx):
            continue
        if not (paths & set(target_paths(other))):
            continue
        if payload == _canonical_payload(extract_payload(other)):
            matches.append(int(other_idx))
    return sorted(matches)


def _canonical_payload(payload: str) -> str:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    normalized = "\n".join(lines)
    if len(normalized) < 40:
        return ""
    return normalized


def _looks_structural_payload(payload: str) -> bool:
    text = payload.strip()
    if not text:
        return False
    structural_re = re.compile(
        r"(?m)^\s*(import\b|type\b|func\b|class\b|interface\b|struct\b|enum\b|const\b|var\b|def\b)"
    )
    if structural_re.search(text):
        return True
    if re.search(r'(?m)^\s*"[^"]+"\s*:\s*[{[]?\s*$', text) and ("type" in text or "properties" in text):
        return True
    return False


def _looks_interface_boundary_change(old_payload: str, new_payload: str) -> bool:
    if not old_payload or not new_payload:
        return False
    old = old_payload.strip()
    new = new_payload.strip()
    combined = f"{old}\n{new}"
    if re.search(r"\bPromise\s*<|\bawait\b|\basync\b", new) and not re.search(r"\bPromise\s*<|\bawait\b|\basync\b", old):
        return True
    if re.search(r"\b(type|interface|struct|class|func|def|private|public|readonly)\b", combined):
        return True
    if re.search(r"\b(cache|session|credential|provider|query|params?|config|client)\b", combined, re.I):
        return True
    if re.search(r"append\s*\(\s*\(|\[\s*[\"'][^\"']+[\"']\s*\]|[\"'][A-Za-z0-9_.-]+[\"']\s*:", combined):
        return True
    return False


def _future_validation_errors(
    idx: int,
    paths: list[str],
    aligned_actions: list[dict[str, Any]],
    obs_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    if not paths:
        return []
    names = {Path(p).name for p in paths}
    rels = {p.lstrip("/") for p in paths}
    out: list[dict[str, Any]] = []
    for later in aligned_actions:
        later_idx = later.get("action_index")
        if later_idx is None or int(later_idx) <= idx or int(later_idx) > idx + 12:
            continue
        if classify_action_kind(later) != "test":
            continue
        obs = obs_by_index.get(int(later_idx), "")
        if not re.search(r"(syntax error|imports must appear|undefined|redeclaration|duplicate|invalid|parse error)", obs, re.I):
            continue
        if not any(name and name in obs for name in names) and not any(rel and rel in obs for rel in rels):
            continue
        out.append({
            "action_index": int(later_idx),
            "observation": _truncate(obs.strip(), 700),
        })
        if len(out) >= 3:
            break
    return out


def _observation_shows_structural_misplacement(observation: str) -> bool:
    if not observation:
        return False
    if re.search(r"syntax error|imports must appear|expected declaration", observation, re.I):
        return True
    if re.search(r"(?s)import [^\n]+\n\s*type\b|type\b.+\n\s*import\b", observation):
        return True
    if len(re.findall(r"(?m)^\s*\d+\s+import\b", observation)) >= 2:
        return True
    return False


def _compact_observation_snippet(observation: str) -> str:
    if not observation:
        return ""
    lines = [line for line in observation.splitlines() if line.strip()]
    focused = [
        line for line in lines
        if re.search(r"(has been edited|syntax error|imports must appear|type\b|import\b|func\b|class\b|struct\b|enum\b|duplicate|redeclaration)", line, re.I)
    ]
    if not focused:
        focused = lines[:20]
    return _truncate("\n".join(focused[:30]), 1200)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _final_contribution_evidence(patch: PatchSurvival, paths: list[str]) -> str:
    if patch.status in {"survived", "partial"}:
        return f"{patch.status} patch evidence for {paths}."
    return f"No final contribution inferred: {patch.status}."


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text())


def _path_in_patch(path: str, patch: str) -> bool:
    rel = _normalize_path(path)
    return (
        f"diff --git a/{rel} b/{rel}\n" in patch
        or f"+++ b/{rel}\n" in patch
        or f"--- a/{rel}\n" in patch
    )


def _normalize_path(path: str) -> str:
    for prefix in ("/app/", "/workspace/", "/repo/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path.lstrip("/")
