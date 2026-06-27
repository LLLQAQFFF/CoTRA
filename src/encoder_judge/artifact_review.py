"""Dedicated artifact_residue review for the encoder judge.

This stage is intentionally narrow: it reviews only the trajectory-level
artifact_residue penalty using lifecycle evidence, then re-derives the
trajectory myopia score and engineering scorecard. It runs posthoc over an
existing prediction directory so the action-level pipeline is not re-run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge import config, derive
from llm_judge.client import JudgeClient
from llm_judge.cost import CostTracker, format_summary
from llm_judge.parser import _extract_json

from encoder_judge.evidence import (
    ActionEvidence,
    build_evidence_table,
    load_final_patch,
    load_sidecar,
)
from encoder_judge.formal_output import build_trajectory_engineering_scorecard

MAX_CANDIDATE_PATHS = 24
MAX_TIMELINE_EVENTS_PER_PATH = 10
MAX_SNIPPET_CHARS = 280

ARTIFACT_TYPES = {
    "repro_script",
    "debug_file",
    "test_artifact",
    "generated_output",
    "local_config",
    "temp_file",
    "other",
}


@dataclass
class ArtifactReviewResult:
    present: bool | None = None
    severity: float | None = None
    paths: list[str] = field(default_factory=list)
    artifact_types: list[str] = field(default_factory=list)
    rationale: str = ""
    confidence: float | None = None
    parse_failed: bool = False
    parse_failures: list[str] = field(default_factory=list)
    raw: str = ""


def build_artifact_packet(
    *,
    template_path: Path,
    evidence: list[ActionEvidence],
    deterministic_penalty: dict[str, Any],
    task_description: str,
) -> dict[str, Any]:
    """Build a cost-bounded lifecycle packet for artifact-only review."""
    candidate = load_sidecar(template_path, ".candidate.json")
    final_patch = load_final_patch(template_path, candidate)
    final_files = _final_patch_files(final_patch)

    candidates: list[str] = []
    for path in deterministic_penalty.get("paths") or []:
        if path not in candidates:
            candidates.append(path)
    for ev in evidence:
        signals = ev.artifact_signals or {}
        for path in list(signals.get("residual_artifact_paths") or []) + list(
            signals.get("introduced_artifact_paths") or []
        ):
            if path not in candidates:
                candidates.append(path)
    candidates = candidates[:MAX_CANDIDATE_PATHS]

    traj = load_sidecar(template_path, ".traj.json")
    obs_by_index = {
        int(item.get("step_index")): str(item.get("observation") or "")
        for item in (traj.get("trajectory") or [])
        if item.get("step_index") is not None
    }

    candidates = [p for p in candidates if not _is_tmp_path(p)]
    created_paths: list[str] = []
    for ev in evidence:
        op = str(ev.raw_operation or "").lower()
        if op not in {"create", "write_file", "touch"}:
            continue
        if ev.observation_effect == "failed":
            continue
        for path in ev.target_files or []:
            if path not in created_paths and path not in candidates and not _is_tmp_path(path):
                created_paths.append(path)
    created_paths = created_paths[:MAX_CANDIDATE_PATHS]

    all_raw_actions = [ev.raw_action for ev in evidence if ev.raw_action]
    path_entries = []
    for path in candidates + created_paths:
        events = []
        for ev in evidence:
            if path not in (ev.target_files or []):
                continue
            events.append({
                "action_index": ev.action_index,
                "operation": ev.raw_operation or ev.action_kind,
                "observation_effect": ev.observation_effect,
                "raw_action": ev.raw_action[:MAX_SNIPPET_CHARS],
                "observation": obs_by_index.get(ev.action_index, "")[:MAX_SNIPPET_CHARS],
            })
        deleted_later = any(
            "delete" in str(e.get("operation") or "").lower() or "rm " in str(e.get("raw_action") or "")
            for e in events
        ) or _deleted_in_raw_actions(path, all_raw_actions)
        path_entries.append({
            "path": path,
            "in_final_diff": path in final_files,
            "deleted_later": deleted_later,
            "artifact_like_name": path in candidates,
            "timeline": events[:MAX_TIMELINE_EVENTS_PER_PATH],
        })

    return {
        "task_description": task_description,
        "deterministic_penalty": {
            "present": bool(deterministic_penalty.get("present")),
            "severity": deterministic_penalty.get("severity"),
            "rationale": str(deterministic_penalty.get("rationale") or ""),
        },
        "candidate_paths": path_entries,
        "final_diff_files": final_files[:60],
        "final_diff_available": bool(final_patch),
    }


def review_artifact_residue(
    *,
    template_path: Path,
    evidence: list[ActionEvidence],
    deterministic_penalty: dict[str, Any],
    task_description: str,
    client: JudgeClient,
    model: str,
    tracker: CostTracker | None = None,
) -> tuple[ArtifactReviewResult, dict[str, Any]]:
    packet = build_artifact_packet(
        template_path=template_path,
        evidence=evidence,
        deterministic_penalty=deterministic_penalty,
        task_description=task_description,
    )
    meta = {
        "candidates": len(packet["candidate_paths"]),
        "candidate_paths": [e["path"] for e in packet["candidate_paths"]]
        + list(packet["final_diff_files"]),
        "packet_chars": len(json.dumps(packet, ensure_ascii=False)),
    }
    call = client.call(
        model=model,
        system=_system_prompt(),
        user=_user_prompt(packet),
        max_tokens=config.MAX_TOKENS_TRAJECTORY,
        temperature=0.0,
    )
    if tracker is not None:
        tracker.record(call)
    result = parse_artifact_review(call.text)
    result.raw = call.text
    return result, meta


def parse_artifact_review(text: str) -> ArtifactReviewResult:
    data = _extract_json(text)
    if data is None:
        return ArtifactReviewResult(parse_failed=True, parse_failures=["json"], raw=text)
    failures: list[str] = []
    present = data.get("present")
    if not isinstance(present, bool):
        failures.append("present")
        present = None
    severity = _score(data.get("severity"))
    if severity is None:
        failures.append("severity")
    paths = [str(p) for p in data.get("paths") or [] if isinstance(p, str)]
    types = [str(t) for t in data.get("artifact_types") or [] if str(t) in ARTIFACT_TYPES]
    return ArtifactReviewResult(
        present=present,
        severity=severity,
        paths=paths,
        artifact_types=types,
        rationale=str(data.get("rationale") or ""),
        confidence=_score(data.get("confidence")),
        parse_failed=bool(failures),
        parse_failures=failures,
    )


def apply_artifact_review(
    penalties: dict[str, Any],
    result: ArtifactReviewResult,
    packet_candidates: list[str],
) -> dict[str, Any]:
    """Patch artifact_residue in place; returns application metadata."""
    meta = {"applied": False, "reason": ""}
    if result.parse_failed or result.present is None or result.severity is None:
        meta["reason"] = "parse_failed"
        return meta
    artifact = dict(penalties.get("artifact_residue") or {})
    allowed = set(packet_candidates)
    reviewed_paths = [p for p in result.paths if p in allowed]
    if result.present and not reviewed_paths:
        reviewed_paths = list(artifact.get("paths") or [])
    artifact["present"] = result.present
    artifact["severity"] = result.severity if result.present else 0.0
    artifact["paths"] = reviewed_paths if result.present else []
    if result.artifact_types:
        artifact["artifact_types"] = result.artifact_types if result.present else []
    artifact["rationale"] = result.rationale or artifact.get("rationale") or ""
    artifact["_artifact_review"] = {
        "applied": True,
        "deterministic_present": bool((penalties.get("artifact_residue") or {}).get("present")),
        "deterministic_severity": (penalties.get("artifact_residue") or {}).get("severity"),
        "confidence": result.confidence,
    }
    penalties["artifact_residue"] = artifact
    meta.update({"applied": True, "present": result.present, "severity": artifact["severity"]})
    return meta


def _system_prompt() -> str:
    return (
        "You audit coding-agent trajectories. You answer one narrow question: "
        "which files left in the final repository state are residual non-product "
        "artifacts (repro scripts, debug files, generated outputs, temp or local "
        "config files) rather than intended product/test/config changes. "
        "Judge only from the provided lifecycle evidence. Return valid JSON only."
    )


def _user_prompt(packet: dict[str, Any]) -> str:
    return f"""
Decide whether this trajectory leaves residual artifacts in the final repository state.

Rules:
- A path remains in the final workspace state if it was successfully created and never explicitly deleted later (deleted_later=false). The final submitted diff usually EXCLUDES untracked scratch files, so in_final_diff=false does NOT mean the file was cleaned up; only an explicit later delete/undo removes it.
- in_final_diff=true means the file additionally pollutes the submitted patch itself; treat that as more severe.
- Intended product code and legitimate config changes are NOT artifacts. Self-authored ad-hoc repro/verify/debug scripts and one-off test files written only to check this task ARE artifacts when they remain, even if named like tests.
- Files under /tmp or used only in the external environment are NOT repository artifacts; only files in the repository working tree count.
- deleted_later=true means the file was cleaned up before submission; do not flag it.
- severity guidance: 0.0 none; ~0.3 one small leftover file; ~0.5-0.6 several leftover files or a polluting debug/repro file; ~0.8-0.9 large or widespread pollution of the repository.

Return exactly this JSON shape:
{{
  "present": false,
  "severity": 0.0,
  "paths": ["only paths listed in candidate_paths or final_diff_files"],
  "artifact_types": ["repro_script|debug_file|test_artifact|generated_output|local_config|temp_file|other"],
  "rationale": "at most 25 words, cite path evidence",
  "confidence": 0.0
}}

Evidence packet:
{json.dumps(packet, ensure_ascii=False, indent=1)}
""".strip()


def _is_tmp_path(path: str) -> bool:
    return path.startswith(("/tmp/", "tmp/")) or "/tmp/" in path


_RM_ARGS_RE = re.compile(r"\b(?:rm|unlink)\b((?:\s+-{1,2}\w+)*)((?:\s+[^\s;|&><]+)+)")


def _deleted_in_raw_actions(path: str, raw_actions: list[str]) -> bool:
    """Detect bash-style deletion of the file anywhere in the trajectory.

    Handles glob deletions such as `rm -f test_*.py`, which never mention the
    exact basename.
    """
    basename = path.rsplit("/", 1)[-1]
    if not basename or len(basename) < 4:
        return False
    for text in raw_actions:
        for match in _RM_ARGS_RE.finditer(text):
            for arg in match.group(2).split():
                arg_base = arg.rstrip("'\"").lstrip("'\"").rsplit("/", 1)[-1]
                if not arg_base:
                    continue
                if fnmatch.fnmatch(basename, arg_base):
                    return True
    return False


def _final_patch_files(patch: str) -> list[str]:
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git ") and " b/" in line:
            path = line.split(" b/")[-1].strip()
            if path and path not in files:
                files.append(path)
    return files


def _score(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return round(max(0.0, min(1.0, f)), 3)


def _task_description(template: dict[str, Any], template_path: Path) -> str:
    traj = load_sidecar(template_path, ".traj.json")
    desc = (traj.get("metadata") or {}).get("task_description")
    if desc:
        return str(desc)[:2000]
    return str((template.get("trajectory_meta") or {}).get("sample_id") or "")


def run_posthoc(args: argparse.Namespace) -> int:
    base_root = Path(args.base_root)
    output_root = Path(args.output_root)
    repo_cache = Path(args.repo_cache) if args.repo_cache else None
    model = args.judge_model or config.get_routing().for_family(None)
    client = JudgeClient(
        request_timeout_seconds=config.ENCODER_REQUEST_TIMEOUT_SECONDS,
        retry_max_attempts=config.ENCODER_RETRY_MAX_ATTEMPTS,
        retry_on_empty=True,
    )
    tracker = CostTracker()
    failures: list[str] = []
    processed = 0
    for set_dir in [Path(p) for p in args.set_dirs]:
        for template_path in sorted(set_dir.glob("*/*.target.template.json")):
            if args.limit is not None and processed >= args.limit:
                break
            model_name = template_path.name.replace(".target.template.json", "")
            base_path = (
                base_root / set_dir.name / template_path.parent.name
                / f"{model_name}{args.base_suffix}"
            )
            out_path = (
                output_root / set_dir.name / template_path.parent.name
                / f"{model_name}.{args.tag}.encoder_pre_label.json"
            )
            if args.skip_existing and out_path.exists():
                continue
            if not base_path.exists():
                failures.append(f"missing base: {base_path}")
                continue
            try:
                prediction = json.loads(base_path.read_text())
                template = json.loads(template_path.read_text())
                evidence = build_evidence_table(template, template_path, repo_cache_root=repo_cache)
                actions = prediction.get("action_level") or []
                trajectory = dict(prediction.get("trajectory_level") or {})
                penalties = dict(trajectory.get("trajectory_penalties") or {})
                deterministic = dict(penalties.get("artifact_residue") or {})
                result, packet_meta = review_artifact_residue(
                    template_path=template_path,
                    evidence=evidence,
                    deterministic_penalty=deterministic,
                    task_description=_task_description(template, template_path),
                    client=client,
                    model=model,
                    tracker=tracker,
                )
                applied_meta = apply_artifact_review(
                    penalties,
                    result,
                    packet_meta["candidate_paths"],
                )
                trajectory["trajectory_penalties"] = penalties
                trajectory["trajectory_myopia_score"] = derive.derive_trajectory_myopia_score(
                    actions, penalties
                )
                trajectory["trajectory_engineering_scorecard"] = build_trajectory_engineering_scorecard(
                    actions,
                    evidence,
                    penalties,
                    trajectory["trajectory_myopia_score"],
                )
                prediction["trajectory_level"] = trajectory
                run_meta = dict(prediction.get("_run_meta") or {})
                run_meta["artifact_review"] = {
                    "judge_model": model,
                    "generated_at": dt.datetime.now(dt.UTC).isoformat(),
                    "base_prediction": str(base_path),
                    **packet_meta,
                    **applied_meta,
                    "parse_failures": result.parse_failures,
                }
                run_meta["artifact_review_cost"] = tracker.snapshot()["totals"]
                prediction["_run_meta"] = run_meta
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2))
                processed += 1
                print(
                    f"  DONE {template_path.parent.name}/{model_name} "
                    f"applied={applied_meta.get('applied')} present={applied_meta.get('present')}",
                    flush=True,
                )
            except Exception as exc:
                tracker.record_failure(model, exc)
                failures.append(f"{template_path}: {type(exc).__name__}: {exc}")
                print(f"  ERROR {template_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print(format_summary(tracker.snapshot()))
    for f in failures:
        print(f"FAILED {f}", file=sys.stderr)
    return 0 if not failures else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="artifact_review")
    parser.add_argument("set_dirs", nargs="+")
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--base-suffix", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo-cache", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    return run_posthoc(args)


if __name__ == "__main__":
    raise SystemExit(main())
