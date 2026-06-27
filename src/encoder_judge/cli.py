"""CLI for the evidence-first encoder judge.

Usage:
  python -m encoder_judge.cli prelabel <template>
  python -m encoder_judge.cli prelabel-set <set_dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from encoder_judge.baselines import (
    static_coverage,
    synthesize_static_action,
    synthesize_static_trajectory,
)
from encoder_judge.calibration import ScopeCalibrator, calibration_templates, load_base_prelabel
from encoder_judge.evidence import build_evidence_table
from encoder_judge.e2e_baseline import run_e2e_baseline
from encoder_judge.formal_output import (
    attach_engineering_consequences,
    attach_repository_evidence,
    build_trajectory_engineering_scorecard,
    formal_output_meta,
)
from encoder_judge.review import (
    apply_trajectory_review,
    review_trajectory,
    should_run_trajectory_review,
)
from encoder_judge.rules import synthesize_action, synthesize_trajectory, zero_risk_vector
from encoder_judge.semantic import score_action_semantics
from encoder_judge.trajectory import (
    apply_encoder_scope_review,
    synthesize_encoder_trajectory,
    trajectory_summary,
)
from encoder_judge.wrong_abstraction import (
    MAX_WA_CANDIDATES,
    apply_wa_review,
    build_wa_candidate_packet,
    review_wrong_abstraction_candidates,
    should_run_wa_review,
)
from llm_judge import derive
from llm_judge import config as llm_config
from llm_judge import rules as llm_rules
from llm_judge.client import JudgeClient
from llm_judge.cost import CostTracker


SCHEMA_VERSION = "human-target-v2"
JUDGE_NAME = "encoder-llm-v2"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "encoder_judge"


def prelabel_template(
    template_path: Path,
    calibrator: ScopeCalibrator | None = None,
    *,
    repo_cache_root: Path | None = None,
    baseline: str | None = None,
    use_llm: bool = True,
    use_trajectory_review: bool = True,
    use_wa_review: bool = True,
    max_wa_candidates: int = MAX_WA_CANDIDATES,
    judge_model: str | None = None,
    freeze_scope: bool = False,
    scope_review_mode: str = "off",
    base_prediction_root: Path | None = None,
    base_prediction_suffix: str | None = None,
) -> dict[str, Any]:
    if baseline not in {None, "static-only", "evidence-rules", "e2e-llm"}:
        raise ValueError(f"Unsupported baseline: {baseline}")
    if baseline == "static-only" and repo_cache_root is None:
        raise ValueError("static-only baseline requires --repo-cache")
    if baseline is not None and calibrator is not None:
        raise ValueError("baseline runs do not support --calibration-set")
    if baseline in {"static-only", "evidence-rules"}:
        use_llm = False
        use_trajectory_review = False
        use_wa_review = False
    if scope_review_mode not in {"off", "standard", "evidence-conflict"}:
        raise ValueError(f"Unsupported scope_review_mode: {scope_review_mode}")

    template = json.loads(template_path.read_text())
    if baseline == "e2e-llm":
        return prelabel_template_e2e(template, template_path, judge_model=judge_model)

    evidence = build_evidence_table(template, template_path, repo_cache_root=repo_cache_root)
    coverage = static_coverage(evidence)
    if baseline == "static-only" and coverage["effectful_actions"] and not coverage["static_actions"]:
        raise ValueError("static-only baseline has no static evidence; check --repo-cache and replay metadata")
    if len(evidence) != len(template.get("action_level") or []):
        raise ValueError(
            f"evidence/action length mismatch: {len(evidence)} != {len(template.get('action_level') or [])}"
        )

    result = dict(template)
    base_by_id = load_base_prelabel(template_path)
    frozen_base_by_id = load_external_base_prelabel(
        template_path,
        base_prediction_root=base_prediction_root,
        base_prediction_suffix=base_prediction_suffix,
    )
    actions = []
    for action, ev in zip(template.get("action_level") or [], evidence, strict=True):
        synthesized = (
            synthesize_static_action(action, ev)
            if baseline == "static-only"
            else synthesize_action(action, ev)
        )
        if calibrator is not None:
            synthesized = apply_calibration(synthesized, ev, base_by_id.get(ev.action_id), calibrator)
        if frozen_base_by_id:
            synthesized = apply_base_scope(synthesized, frozen_base_by_id.get(ev.action_id))
        actions.append(synthesized)

    tracker = CostTracker()
    llm_parse_failures: list[dict[str, Any]] = []
    action_llm_calls = 0
    trajectory_review_calls = 0
    wa_review_calls = 0
    trajectory_review_parse_failures: list[str] = []
    wa_review_parse_failures: list[str] = []
    trajectory_review_meta: dict[str, Any] = {
        "skipped_reason": "llm_disabled" if not use_llm else "",
        "overrides_applied": {},
    }
    wa_review_meta: dict[str, Any] = {
        "skipped_reason": "llm_disabled" if not use_llm else "",
        "overrides_applied": {},
        "candidates": 0,
    }
    model = judge_model or llm_config.get_routing().for_family((template.get("trajectory_meta") or {}).get("family"))
    task = extract_task_description(template, template_path)
    if use_llm:
        client = JudgeClient(
            request_timeout_seconds=llm_config.ENCODER_REQUEST_TIMEOUT_SECONDS,
            retry_max_attempts=llm_config.ENCODER_RETRY_MAX_ATTEMPTS,
            retry_on_empty=True,
        )
        summary = trajectory_summary(actions, evidence)
        candidates = [
            (i, action, ev)
            for i, (action, ev) in enumerate(zip(actions, evidence, strict=True))
            if action.get("risk_scope") in {"substantive", "uncertain"}
            and (action.get("_encoder_rule") or {}).get("llm_required", True)
        ]

        def _score_one(item: tuple[int, dict[str, Any], Any]) -> tuple[int, dict[str, Any]]:
            i, action, ev = item
            try:
                scored = score_action_semantics(
                    action=action,
                    evidence=ev,
                    task_description=task,
                    trajectory_summary=summary,
                    client=client,
                    model=model,
                    tracker=tracker,
                )
                if freeze_scope:
                    scored = restore_frozen_scope(scored, action)
            except Exception as exc:
                tracker.record_failure(model, exc)
                scored = dict(action)
                scored["_encoder_semantic"] = {
                    "model": model,
                    "parse_failed": True,
                    "parse_failures": ["llm_call_failed"],
                    "fallback": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw": "",
                }
            processed = llm_rules.postprocess_action(scored)
            if freeze_scope:
                processed = restore_frozen_scope(processed, action)
            return i, processed

        with ThreadPoolExecutor(max_workers=llm_config.CONCURRENCY_PER_TRAJECTORY) as ex:
            futures = [ex.submit(_score_one, item) for item in candidates]
            for fut in as_completed(futures):
                i, scored = fut.result()
                actions[i] = scored
                action_llm_calls += 1
                meta = scored.get("_encoder_semantic") or {}
                if meta.get("parse_failed"):
                    llm_parse_failures.append({
                        "action_index": scored.get("action_index"),
                        "parse_failures": meta.get("parse_failures") or [],
                    })

    scope_review_applied = 0
    if baseline != "static-only" and not freeze_scope and scope_review_mode != "off":
        scope_review_applied = apply_encoder_scope_review(
            actions,
            evidence,
            mode=scope_review_mode,
        )

    trajectory = dict(template.get("trajectory_level") or {})
    penalties = (
        synthesize_static_trajectory(evidence)
        if baseline == "static-only"
        else synthesize_encoder_trajectory(actions, evidence)
    )
    if use_llm and use_trajectory_review:
        should_review, skipped_reason = should_run_trajectory_review(actions, evidence, penalties)
        trajectory_review_meta["skipped_reason"] = skipped_reason
        if should_review:
            try:
                client = JudgeClient(
                    request_timeout_seconds=llm_config.ENCODER_REQUEST_TIMEOUT_SECONDS,
                    retry_max_attempts=llm_config.ENCODER_RETRY_MAX_ATTEMPTS,
                    retry_on_empty=True,
                )
                review_result, packet_meta = review_trajectory(
                    template=template,
                    template_path=template_path,
                    actions=actions,
                    evidence=evidence,
                    deterministic_penalties=penalties,
                    task_description=task,
                    client=client,
                    model=model,
                    tracker=tracker,
                )
                trajectory_review_calls = 1
                trajectory_review_meta.update(packet_meta)
                if review_result.parse_failed:
                    trajectory_review_parse_failures = review_result.parse_failures
                else:
                    applied = apply_trajectory_review(
                        actions,
                        penalties,
                        review_result,
                        apply_wrong_abstraction=not use_wa_review,
                        apply_scope_overrides=not freeze_scope,
                    )
                    penalties = applied["penalties"]
                    trajectory_review_meta["overrides_applied"] = applied["meta"]
            except Exception as exc:
                tracker.record_failure(model, exc)
                trajectory_review_meta["skipped_reason"] = "llm_call_failed"
                trajectory_review_parse_failures = [f"{type(exc).__name__}: {exc}"]
    elif not use_trajectory_review:
        trajectory_review_meta["skipped_reason"] = "disabled"

    if use_llm and use_wa_review:
        packet = build_wa_candidate_packet(
            template=template,
            template_path=template_path,
            actions=actions,
            evidence=evidence,
            task_description=task,
            max_candidates=max_wa_candidates,
        )
        should_review_wa, skipped_reason = should_run_wa_review(packet)
        wa_review_meta.update({
            "skipped_reason": skipped_reason,
            "candidates": len(packet.get("candidates") or []),
            "packet_chars": len(json.dumps(packet, ensure_ascii=False, indent=2)),
            "packet_truncated": bool(packet.get("packet_truncated")),
        })
        if should_review_wa:
            try:
                client = JudgeClient(
                    request_timeout_seconds=llm_config.ENCODER_REQUEST_TIMEOUT_SECONDS,
                    retry_max_attempts=llm_config.ENCODER_RETRY_MAX_ATTEMPTS,
                    retry_on_empty=True,
                )
                wa_result, packet_meta = review_wrong_abstraction_candidates(
                    template=template,
                    template_path=template_path,
                    actions=actions,
                    evidence=evidence,
                    task_description=task,
                    client=client,
                    model=model,
                    tracker=tracker,
                    max_candidates=max_wa_candidates,
                )
                wa_review_calls = 1
                wa_review_meta.update(packet_meta)
                if wa_result.parse_failed:
                    wa_review_parse_failures = wa_result.parse_failures
                else:
                    wa_review_meta["overrides_applied"] = apply_wa_review(actions, wa_result)
            except Exception as exc:
                tracker.record_failure(model, exc)
                wa_review_meta["skipped_reason"] = "llm_call_failed"
                wa_review_parse_failures = [f"{type(exc).__name__}: {exc}"]
    elif not use_wa_review:
        wa_review_meta["skipped_reason"] = "disabled"

    trajectory["trajectory_penalties"] = penalties
    trajectory["trajectory_myopia_score"] = derive.derive_trajectory_myopia_score(actions, penalties)
    attach_repository_evidence(actions, evidence)
    attach_engineering_consequences(actions)
    trajectory["trajectory_engineering_scorecard"] = build_trajectory_engineering_scorecard(
        actions,
        evidence,
        penalties,
        trajectory["trajectory_myopia_score"],
    )
    result["action_level"] = actions
    result["trajectory_level"] = trajectory
    result["_run_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "judge": JUDGE_NAME,
        "baseline": baseline,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "judge_model": model if use_llm else None,
        "llm_calls": action_llm_calls + trajectory_review_calls + wa_review_calls,
        "action_llm_calls": action_llm_calls,
        "trajectory_review_calls": trajectory_review_calls,
        "wa_review_calls": wa_review_calls,
        "cost_metric_version": "cost-v2",
        "cost": tracker.snapshot(),
        "parse_failures": llm_parse_failures,
        "trajectory_review_parse_failures": trajectory_review_parse_failures,
        "wa_review_parse_failures": wa_review_parse_failures,
        "trajectory_review_skipped_reason": trajectory_review_meta.get("skipped_reason", ""),
        "trajectory_review_overrides_applied": trajectory_review_meta.get("overrides_applied", {}),
        "trajectory_review_meta": trajectory_review_meta,
        "wa_review_candidates": wa_review_meta.get("candidates", 0),
        "wa_review_overrides_applied": wa_review_meta.get("overrides_applied", {}),
        "wa_review_packet_chars": wa_review_meta.get("packet_chars", 0),
        "wa_review_skipped_reason": wa_review_meta.get("skipped_reason", ""),
        "wa_review_meta": wa_review_meta,
        "llm_timeout_seconds": llm_config.ENCODER_REQUEST_TIMEOUT_SECONDS,
        "llm_retry_max_attempts": llm_config.ENCODER_RETRY_MAX_ATTEMPTS,
        "repo_env": {
            "used": repo_cache_root is not None,
            "static_coverage": coverage,
        },
        "formal_output": formal_output_meta(),
        "scope_review": {
            "applied": scope_review_applied,
            "mode": "frozen" if freeze_scope else scope_review_mode,
        },
        "freeze_scope": freeze_scope,
        "base_prediction": {
            "root": str(base_prediction_root) if base_prediction_root else "",
            "suffix": base_prediction_suffix or "",
            "loaded": bool(frozen_base_by_id),
        },
    }
    return result


def prelabel_template_e2e(
    template: dict[str, Any],
    template_path: Path,
    *,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Run the one-call end-to-end LLM baseline for one target template."""
    model = judge_model or llm_config.get_routing().for_family((template.get("trajectory_meta") or {}).get("family"))
    tracker = CostTracker()
    client = JudgeClient(
        request_timeout_seconds=llm_config.E2E_REQUEST_TIMEOUT_SECONDS,
        retry_max_attempts=llm_config.E2E_RETRY_MAX_ATTEMPTS,
        retry_on_empty=True,
    )
    e2e = run_e2e_baseline(
        template=template,
        template_path=template_path,
        client=client,
        model=model,
        tracker=tracker,
    )
    for action in e2e.actions:
        action["_llm_judge"] = {
            "model": model,
            "baseline": "e2e-llm",
        }
    result = dict(template)
    trajectory = dict(template.get("trajectory_level") or {})
    trajectory["trajectory_penalties"] = e2e.trajectory_penalties
    trajectory["trajectory_myopia_score"] = derive.derive_trajectory_myopia_score(
        e2e.actions,
        e2e.trajectory_penalties,
    )
    result["action_level"] = e2e.actions
    result["trajectory_level"] = trajectory
    result["_run_meta"] = {
        "schema_version": SCHEMA_VERSION,
        "judge": "e2e-llm-baseline-v2",
        "baseline": "e2e-llm",
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "judge_model": model,
        "llm_calls": 1,
        "action_llm_calls": 0,
        "trajectory_review_calls": 0,
        "wa_review_calls": 0,
        "cost_metric_version": "cost-v2",
        "cost": tracker.snapshot(),
        "parse_failed": e2e.parse_failed,
        "parse_failures": e2e.parse_failures,
        "e2e_packet_meta": e2e.packet_meta,
        "llm_timeout_seconds": llm_config.E2E_REQUEST_TIMEOUT_SECONDS,
        "llm_retry_max_attempts": llm_config.E2E_RETRY_MAX_ATTEMPTS,
        "max_tokens_e2e": llm_config.MAX_TOKENS_E2E,
    }
    return result


def cmd_prelabel(args: argparse.Namespace) -> int:
    template = Path(args.template)
    judge_model = args.judge_model or llm_config.get_routing().for_family(None)
    output_tag = args.output_tag or args.baseline or judge_model
    out = (
        Path(args.output)
        if args.output
        else output_path_for_template(
            template,
            output_tag,
            output_dir=Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / sanitize_output_tag(output_tag),
        )
    )
    calibrator = build_calibrator(
        args.calibration_set,
        repo_cache_root=Path(args.repo_cache) if args.repo_cache else None,
    )
    result = prelabel_template(
        template,
        calibrator,
        repo_cache_root=Path(args.repo_cache) if args.repo_cache else None,
        baseline=args.baseline,
        use_llm=not args.no_llm and args.baseline is None,
        use_trajectory_review=not args.no_trajectory_review,
        use_wa_review=not args.no_wa_review,
        max_wa_candidates=args.max_wa_candidates,
        judge_model=judge_model,
        freeze_scope=args.freeze_scope,
        scope_review_mode=args.scope_review_mode,
        base_prediction_root=Path(args.base_prediction_root) if args.base_prediction_root else None,
        base_prediction_suffix=args.base_prediction_suffix,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[encoder-prelabel] {template} -> {out}", flush=True)
    return 0


def cmd_prelabel_set(args: argparse.Namespace) -> int:
    set_dir = Path(args.set_dir)
    templates = sorted(set_dir.glob("*/*.target.template.json"))
    if not templates:
        print(f"error: no *.target.template.json under {set_dir}/*/", file=sys.stderr)
        return 1

    failures: list[dict[str, str]] = []
    calibrator = build_calibrator(
        args.calibration_set,
        repo_cache_root=Path(args.repo_cache) if args.repo_cache else None,
    )
    judge_model = args.judge_model or llm_config.get_routing().for_family(None)
    output_tag = args.output_tag or args.baseline or judge_model
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / sanitize_output_tag(output_tag)
    print(f"[encoder-prelabel-set] {len(templates)} templates under {set_dir}", flush=True)
    print(f"[encoder-prelabel-set] judge_model={judge_model} output_tag={output_tag}", flush=True)
    print(f"[encoder-prelabel-set] output_dir={output_dir}", flush=True)
    if args.repo_cache:
        print(f"[encoder-prelabel-set] replay + static enabled (cache: {args.repo_cache})", flush=True)
    processed = 0
    for template in templates:
        if args.limit is not None and processed >= args.limit:
            break
        try:
            out = output_path_for_template(template, output_tag, output_dir=output_dir, set_dir=set_dir)
            if args.skip_existing and out.exists():
                print(f"  SKIP existing {out.name}", flush=True)
                continue
            print(f"  START {template.parent.name}/{template.name}", flush=True)
            result = prelabel_template(
                template,
                calibrator,
                repo_cache_root=Path(args.repo_cache) if args.repo_cache else None,
                baseline=args.baseline,
                use_llm=not args.no_llm and args.baseline is None,
                use_trajectory_review=not args.no_trajectory_review,
                use_wa_review=not args.no_wa_review,
                max_wa_candidates=args.max_wa_candidates,
                judge_model=judge_model,
                freeze_scope=args.freeze_scope,
                scope_review_mode=args.scope_review_mode,
                base_prediction_root=Path(args.base_prediction_root) if args.base_prediction_root else None,
                base_prediction_suffix=args.base_prediction_suffix,
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            processed += 1
            meta = result.get("_run_meta") or {}
            print(f"  DONE {template.name} -> {out.name} | llm_calls={meta.get('llm_calls', 0)}", flush=True)
        except Exception as exc:
            print(f"  ERROR {template}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append({"template": str(template), "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_fast:
                break
    return 0 if not failures else 2


def default_output_path(template: Path, output_tag: str | None = None) -> Path:
    tag = sanitize_output_tag(output_tag) if output_tag else ""
    suffix = f".{tag}.encoder_pre_label.json" if tag else ".encoder_pre_label.json"
    return template.with_name(template.name.replace(".target.template.json", suffix))


def output_path_for_template(
    template: Path,
    output_tag: str | None,
    *,
    output_dir: Path | None = None,
    set_dir: Path | None = None,
) -> Path:
    name = default_output_path(template, output_tag).name
    if output_dir is None:
        return template.with_name(name)
    if set_dir is not None:
        try:
            rel_parent = template.parent.relative_to(set_dir)
            return output_dir / set_dir.name / rel_parent / name
        except ValueError:
            pass
    return output_dir / template.parent.name / name


def load_external_base_prelabel(
    template_path: Path,
    *,
    base_prediction_root: Path | None,
    base_prediction_suffix: str | None,
) -> dict[str, dict[str, Any]]:
    if base_prediction_root is None or not base_prediction_suffix:
        return {}
    model = template_path.name.replace(".target.template.json", "")
    set_name = template_path.parent.parent.name
    path = base_prediction_root / set_name / template_path.parent.name / f"{model}{base_prediction_suffix}"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {str(a.get("action_id")): a for a in data.get("action_level", [])}


def apply_base_scope(action: dict[str, Any], base_action: dict[str, Any] | None) -> dict[str, Any]:
    if not base_action or not base_action.get("risk_scope"):
        return action
    out = dict(action)
    out["risk_scope"] = base_action.get("risk_scope")
    out["risk_scope_rationale"] = base_action.get("risk_scope_rationale") or out.get("risk_scope_rationale") or ""
    for key in ("action_role", "actual_effect", "relates_to_target"):
        if key in base_action:
            out[key] = base_action[key]
    if out["risk_scope"] not in {"substantive", "uncertain"}:
        out["manual_risk_vector"] = zero_risk_vector(out["risk_scope_rationale"])
        out["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_encoder_base_scope"] = {
        "enabled": True,
        "base_scope": base_action.get("risk_scope"),
    }
    rule_meta = dict(out.get("_encoder_rule") or {})
    rule_meta["llm_required"] = out["risk_scope"] in {"substantive", "uncertain"}
    out["_encoder_rule"] = rule_meta
    return out


def restore_frozen_scope(scored: dict[str, Any], frozen_action: dict[str, Any]) -> dict[str, Any]:
    frozen_scope = frozen_action.get("risk_scope")
    if not frozen_scope:
        return scored
    out = dict(scored)
    out["risk_scope"] = frozen_scope
    out["risk_scope_rationale"] = frozen_action.get("risk_scope_rationale") or out.get("risk_scope_rationale") or ""
    if frozen_scope not in {"substantive", "uncertain"}:
        out["manual_risk_vector"] = zero_risk_vector(out["risk_scope_rationale"])
        out["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}
        out["relates_to_target"] = False
    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_encoder_scope_freeze"] = {
        "enabled": True,
        "frozen_scope": frozen_scope,
        "model_scope_ignored": scored.get("risk_scope"),
    }
    return out


def sanitize_output_tag(value: str | None) -> str:
    if not value:
        return ""
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "-")
    return "".join(out).strip(".-")


def build_calibrator(paths: list[str] | None, repo_cache_root: Path | None = None) -> ScopeCalibrator | None:
    if not paths:
        return None
    calibrator = ScopeCalibrator()
    templates = calibration_templates([Path(p) for p in paths])
    calibrator.fit(templates, repo_cache_root=repo_cache_root)
    print(f"[encoder-calibration] trained on {len(templates)} templates from {', '.join(paths)}")
    return calibrator


def extract_task_description(template: dict[str, Any], template_path: Path) -> str:
    model = template_path.name.replace(".target.template.json", "")
    for candidate in [template_path.with_name(f"{model}.traj.json"), *sorted(template_path.parent.glob("*.traj.json"))]:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        desc = (data.get("metadata") or {}).get("task_description")
        if desc:
            return str(desc)
    return str((template.get("trajectory_meta") or {}).get("sample_id") or "")


def apply_calibration(
    action: dict[str, Any],
    ev,
    base_action: dict[str, Any] | None,
    calibrator: ScopeCalibrator,
) -> dict[str, Any]:
    predicted_scope = calibrator.predict_scope(ev, base_action)
    out = dict(action)
    out["risk_scope"] = predicted_scope
    out["risk_scope_rationale"] = (
        f"Human-calibrated encoder scope from evidence and base semantic judgment. "
        f"Rule baseline was {action.get('risk_scope')}."
    )

    if predicted_scope in {"substantive", "uncertain"} and base_action:
        out["manual_risk_vector"] = dict(base_action.get("manual_risk_vector") or {})
        out["wrong_abstraction"] = dict(base_action.get("wrong_abstraction") or {})
    elif predicted_scope not in {"substantive", "uncertain"}:
        out["manual_risk_vector"] = zero_risk_vector(out["risk_scope_rationale"])
        out["wrong_abstraction"] = {"present": False, "severity": 0.0, "rationale": ""}

    out["action_myopia_score"] = derive.derive_action_myopia_score(out)
    out["is_myopic"] = derive.derive_is_myopic(out)
    out["_encoder_calibration"] = {
        "enabled": True,
        "base_scope": (base_action or {}).get("risk_scope"),
        "rule_scope": action.get("risk_scope"),
        "calibrated_scope": predicted_scope,
    }
    rule_meta = dict(out.get("_encoder_rule") or {})
    rule_meta["llm_required"] = predicted_scope in {"substantive", "uncertain"}
    out["_encoder_rule"] = rule_meta
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="encoder_judge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prelabel")
    p1.add_argument("template")
    p1.add_argument("-o", "--output", default=None)
    p1.add_argument("--calibration-set", action="append", default=None)
    p1.add_argument("--repo-cache", default=None)
    p1.add_argument("--judge-model", default=None, help="override judge model for this run")
    p1.add_argument("--output-tag", default=None, help="output tag; default is sanitized judge model")
    p1.add_argument("--output-dir", default=None, help="directory for generated predictions")
    p1.add_argument("--baseline", choices=["static-only", "evidence-rules", "e2e-llm"], default=None)
    p1.add_argument("--no-llm", action="store_true")
    p1.add_argument("--no-trajectory-review", action="store_true")
    p1.add_argument("--no-wa-review", action="store_true")
    p1.add_argument("--freeze-scope", action="store_true")
    p1.add_argument("--scope-review-mode", choices=["off", "standard", "evidence-conflict"], default="off")
    p1.add_argument("--base-prediction-root", default=None)
    p1.add_argument("--base-prediction-suffix", default=None)
    p1.add_argument("--max-wa-candidates", type=int, default=MAX_WA_CANDIDATES)
    p1.set_defaults(func=cmd_prelabel)

    p2 = sub.add_parser("prelabel-set")
    p2.add_argument("set_dir")
    p2.add_argument("--fail-fast", action="store_true")
    p2.add_argument("--limit", type=int, default=None, help="maximum number of templates to process")
    p2.add_argument("--skip-existing", action="store_true", help="skip templates whose output already exists")
    p2.add_argument("--calibration-set", action="append", default=None)
    p2.add_argument("--repo-cache", default=None)
    p2.add_argument("--judge-model", default=None, help="override judge model for this run")
    p2.add_argument("--output-tag", default=None, help="output tag; default is sanitized judge model")
    p2.add_argument("--output-dir", default=None, help="directory for generated predictions")
    p2.add_argument("--baseline", choices=["static-only", "evidence-rules", "e2e-llm"], default=None)
    p2.add_argument("--no-llm", action="store_true")
    p2.add_argument("--no-trajectory-review", action="store_true")
    p2.add_argument("--no-wa-review", action="store_true")
    p2.add_argument("--freeze-scope", action="store_true")
    p2.add_argument("--scope-review-mode", choices=["off", "standard", "evidence-conflict"], default="off")
    p2.add_argument("--base-prediction-root", default=None)
    p2.add_argument("--base-prediction-suffix", default=None)
    p2.add_argument("--max-wa-candidates", type=int, default=MAX_WA_CANDIDATES)
    p2.set_defaults(func=cmd_prelabel_set)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
