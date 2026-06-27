"""Analyze human-target v2 judge predictions against consensus gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCALAR_DIMS = [
    "task_advancement",
    "debt_density",
    "fragility_delta",
    "regression_surface",
    "observability_loss",
]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, traj_rows = collect_pairs(
        [Path(p) for p in args.set_dir],
        args.prediction_suffix,
        prediction_root=Path(args.prediction_root) if args.prediction_root else None,
    )
    report = render_report(rows, traj_rows, args.label or "+".join(Path(p).name for p in args.set_dir))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"collected action_rows={len(rows)} trajectory_rows={len(traj_rows)}")
    print(f"wrote {out}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-dir", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--prediction-suffix", default=".pre_label.json")
    parser.add_argument(
        "--prediction-root",
        default=None,
        help="external prediction root; expects <root>/<set_name>/<instance>/<prediction_file>",
    )
    return parser.parse_args(argv)


def collect_pairs(
    set_dirs: list[Path],
    prediction_suffix: str = ".pre_label.json",
    prediction_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    action_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for set_dir in set_dirs:
        for target_path in sorted(set_dir.glob("*/*.target.json")):
            if target_path.name.endswith((".HA.target.json", ".HB.target.json", ".target.template.json")):
                continue
            pre_name = target_path.name.replace(".target.json", prediction_suffix)
            if prediction_root is None:
                pre_path = target_path.with_name(pre_name)
            else:
                pre_path = prediction_root / set_dir.name / target_path.parent.name / pre_name
            if not target_path.exists():
                continue
            if not pre_path.exists():
                continue
            pre = json.loads(pre_path.read_text())
            target = json.loads(target_path.read_text())
            model = target_path.name.replace(".target.json", "")
            instance = pre_path.parent.name
            judge_model = None
            for a in pre.get("action_level", []):
                judge_model = (a.get("_llm_judge") or {}).get("model")
                if judge_model:
                    break
            target_by_id = {a.get("action_id"): a for a in target.get("action_level", [])}
            for la in pre.get("action_level", []):
                ha = target_by_id.get(la.get("action_id"))
                if ha is None:
                    continue
                action_rows.append({
                    "set": set_dir.name,
                    "instance": instance,
                    "model": model,
                    "judge": judge_model,
                    "action_index": la.get("action_index"),
                    "gold": ha,
                    "llm": la,
                })
            trajectory_rows.append({
                "set": set_dir.name,
                "instance": instance,
                "model": model,
                "judge": judge_model,
                "gold": target,
                "llm": pre,
            })
    return action_rows, trajectory_rows


def render_report(rows: list[dict[str, Any]], traj_rows: list[dict[str, Any]], label: str) -> str:
    lines = [f"# Judge prediction vs consensus gold ({label})", ""]
    lines.append(f"Action pairs: {len(rows)}")
    lines.append(f"Trajectory pairs: {len(traj_rows)}")
    lines.append("")

    lines.append("## Action Metrics")
    lines.append("")
    scope_pairs = field_values(rows, 'risk_scope')
    lines.append(f"- risk_scope κ: **{fmt(kappa(scope_pairs))}**")
    lines.append(f"- risk_scope exact agreement: **{fmt(agreement(scope_pairs))}**")
    lines.append(f"- wrong_abstraction.present κ: **{fmt(kappa(bool_values(rows, 'wrong_abstraction', 'present', skip_null=True)))}**")
    lines.append(f"- wrong_abstraction.present F1: **{fmt(f1(bool_values(rows, 'wrong_abstraction', 'present', skip_null=True)))}**")
    lines.append("")
    lines.append("| metric | n | Spearman | Pearson | MAE |")
    lines.append("|---|---:|---:|---:|---:|")
    for dim in SCALAR_DIMS:
        pairs = scalar_pairs(rows, ("manual_risk_vector", dim))
        lines.append(metric_row(dim, pairs))
    lines.append(metric_row("action_myopia_score", scalar_pairs(rows, ("action_myopia_score",))))
    lines.append("")
    lines.extend(render_scope_diagnostics(rows))
    lines.append("")
    lines.extend(render_evidence_pattern_diagnostics(rows))
    lines.append("")

    lines.append("## Trajectory Metrics")
    lines.append("")
    for name in ["broad_rewrite", "artifact_residue"]:
        present_pairs = trajectory_bool_pairs(traj_rows, name)
        severity_pairs = trajectory_scalar_pairs(traj_rows, name, "severity")
        lines.append(f"- {name}.present κ: **{fmt(kappa(present_pairs))}**")
        lines.append(f"- {name}.present F1: **{fmt(f1(present_pairs))}**")
        lines.append(f"- {name}.severity: {metric_inline(severity_pairs)}")
    lines.append(f"- trajectory_myopia_score: {metric_inline(trajectory_myopia_pairs(traj_rows))}")
    lines.append("")
    lines.extend(render_cost_summary(traj_rows))
    lines.append("")
    lines.extend(render_evidence_summary(traj_rows))
    lines.append("")
    return "\n".join(lines)


def metric_row(name: str, pairs: list[tuple[float, float]]) -> str:
    return f"| {name} | {len(pairs)} | {fmt(spearman(pairs))} | {fmt(pearson(pairs))} | {fmt(mae(pairs))} |"


def metric_inline(pairs: list[tuple[float, float]]) -> str:
    return f"n={len(pairs)}, ρ={fmt(spearman(pairs))}, r={fmt(pearson(pairs))}, MAE={fmt(mae(pairs))}"


def scalar_pairs(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[tuple[float, float]]:
    out = []
    for row in rows:
        g = get_path(row["gold"], path)
        l = get_path(row["llm"], path)
        if g is None or l is None:
            continue
        out.append((float(g), float(l)))
    return out


def field_values(rows: list[dict[str, Any]], field: str) -> list[tuple[Any, Any]]:
    return [(r["gold"].get(field), r["llm"].get(field)) for r in rows
            if r["gold"].get(field) is not None and r["llm"].get(field) is not None]


def render_scope_diagnostics(rows: list[dict[str, Any]]) -> list[str]:
    watched = [
        ("artifact_only", "noise_no_effect"),
        ("substantive", "noise_no_effect"),
        ("substantive", "noise_reverted"),
        ("noise_reverted", "substantive"),
    ]
    lines = ["## Scope Diagnostics", ""]
    lines.append("| gold -> predicted | count |")
    lines.append("|---|---:|")
    for gold, pred in watched:
        n = sum(1 for r in rows if r["gold"].get("risk_scope") == gold and r["llm"].get("risk_scope") == pred)
        lines.append(f"| {gold} -> {pred} | {n} |")

    trajectory_review = review_effectiveness(rows, "_encoder_trajectory_review")
    scope_review = review_effectiveness(rows, "_encoder_scope_review")
    lines.append("")
    lines.append("| review stage | applied | helped | harmed | same-wrong | same-right |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(review_effectiveness_row("trajectory_review", trajectory_review))
    lines.append(review_effectiveness_row("encoder_scope_review", scope_review))
    return lines


def render_evidence_pattern_diagnostics(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Evidence Pattern Diagnostics", ""]
    lines.append("| pattern | rows | exact | substantive -> noise_no_effect | noise_reverted -> substantive |")
    lines.append("|---|---:|---:|---:|---:|")
    patterns = [
        ("apply_failed_but_final_present", lambda ev: bool((ev.get("replay_evidence") or {}).get("apply_failed")) and bool((ev.get("final_diff_contribution") or {}).get("present"))),
        ("final_present_source_path", lambda ev: bool((ev.get("final_diff_contribution") or {}).get("present")) and bool((ev.get("final_diff_contribution") or {}).get("source_paths"))),
        ("patch_no_effect_final_present", lambda ev: ((ev.get("patch_survival") or {}).get("status") == "no_effect") and bool((ev.get("final_diff_contribution") or {}).get("present"))),
        ("semantic_exact_text_survived", lambda ev: (ev.get("final_diff_contribution") or {}).get("semantic_survival") == "exact_text_survived"),
        ("semantic_later_refined", lambda ev: (ev.get("final_diff_contribution") or {}).get("semantic_survival") == "later_refined"),
        ("semantic_fully_removed", lambda ev: (ev.get("final_diff_contribution") or {}).get("semantic_survival") == "fully_removed"),
    ]
    for label, predicate in patterns:
        subset = [r for r in rows if predicate(r["llm"].get("_encoder_evidence") or {})]
        lines.append(_evidence_pattern_row(label, subset))
    return lines


def _evidence_pattern_row(label: str, rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    exact = agreement(field_values(rows, "risk_scope")) if rows else float("nan")
    sub_to_noise = sum(
        1 for r in rows
        if r["gold"].get("risk_scope") == "substantive"
        and r["llm"].get("risk_scope") == "noise_no_effect"
    )
    reverted_to_sub = sum(
        1 for r in rows
        if r["gold"].get("risk_scope") == "noise_reverted"
        and r["llm"].get("risk_scope") == "substantive"
    )
    return f"| {label} | {total} | {fmt(exact)} | {sub_to_noise} | {reverted_to_sub} |"


def review_effectiveness(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    stats = {"applied": 0, "helped": 0, "harmed": 0, "same_wrong": 0, "same_right": 0}
    for r in rows:
        meta = r["llm"].get(key) or {}
        if not meta.get("applied") and not meta.get("scope_override_applied"):
            continue
        old = meta.get("original_scope") or meta.get("old_scope")
        new = meta.get("new_scope") or r["llm"].get("risk_scope")
        gold = r["gold"].get("risk_scope")
        if old is None or new is None or gold is None:
            continue
        stats["applied"] += 1
        if old != gold and new == gold:
            stats["helped"] += 1
        elif old == gold and new != gold:
            stats["harmed"] += 1
        elif new == gold:
            stats["same_right"] += 1
        else:
            stats["same_wrong"] += 1
    return stats


def review_effectiveness_row(name: str, stats: dict[str, int]) -> str:
    return (
        f"| {name} | {stats['applied']} | {stats['helped']} | {stats['harmed']} | "
        f"{stats['same_wrong']} | {stats['same_right']} |"
    )


def render_cost_summary(traj_rows: list[dict[str, Any]]) -> list[str]:
    totals = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_amount": 0.0,
        "cost_unknown": False,
    }
    currency = None
    for row in traj_rows:
        meta = row["llm"].get("_run_meta") or {}
        totals["llm_calls"] += int(meta.get("llm_calls") or 0)
        cost = meta.get("cost") or {}
        ct = cost.get("totals") or {}
        currency = currency or ct.get("currency")
        totals["prompt_tokens"] += int(ct.get("prompt_tokens") or 0)
        totals["cached_prompt_tokens"] += int(ct.get("cached_prompt_tokens") or 0)
        totals["completion_tokens"] += int(ct.get("completion_tokens") or 0)
        amount = ct.get("cost_amount")
        if amount is None:
            totals["cost_unknown"] = True
        else:
            totals["cost_amount"] += float(amount)
    lines = ["## Cost Summary", ""]
    lines.append(f"- llm_calls: **{totals['llm_calls']}**")
    lines.append(f"- prompt_tokens: **{totals['prompt_tokens']}**")
    lines.append(f"- cached_prompt_tokens: **{totals['cached_prompt_tokens']}**")
    lines.append(f"- completion_tokens: **{totals['completion_tokens']}**")
    if totals["cost_unknown"]:
        lines.append(f"- cost: **unknown {currency or ''}**")
    else:
        lines.append(f"- cost: **{totals['cost_amount']:.6f} {currency or ''}**")
    return lines


def render_evidence_summary(traj_rows: list[dict[str, Any]]) -> list[str]:
    baselines = set()
    effectful_actions = 0
    static_actions = 0
    trajectories_with_static = 0
    for row in traj_rows:
        meta = row["llm"].get("_run_meta") or {}
        if meta.get("baseline"):
            baselines.add(str(meta["baseline"]))
        coverage = ((meta.get("repo_env") or {}).get("static_coverage") or {})
        effectful_actions += int(coverage.get("effectful_actions") or 0)
        current_static = int(coverage.get("static_actions") or 0)
        static_actions += current_static
        if current_static:
            trajectories_with_static += 1
    return [
        "## Evidence Summary",
        "",
        f"- baseline: **{', '.join(sorted(baselines)) or 'none'}**",
        f"- trajectories_with_static_evidence: **{trajectories_with_static}/{len(traj_rows)}**",
        f"- effectful_actions: **{effectful_actions}**",
        f"- static_actions: **{static_actions}**",
    ]


def bool_values(rows: list[dict[str, Any]], obj: str, field: str,
                skip_null: bool = False) -> list[tuple[bool, bool]]:
    out = []
    for r in rows:
        g = (r["gold"].get(obj) or {}).get(field)
        l = (r["llm"].get(obj) or {}).get(field)
        if skip_null and (g is None or l is None):
            continue
        out.append((bool(g), bool(l)))
    return out


def trajectory_bool_pairs(rows: list[dict[str, Any]], penalty: str) -> list[tuple[bool, bool]]:
    out = []
    for r in rows:
        g = get_path(r["gold"], ("trajectory_level", "trajectory_penalties", penalty, "present"))
        l = get_path(r["llm"], ("trajectory_level", "trajectory_penalties", penalty, "present"))
        if g is None or l is None:
            continue
        out.append((bool(g), bool(l)))
    return out


def trajectory_scalar_pairs(rows: list[dict[str, Any]], penalty: str, field: str) -> list[tuple[float, float]]:
    out = []
    for r in rows:
        g = get_path(r["gold"], ("trajectory_level", "trajectory_penalties", penalty, field))
        l = get_path(r["llm"], ("trajectory_level", "trajectory_penalties", penalty, field))
        if g is None or l is None:
            continue
        out.append((float(g), float(l)))
    return out


def trajectory_myopia_pairs(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    out = []
    for r in rows:
        g = get_path(r["gold"], ("trajectory_level", "trajectory_myopia_score"))
        l = get_path(r["llm"], ("trajectory_level", "trajectory_myopia_score"))
        if g is None or l is None:
            continue
        out.append((float(g), float(l)))
    return out


def get_path(data: dict, path: tuple[str, ...]):
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def kappa(pairs: list[tuple[Any, Any]]) -> float:
    if not pairs:
        return float("nan")
    xs, ys = zip(*pairs)
    labels = set(xs) | set(ys)
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == label for x in xs) / n) * (sum(y == label for y in ys) / n)
             for label in labels)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1.0 - pe)


def agreement(pairs: list[tuple[Any, Any]]) -> float:
    if not pairs:
        return float("nan")
    return sum(x == y for x, y in pairs) / len(pairs)


def f1(pairs: list[tuple[bool, bool]]) -> float:
    if not pairs:
        return float("nan")
    tp = sum(g and l for g, l in pairs)
    fp = sum((not g) and l for g, l in pairs)
    fn = sum(g and (not l) for g, l in pairs)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom else 0.0


def pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / (vx * vy) ** 0.5


def spearman(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 3:
        return float("nan")
    xs, ys = zip(*pairs)
    return pearson(list(zip(rank(list(xs)), rank(list(ys)))))


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def mae(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return float("nan")
    return sum(abs(g - l) for g, l in pairs) / len(pairs)


def fmt(value: float) -> str:
    return "N/A" if value != value else f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
