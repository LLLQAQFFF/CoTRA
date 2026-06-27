#!/usr/bin/env python3
"""Reproduce the main scope-agreement numbers, bootstrap CIs, per-family
breakdown, and token usage from cached predictions.

This reuses the official metric implementation in ``eval_predictions.py``
(Cohen's kappa, exact agreement, Spearman) so the numbers match the paper.

Usage:
    python scripts/reproduce_tables.py \
        --data-root data/CoTRA-Bench \
        --outputs-root outputs/encoder_judge

It expects, under <outputs-root>, one directory per method whose prediction
files are named <model>.<tag>.encoder_pre_label.json, mirroring the split/
instance layout of <data-root>.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_predictions import collect_pairs, kappa, agreement  # noqa: E402

SPLITS = ["set1_v2", "set2_v2_batch03", "set3_v2"]

# method -> (output-tag directory, prediction filename suffix)
METHODS = {
    "B1 Evid+rules": ("rules-only-nollm", ".rules-only-nollm.encoder_pre_label.json"),
    "B3 Evid+LLM":   ("abl-nogate", ".abl-nogate.encoder_pre_label.json"),
    "B4 E2E judge":  ("e2e-judge", ".e2e-judge.encoder_pre_label.json"),
    "CoTRA":         ("crossfit-semantic-frozen", ".crossfit-semantic-frozen.encoder_pre_label.json"),
}


def family(model: str) -> str:
    for prefix, name in (("gpt", "GPT"), ("claude", "Claude"),
                         ("gemini", "Gemini"), ("glm", "GLM")):
        if model.startswith(prefix):
            return name
    return "Other"


def scope_pairs(rows):
    return [(r["gold"].get("risk_scope"), r["llm"].get("risk_scope")) for r in rows
            if r["gold"].get("risk_scope") is not None and r["llm"].get("risk_scope") is not None]


def bootstrap_ci(rows, n_boot=2000, seed=0):
    rng = random.Random(seed)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["set"], r["instance"], r["model"]), []).append(r)
    keys = list(groups)
    stats = []
    for _ in range(n_boot):
        samp = []
        for _ in range(len(keys)):
            samp.extend(groups[keys[rng.randrange(len(keys))]])
        k = kappa(scope_pairs(samp))
        if k == k:
            stats.append(k)
    stats.sort()
    return stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]


def token_usage(data_root: Path, root: Path, suffix: str):
    tot = {"calls": 0, "prompt": 0, "cached": 0, "completion": 0, "cost": 0.0, "cur": None}
    for split in SPLITS:
        for tpl in sorted((data_root / split).glob("*/*.target.json")):
            if tpl.name.endswith((".HA.target.json", ".HB.target.json", ".target.template.json")):
                continue
            pre = root / split / tpl.parent.name / tpl.name.replace(".target.json", suffix)
            if not pre.exists():
                continue
            m = json.loads(pre.read_text()).get("_run_meta", {})
            tot["calls"] += int(m.get("llm_calls") or 0)
            c = (m.get("cost") or {}).get("totals") or {}
            tot["cur"] = tot["cur"] or c.get("currency")
            tot["prompt"] += int(c.get("prompt_tokens") or 0)
            tot["cached"] += int(c.get("cached_prompt_tokens") or 0)
            tot["completion"] += int(c.get("completion_tokens") or 0)
            tot["cost"] += float(c.get("cost_amount") or 0)
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/CoTRA-Bench")
    ap.add_argument("--outputs-root", default="outputs/encoder_judge")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    out_root = Path(args.outputs_root)
    gold_dirs = [data_root / s for s in SPLITS]

    collected = {}
    print("=== Scope agreement (M1) ===")
    for name, (tag, suffix) in METHODS.items():
        root = out_root / tag
        if not root.exists():
            print(f"{name:16s}  (missing: {root})")
            continue
        rows, traj = collect_pairs(gold_dirs, suffix, prediction_root=root)
        collected[name] = (rows, traj)
        sp = scope_pairs(rows)
        print(f"{name:16s}  kappa={kappa(sp):.3f}  exact={agreement(sp):.3f}  (n={len(sp)})")

    print("\n=== Bootstrap 95% CI on scope kappa (trajectory-resampled) ===")
    for name, (rows, _) in collected.items():
        lo, hi = bootstrap_ci(rows)
        print(f"{name:16s}  kappa={kappa(scope_pairs(rows)):.3f}  CI [{lo:.3f}, {hi:.3f}]")

    if "CoTRA" in collected:
        print("\n=== CoTRA per-agent-family scope kappa ===")
        rows, traj = collected["CoTRA"]
        by_rows: dict = {}
        by_traj: dict = {}
        for r in rows:
            by_rows.setdefault(family(r["model"]), []).append(r)
        for t in traj:
            by_traj.setdefault(family(t["model"]), []).append(t)
        for f in ["GPT", "Claude", "Gemini", "GLM"]:
            rs = by_rows.get(f, [])
            print(f"{f:8s}  n_traj={len(by_traj.get(f, [])):3d}  n_act={len(rs):5d}  "
                  f"kappa={kappa(scope_pairs(rs)):.3f}  exact={agreement(scope_pairs(rs)):.3f}")

    print("\n=== Token / call usage (cost in provider-native currency) ===")
    for name, (tag, suffix) in METHODS.items():
        root = out_root / tag
        if not root.exists():
            continue
        u = token_usage(data_root, root, suffix)
        print(f"{name:16s}  calls={u['calls']:5d}  prompt={u['prompt']:>9d}  "
              f"completion={u['completion']:>9d}  cost={u['cost']:.3f}{u['cur'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
