"""Orchestrates all static analyzers for one action and assembles a unified
`static_signals` dict for downstream LLM prompts.

Caller responsibility: build the AnalyzerInput correctly (snapshot, action,
state, workdir, base_commit). Each analyzer's output is namespaced under its
.name attribute to keep the dict flat-ish but unambiguous.
"""

from __future__ import annotations

from typing import Any

from static_analysis.artifact_scan import ArtifactScanAnalyzer
from static_analysis.base import AnalyzerInput, StaticAnalyzer
from static_analysis.breadth_metrics import BreadthMetricsAnalyzer
from static_analysis.direction_lock import DirectionLockAnalyzer
from static_analysis.fragility_patterns import FragilityPatternsAnalyzer
from static_analysis.lint_delta import LintDeltaAnalyzer


DEFAULT_ANALYZERS: list[StaticAnalyzer] = [
    DirectionLockAnalyzer(),
    FragilityPatternsAnalyzer(),
    LintDeltaAnalyzer(),
    BreadthMetricsAnalyzer(),
    ArtifactScanAnalyzer(),
]


def run_analyzers(inp: AnalyzerInput,
                   analyzers: list[StaticAnalyzer] | None = None) -> dict[str, Any]:
    """Run all analyzers and namespace their outputs under analyzer.name."""
    analyzers = analyzers if analyzers is not None else DEFAULT_ANALYZERS
    out: dict[str, Any] = {}
    for a in analyzers:
        try:
            out[a.name] = a.analyze(inp)
        except Exception as e:
            out[a.name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def format_for_prompt(signals: dict[str, Any], unreplayed_mutations: int = 0) -> str:
    """Compact human-target v2 digest for inclusion in LLM prompts.

    `unreplayed_mutations` > 0 means the replayed workdir has drifted from the
    agent's real repo (shell writes / apply_patch / rm not reproduced), so the
    signals below are PARTIAL — the prompt is told to weight them accordingly.
    """
    lines = ["Static/state signals for human-target v2:"]
    if unreplayed_mutations > 0:
        lines.append(
            f"  - replay fidelity reduced: {unreplayed_mutations} earlier "
            f"file-mutating action(s) were not reproduced; treat missing signals as partial."
        )

    dl = signals.get("direction_lock") or {}
    if dl.get("post_file_parses") is False:
        lines.append(f"  - wrong_abstraction evidence: post-action file has syntax error: {dl.get('syntax_error_msg')}")
    if dl.get("inserted_at_wrong_scope"):
        lines.append("  - wrong_abstraction evidence: insertion appears to place module-scope code inside a function body.")

    fp = signals.get("fragility_patterns") or {}
    if fp:
        items = [
            f"bare_except={fp.get('bare_excepts', 0)}",
            f"broad_except={fp.get('broad_excepts', 0)}",
            f"silent_except={fp.get('silent_excepts', 0)}",
            f"hardcoded_paths={fp.get('hardcoded_paths', 0)}",
            f"version_literals={fp.get('version_literals', 0)}",
        ]
        lines.append(f"  - fragility/observability hints: {', '.join(items)}")

    ld = signals.get("lint_delta") or {}
    if ld.get("ruff_warnings_delta") is not None:
        lines.append(
            f"  - debt/fragility hint: ruff warnings "
            f"{ld['ruff_warnings_base']} -> {ld['ruff_warnings_post']} "
            f"(delta {ld['ruff_warnings_delta']:+d})"
        )

    bm = signals.get("breadth_metrics") or {}
    if bm:
        lines.append(
            f"  - broad_rewrite evidence: cumulative diff {bm.get('cumulative_files_changed', 0)} files, "
            f"+{bm.get('cumulative_lines_added', 0)} / -{bm.get('cumulative_lines_removed', 0)} lines"
        )
        if bm.get("max_single_file_changed_lines", 0) > 100:
            lines.append(
                f"    largest single-file change: "
                f"{bm['max_single_file_changed_lines']} lines"
            )

    asc = signals.get("artifact_scan") or {}
    introduced = asc.get("artifacts_introduced_this_action") or []
    if introduced:
        lines.append(
            f"  - artifact_residue evidence: this action introduces ad-hoc artifact file(s): "
            f"{introduced}"
        )
    deps_new = asc.get("deps_introduced_this_action") or []
    if deps_new:
        lines.append(
            f"  - artifact_residue evidence: this action modifies dependency/build file(s): {deps_new}"
        )
    carried = [p for p in (asc.get("artifacts_still_present") or [])
               if p not in introduced]
    if carried:
        lines.append(
            f"  - artifact context: ad-hoc artifact(s) already present from earlier actions: {carried}"
        )

    if len(lines) == 1:
        lines.append("  - no notable static/state signal")
    return "\n".join(lines)
