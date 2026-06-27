"""BreadthMetrics analyzer — `regression_surface` + Lock.B signal provider.

Pulls hard numbers from git for the current cumulative diff vs base_commit:
  - n_files_changed       (cumulative)
  - n_lines_added
  - n_lines_removed
  - max_single_file_changed_lines  (largest file delta)
  - this_action_lines_changed       (just for this step)

Sub-second: uses git diff --stat, no AST parsing.
"""

from __future__ import annotations

import subprocess
from typing import Any

from static_analysis.base import AnalyzerInput


def _git(workdir, *args, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


class BreadthMetricsAnalyzer:
    name = "breadth_metrics"

    def analyze(self, inp: AnalyzerInput) -> dict[str, Any]:
        # Stage working-tree changes so we can include untracked new files.
        _git(inp.workdir, "add", "-A")
        numstat = _git(inp.workdir, "diff", "--cached", "--numstat", inp.base_commit, "--")

        n_files = 0
        added = 0
        removed = 0
        max_single = 0
        for line in numstat.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            a, r, _path = parts[0], parts[1], parts[2]
            # binary files show '-' for both columns
            a_n = int(a) if a.isdigit() else 0
            r_n = int(r) if r.isdigit() else 0
            n_files += 1
            added += a_n
            removed += r_n
            max_single = max(max_single, a_n + r_n)

        # Approximate lines changed by THIS action: from action body text.
        body = inp.action.get("raw_action") or ""
        this_action_lines = body.count("\n") if isinstance(body, str) else 0

        return {
            "cumulative_files_changed": n_files,
            "cumulative_lines_added": added,
            "cumulative_lines_removed": removed,
            "cumulative_lines_total": added + removed,
            "max_single_file_changed_lines": max_single,
            "this_action_lines_in_raw": this_action_lines,
        }
