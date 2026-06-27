"""LintDelta analyzer — `debt_density` signal via ruff warning count.

We run `ruff check --no-cache --output-format=json` on the changed file
post-action and on the same path's base_commit version (fetched via
`git show`). The delta of warning counts is a hard signal of debt
introduced by THIS action.

Performance: ruff per-file is sub-second. We tolerate ruff being absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from static_analysis.base import AnalyzerInput, is_python_file


def _ruff_warning_count(content: str) -> int | None:
    """Run ruff on a string content; return warning count or None if ruff unavailable."""
    ruff = shutil.which("ruff")
    if ruff is None:
        return None
    try:
        proc = subprocess.run(
            [ruff, "check", "--no-cache", "--output-format=json", "-"],
            input=content, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    # ruff returns rc=1 when there are issues, rc=0 when clean. JSON is in stdout.
    if not proc.stdout:
        return 0
    try:
        items = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return len(items) if isinstance(items, list) else None


def _git_show(workdir: Path, base_commit: str, path: str) -> str | None:
    """Fetch `git show <base>:<path>` content, or None if file didn't exist at base."""
    try:
        proc = subprocess.run(
            ["git", "show", f"{base_commit}:{path}"],
            cwd=workdir, capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


class LintDeltaAnalyzer:
    name = "lint_delta"

    def analyze(self, inp: AnalyzerInput) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ruff_warnings_post": None,
            "ruff_warnings_base": None,
            "ruff_warnings_delta": None,
            "candidate_paths_analyzed": [],
            "ruff_available": shutil.which("ruff") is not None,
        }
        py_targets = [p for p in inp.snapshot.changed_files if is_python_file(p)]
        out["candidate_paths_analyzed"] = py_targets
        if not py_targets or not out["ruff_available"]:
            return out
        path = py_targets[0]
        post = inp.snapshot.read_file(path)
        if post is None:
            return out
        post_count = _ruff_warning_count(post)
        out["ruff_warnings_post"] = post_count

        base_content = _git_show(inp.workdir, inp.base_commit, path)
        if base_content is not None:
            base_count = _ruff_warning_count(base_content)
            out["ruff_warnings_base"] = base_count
            if post_count is not None and base_count is not None:
                out["ruff_warnings_delta"] = post_count - base_count
        else:
            # File didn't exist at base → all post warnings are new.
            out["ruff_warnings_base"] = 0
            if post_count is not None:
                out["ruff_warnings_delta"] = post_count
        return out
