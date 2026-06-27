"""Common interface for fast local-scope static analyzers.

Constraint: each analyzer.analyze(...) must complete in <1s per action.
No whole-repo type checking, no test execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_judge.state import TrajectoryState
from repo_env.snapshot import CodebaseSnapshot


@dataclass
class AnalyzerInput:
    """Bundle of context passed to every analyzer for one action."""

    snapshot: CodebaseSnapshot   # post-state (after this action applied)
    action: dict                 # the normalized action dict
    state: TrajectoryState       # cumulative trajectory state (snapshot)
    workdir: Path                # snapshot.workdir, surfaced for convenience
    base_commit: str             # SHA of the trajectory's base commit


class StaticAnalyzer(Protocol):
    """All analyzers expose a stable name and return flat dict of signals."""

    name: str

    def analyze(self, inp: AnalyzerInput) -> dict[str, Any]: ...


def is_python_file(path: str) -> bool:
    return path.endswith(".py")
