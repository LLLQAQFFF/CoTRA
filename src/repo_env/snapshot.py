"""Per-step CodebaseSnapshot for downstream static analysis.

Lightweight on purpose: we don't snapshot whole-file contents (too much
memory for long trajectories). Instead we record paths + a cumulative
git-diff vs the base commit, and let the analyzer read files directly from
the workdir when needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodebaseSnapshot:
    """Snapshot of the workdir after applying an action."""

    workdir: Path                           # absolute path to checkout
    action_index: int                       # which action produced this snapshot
    changed_files: list[str] = field(default_factory=list)
    # Files written or deleted by the action that produced this snapshot.
    # Excludes the cumulative history.

    cumulative_changed_files: set[str] = field(default_factory=set)
    # Union of every file touched by any action since setup() (i.e. all files
    # that differ from base_commit).

    apply_failed: bool = False              # last action's apply succeeded?
    apply_failure_reason: str = ""

    unreplayed_mutations: int = 0
    # Cumulative count of file-mutating actions (apply_patch / rm / mv / cp /
    # shell writes) that the v1 replayer could NOT reproduce. When > 0, the
    # workdir has drifted from the agent's real repo state and static-analysis
    # signals on this snapshot should be treated as partial.

    def read_file(self, repo_relative_path: str) -> str | None:
        """Convenience: read a file from this snapshot's workdir."""
        p = self.workdir / repo_relative_path
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return p.read_text(encoding="utf-8", errors="replace")
