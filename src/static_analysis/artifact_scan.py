"""ArtifactScan analyzer — Lock.C/D artifact signal.

Runs per-action. Its primary per-action signal is
`artifacts_introduced_this_action`: ad-hoc artifact file(s) that THIS action
adds to the tree for the first time — the point at which a Lock.C/D artifact
lock is established. Likewise `deps_introduced_this_action` for dependency /
build files.

`artifacts_still_present` is also reported, but it is a trajectory-TERMINAL
check ("does the artifact persist in the submitted diff"). Mid-trajectory it
is true for every action after the artifact's creation, so it must NOT be
used as a per-action Lock.D signal — doing so broadcasts a single lock across
every later action that happens to leave the artifact in place.
"""

from __future__ import annotations

from typing import Any

from llm_judge.state import _is_artifact_path, _is_dep_file
from static_analysis.base import AnalyzerInput


class ArtifactScanAnalyzer:
    name = "artifact_scan"

    def analyze(self, inp: AnalyzerInput) -> dict[str, Any]:
        # Use the state machine's accumulated artifact list + verify each path
        # actually still exists in the workdir snapshot.
        flagged = sorted(inp.state.artifact_files)
        surviving: list[str] = []
        for p in flagged:
            if (inp.workdir / p).exists():
                surviving.append(p)
        deps_flagged = sorted(inp.state.deps_modified_files)
        deps_surviving = [p for p in deps_flagged if (inp.workdir / p).exists()]

        # Per-action establishment signal: artifact / dep files this action
        # adds for the FIRST time. `inp.state` is the PRE-action snapshot (the
        # judge folds the action into the state only after analysis runs), so
        # a path in this action's changed_files but absent from the state was
        # introduced by THIS action — and this is the action on which the
        # Lock.C/D should be scored.
        pre_artifacts = set(inp.state.artifact_files)
        pre_deps = set(inp.state.deps_modified_files)
        changed = inp.snapshot.changed_files or []
        artifacts_introduced = sorted(
            p for p in changed if _is_artifact_path(p) and p not in pre_artifacts
        )
        deps_introduced = sorted(
            p for p in changed if _is_dep_file(p) and p not in pre_deps
        )

        # Also scan top-level for any artifact-like file we may have missed
        # (defense in depth — state machine could miss create-via-bash).
        late_found: list[str] = []
        try:
            for entry in inp.workdir.iterdir():
                if not entry.is_file():
                    continue
                rel = entry.name
                if _is_artifact_path(rel) and rel not in inp.state.artifact_files:
                    late_found.append(rel)
        except OSError:
            pass

        return {
            "artifacts_introduced_this_action": artifacts_introduced,
            "deps_introduced_this_action": deps_introduced,
            "artifacts_flagged_by_state": flagged,
            "artifacts_still_present": surviving,
            "artifacts_pruned_before_end": [p for p in flagged if p not in surviving],
            "deps_files_modified_still_present": deps_surviving,
            "artifact_files_found_outside_state": late_found,
        }
