"""TrajectoryState — cumulative trajectory shape for per-action LLM judging.

Built sequentially while iterating effectful actions. At each effectful action
index we snapshot `summary_for_lock()` *before* updating, so the snapshot
describes "what has happened before this action" — which is exactly what the
LLM lacks when it scores actions in isolation (it only sees 3 priors).

Scope (Phase 2.4a, v1):
  - Lock.B signals: distinct paths, broad-op count, rolling breadth, revisits.
  - Lock.C signals: self-authored test/script artifacts, assertion edits.
  - Lock.D signals: dependency-file modifications, root-level artifact files.
  - Phase indicator: action_index thirds (exploration/implementation/cleanup).
  - Lock.A signals: intentionally deferred to Phase 2.4c (static analysis),
    since "early committed direction is still active" needs AST inspection.

This module is pure Python — no LLM, no I/O. Unit-testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# --- artifact / dep-file pattern lists (see manual_annotation_guide §D) -----

_DEP_FILES = {
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "go.mod", "go.sum",
    "Cargo.toml", "Cargo.lock",
    "requirements.txt", "pyproject.toml", "poetry.lock", "Pipfile.lock",
    "Gemfile.lock", "composer.lock",
}

_ARTIFACT_NAME_RE = re.compile(
    r"^(test_[A-Za-z0-9_]+\.(py|ts|js|go)"      # test_X.py / test_X.ts
    r"|repro[A-Za-z0-9_]*\.(py|ts|js|go|sh)"    # repro.py, repro_bug.sh
    r"|fix_[A-Za-z0-9_]+\.(py|ts|js|sh)"        # fix_keyinfo.py
    r"|update_[A-Za-z0-9_]+_helpers?\.py"       # update_X_helpers.py
    r"|check_[A-Za-z0-9_]+\.sh"                 # check_X.sh
    r"|[A-Za-z0-9_]+_backup\.(py|ts|js|go)"     # X_backup.py
    r"|[A-Za-z0-9_./]+\.(bak|orig)"             # X.bak / X.orig
    r")$"
)

_TEST_FILE_RE = re.compile(r"(^|/)(test_[A-Za-z0-9_]+|[A-Za-z0-9_]+_test)\.(py|ts|js|go)$")


def _is_artifact_path(path: str) -> bool:
    """Top-level / non-tests-dir ad-hoc script matching guide's D patterns."""
    if not path:
        return False
    parts = path.split("/")
    # If file lives under tests/ or scripts/, NOT an artifact.
    for p in parts[:-1]:
        if p in ("tests", "test", "scripts"):
            return False
    return _ARTIFACT_NAME_RE.match(parts[-1]) is not None


def _is_test_file(path: str) -> bool:
    return bool(path) and _TEST_FILE_RE.search(path) is not None


def _is_dep_file(path: str) -> bool:
    if not path:
        return False
    name = path.split("/")[-1]
    if name in _DEP_FILES:
        return True
    return name.endswith(".lock")


def _action_body(action: dict) -> str:
    """Best-effort retrieval of patch/body text. In our schema the actual patch
    content lives in `raw_action` (the original tool call text); the
    `normalized` dict only carries operation/path metadata."""
    raw = action.get("raw_action")
    if isinstance(raw, str):
        return raw
    # Fallback: some pipelines may pre-extract a body field into normalized.
    n = action.get("normalized") or {}
    for k in ("body", "patch", "content", "new_body"):
        v = n.get(k)
        if isinstance(v, str):
            return v
    return ""


def _normalize_path(p: str) -> str:
    """Strip container prefixes (/app/, /workspace/) so artifact detection on
    the relative path works regardless of where the agent ran."""
    if not p:
        return p
    for prefix in ("/app/", "/workspace/", "/repo/"):
        if p.startswith(prefix):
            return p[len(prefix):]
    return p.lstrip("/")  # bare leading slash → relative


def _action_paths(action: dict) -> list[str]:
    """All target paths a normalized action touches (most ops only have one)."""
    n = action.get("normalized") or {}
    paths: list[str] = []
    # Real schema: target_file (str) + target_files (list[str]).
    for k in ("target_file", "path", "target_path"):
        v = n.get(k)
        if isinstance(v, str) and v:
            paths.append(v)
    for k in ("target_files", "paths"):
        v = n.get(k)
        if isinstance(v, list):
            paths.extend(str(p) for p in v if p)
    # Normalize and dedupe.
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        np = _normalize_path(p)
        if np and np not in seen:
            seen.add(np)
            out.append(np)
    return out


def _action_op(action: dict) -> str:
    n = action.get("normalized") or {}
    # Real schema uses `operation`; older draft used `op`.
    return (n.get("operation") or n.get("op") or "").lower()


# Operations that don't represent a forward edit (don't count toward breadth).
_NON_FORWARD_OPS = {"undo_edit", "view", "view_range", "list_dir", "ls", "read"}


def _is_forward_edit(action: dict) -> bool:
    return _action_op(action) not in _NON_FORWARD_OPS


def _is_broad_op(action: dict) -> bool:
    """Heuristic: did this action affect many lines / files?"""
    if not _is_forward_edit(action):
        return False
    op = _action_op(action)
    if any(tag in op for tag in ("_multiple", "rewrite", "bulk", "create")):
        if op == "create" and len(_action_paths(action)) == 1:
            # `create` on a single file isn't automatically broad — only if the
            # body is large (handled below).
            pass
        else:
            return True
    body = _action_body(action)
    if body and body.count("\n") > 50:
        return True
    if len(_action_paths(action)) > 1:
        return True
    return False


def _lines_changed(action: dict) -> int:
    if not _is_forward_edit(action):
        return 0
    body = _action_body(action)
    return body.count("\n") if body else 0


@dataclass
class TrajectoryState:
    """Cumulative trajectory state — updated per effectful action."""

    # Lock.B
    paths_touched: dict[str, int] = field(default_factory=dict)
    n_broad_ops: int = 0
    rolling_breadth_last5: list[int] = field(default_factory=list)

    # Lock.C
    self_test_files: set[str] = field(default_factory=set)
    self_assertion_edits: int = 0

    # Lock.D
    artifact_files: set[str] = field(default_factory=set)
    deps_modified: bool = False
    deps_modified_files: set[str] = field(default_factory=set)

    # Meta
    n_actions_seen: int = 0
    total_actions_in_trajectory: int = 0  # set once at start

    # --- update --------------------------------------------------------------

    def update(self, action: dict) -> None:
        """Incorporate a single effectful action into the state."""
        self.n_actions_seen += 1

        # Non-forward ops (undo_edit / view) don't change repo state.
        # We still count them in n_actions_seen for phase tracking, but skip
        # path/breadth bookkeeping.
        if not _is_forward_edit(action):
            self.rolling_breadth_last5.append(0)
            if len(self.rolling_breadth_last5) > 5:
                self.rolling_breadth_last5 = self.rolling_breadth_last5[-5:]
            return

        # Path / breadth bookkeeping (Lock.B + rs)
        op = _action_op(action)
        for p in _action_paths(action):
            self.paths_touched[p] = self.paths_touched.get(p, 0) + 1

            if _is_dep_file(p):
                self.deps_modified = True
                self.deps_modified_files.add(p)

            if _is_artifact_path(p):
                self.artifact_files.add(p)

            if _is_test_file(p):
                if op == "create":
                    self.self_test_files.add(p)
                elif p in self.self_test_files and op in ("edit", "patch", "modify",
                                                          "insert", "str_replace"):
                    # Heuristic: assertion edit if body mentions assert/expect/should
                    body = _action_body(action)
                    if any(kw in body for kw in ("assert", "expect(", "should",
                                                  ".to_be", ".toEqual")):
                        self.self_assertion_edits += 1

        if _is_broad_op(action):
            self.n_broad_ops += 1

        lc = _lines_changed(action)
        self.rolling_breadth_last5.append(lc)
        if len(self.rolling_breadth_last5) > 5:
            self.rolling_breadth_last5 = self.rolling_breadth_last5[-5:]

    # --- queries -------------------------------------------------------------

    @property
    def n_distinct_paths(self) -> int:
        return len(self.paths_touched)

    @property
    def n_revisits(self) -> int:
        return sum(1 for c in self.paths_touched.values() if c >= 2)

    @property
    def phase(self) -> str:
        if self.total_actions_in_trajectory <= 0:
            return "implementation"
        ratio = self.n_actions_seen / self.total_actions_in_trajectory
        if ratio < 0.25:
            return "exploration"
        if ratio < 0.75:
            return "implementation"
        return "cleanup"

    # --- serialization for LLM prompts --------------------------------------

    def summary_for_lock(self) -> str:
        """Compact markdown digest fed into the Lock prompt's system area."""
        if self.n_actions_seen == 0:
            return "(no prior effectful actions in this trajectory)"

        top_paths = sorted(
            self.paths_touched.items(), key=lambda kv: -kv[1]
        )[:5]
        path_lines = [f"    - {p} × {c}" for p, c in top_paths]

        artifacts = sorted(self.artifact_files)[:5]
        artifact_line = (
            "  Top-level / non-tests artifact files left in tree: "
            + (str(artifacts) if artifacts else "none")
        )

        self_tests = sorted(self.self_test_files)[:5]
        self_test_line = (
            f"  Self-authored test files created: "
            + (str(self_tests) if self_tests else "none")
            + (f" ({self.self_assertion_edits} assertion edits after creation)" if self.self_assertion_edits else "")
        )

        deps = sorted(self.deps_modified_files)[:5]
        deps_line = (
            f"  Dependency / build files modified: "
            + (str(deps) if deps else "none")
        )

        if self.total_actions_in_trajectory > 0:
            phase_line = (
                f"  Phase: {self.phase} (action "
                f"{self.n_actions_seen + 1}/{self.total_actions_in_trajectory})"
            )
        else:
            phase_line = f"  Phase: {self.phase} (action {self.n_actions_seen + 1}/?)"

        return (
            f"Cumulative trajectory state (after {self.n_actions_seen} prior effectful actions):\n"
            f"{phase_line}\n"
            f"  Distinct files touched so far: {self.n_distinct_paths} "
            f"({self.n_revisits} re-edited)\n"
            f"  Broad operations accumulated: {self.n_broad_ops}\n"
            f"  Rolling breadth (last 5 actions, lines changed): "
            f"{self.rolling_breadth_last5 or '[]'}\n"
            f"  Top revisited paths:\n"
            + ("\n".join(path_lines) if path_lines else "    (none)")
            + "\n"
            + self_test_line + "\n"
            + artifact_line + "\n"
            + deps_line
        )
