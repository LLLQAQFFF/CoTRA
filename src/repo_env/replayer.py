"""TrajectoryReplayer — orchestrates clone + checkout + step-wise apply.

Caches clones by (owner, repo) under `<cache_root>/<owner>/<repo>`. Each
replayer creates its own per-trajectory `workdir` by `git worktree`-ing off the
shared clone, so concurrent replays don't fight over a single checkout. After
teardown the worktree is removed; the shared clone stays for future reuse.

This module deliberately avoids running any tests / linters during step() — the
goal is to produce CodebaseSnapshot. Static analyzers consume the snapshots in
Phase 2.4c.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from repo_env.action_applier import FileEditHistory, apply_raw, skip_loses_fidelity
from repo_env.metadata import TrajectoryEnv
from repo_env.snapshot import CodebaseSnapshot


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 300) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    return proc.returncode, proc.stdout, proc.stderr


class ReplayError(RuntimeError):
    pass


class TrajectoryReplayer:
    """Clone-once, worktree-per-trajectory replayer."""

    def __init__(self, env: TrajectoryEnv, cache_root: Path,
                 workdir: Path | None = None) -> None:
        self.env = env
        self.cache_root = Path(cache_root)
        self._user_workdir = workdir          # if provided, no temp dir cleanup
        self._workdir: Path | None = None
        self._owned_workdir = workdir is None
        self._history = FileEditHistory()
        self._cumulative_changed: set[str] = set()
        self._action_count = 0
        self._unreplayed_mutations = 0
        self._is_worktree = False

    # ----- properties -------------------------------------------------------

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            raise ReplayError("workdir not set up; call .setup() first")
        return self._workdir

    @property
    def _clone_dir(self) -> Path:
        return self.cache_root / self.env.owner / self.env.repo

    # ----- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        """Ensure the shared clone exists, then create a worktree at the base commit."""
        self._ensure_clone()

        if self._user_workdir is not None:
            self._workdir = self._user_workdir
            self._workdir.mkdir(parents=True, exist_ok=True)
        else:
            self._workdir = Path(tempfile.mkdtemp(
                prefix=f"replay-{self.env.repo}-",
            ))

        # Create a detached worktree at base_commit so we don't pollute the
        # shared clone's working tree.
        rc, _, err = _run(
            ["git", "worktree", "add", "--detach", str(self._workdir),
             self.env.base_commit],
            cwd=self._clone_dir,
        )
        if rc != 0:
            # Fallback: maybe the commit isn't fetched. Try fetching by SHA.
            _run(["git", "fetch", "--depth", "1", "origin", self.env.base_commit],
                 cwd=self._clone_dir, timeout=600)
            rc, _, err = _run(
                ["git", "worktree", "add", "--detach", str(self._workdir),
                 self.env.base_commit],
                cwd=self._clone_dir,
            )
            if rc != 0:
                raise ReplayError(
                    f"git worktree add failed for {self.env.repo}@"
                    f"{self.env.base_commit[:12]}: {err.strip()}"
                )
        self._is_worktree = True

    def _ensure_clone(self) -> None:
        clone_dir = self._clone_dir
        if (clone_dir / ".git").exists() and _clone_is_ready(clone_dir):
            return
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_dir.mkdir()
        rc, _, err = _run(["git", "init"], cwd=clone_dir)
        if rc == 0:
            rc, _, err = _run(["git", "remote", "add", "origin", self.env.repo_url], cwd=clone_dir)
        if rc == 0:
            rc, _, err = _run(
                ["git", "fetch", "--depth", "1", "origin", self.env.base_commit],
                cwd=clone_dir,
                timeout=1800,
            )
        if rc == 0:
            rc, _, err = _run(
                ["git", "update-ref", "refs/heads/replay-cache", self.env.base_commit],
                cwd=clone_dir,
            )
        if rc == 0:
            rc, _, err = _run(
                ["git", "symbolic-ref", "HEAD", "refs/heads/replay-cache"],
                cwd=clone_dir,
            )
        if rc != 0:
            raise ReplayError(
                f"git cache initialization failed for {self.env.repo_url}: {err.strip()}"
            )

    def teardown(self) -> None:
        if self._workdir is None:
            return
        if self._is_worktree:
            # Detach the worktree before removing.
            _run(
                ["git", "worktree", "remove", "--force", str(self._workdir)],
                cwd=self._clone_dir,
            )
        if self._owned_workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None

    def __enter__(self) -> "TrajectoryReplayer":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.teardown()

    # ----- per-action -------------------------------------------------------

    def step(self, action: dict) -> CodebaseSnapshot:
        """Apply one action and return the resulting snapshot."""
        idx = action.get("action_index", self._action_count)
        raw = action.get("raw_action") or ""
        try:
            result = apply_raw(
                raw, self.workdir, self._history,
                container_workdir=self.env.container_workdir,
            )
        except Exception as exc:
            self._action_count += 1
            return CodebaseSnapshot(
                workdir=self.workdir,
                action_index=int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else self._action_count,
                changed_files=[],
                cumulative_changed_files=set(self._cumulative_changed),
                apply_failed=True,
                apply_failure_reason=f"{type(exc).__name__}: {exc}",
                unreplayed_mutations=self._unreplayed_mutations,
            )
        self._action_count += 1
        for f in result.changed_files:
            self._cumulative_changed.add(f)
        # If this action was skipped (non-str_replace_editor) but actually
        # mutates repo state, the workdir has drifted — track it.
        if result.reason == "tool-skipped" and skip_loses_fidelity(action):
            self._unreplayed_mutations += 1
        return CodebaseSnapshot(
            workdir=self.workdir,
            action_index=int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else self._action_count,
            changed_files=list(result.changed_files),
            cumulative_changed_files=set(self._cumulative_changed),
            apply_failed=not result.success,
            apply_failure_reason=result.reason,
            unreplayed_mutations=self._unreplayed_mutations,
        )

    # ----- inspectors -------------------------------------------------------

    def cumulative_diff(self) -> str:
        """git diff vs base_commit, including untracked new files.

        We stage all working-tree changes (including new files) into the index
        so a single `diff --cached` shows the full picture. The index mutation
        is harmless inside a per-trajectory worktree that gets torn down.
        """
        _run(["git", "add", "-A"], cwd=self.workdir, timeout=60)
        rc, out, _ = _run(
            ["git", "diff", "--cached", "--no-color", self.env.base_commit, "--"],
            cwd=self.workdir, timeout=60,
        )
        return out if rc == 0 else ""


def _clone_is_ready(clone_dir: Path) -> bool:
    rc, _, _ = _run(["git", "rev-parse", "--verify", "HEAD"], cwd=clone_dir)
    return rc == 0
