"""Parse and apply normalized actions onto a workdir.

We focus on the `str_replace_editor` tool used by SWE-Bench Pro agents, which
has 5 sub-operations: view / view_range / create / str_replace / insert /
undo_edit. Shell-based writes (echo > file, tee, etc.) are out of scope for
v1 — they're rare in our set2 sample and add a lot of parsing complexity.

The applier is "tolerant by default": malformed actions or failed patches
return ApplyResult(success=False, reason=...) but do NOT raise, so the
replayer can keep advancing through the trajectory.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParsedAction:
    """A normalized + parsed action ready to be applied."""

    tool: str            # 'str_replace_editor' / 'bash' / ...
    operation: str       # 'create' / 'str_replace' / 'insert' / 'undo_edit' / 'view' / ...
    path: str | None     # repo-relative path (container prefix stripped)
    flags: dict          # {'new_str': '...', 'old_str': '...', 'insert_line': '123', 'file_text': '...'}


@dataclass
class ApplyResult:
    """Outcome of applying a single action to the workdir."""

    success: bool
    reason: str = ""                                # error description if any
    changed_files: list[str] = field(default_factory=list)
    op: str = ""                                    # echoes ParsedAction.operation


# ---------- container-path normalization -----------------------------------

# Common in-container roots used by SWE-Bench Pro agents. Stripped to repo-relative.
_CONTAINER_PREFIXES = ("/app/", "/workspace/", "/repo/")


def _to_repo_relative(p: str, container_workdir: str = "/app") -> str:
    """Strip the in-container working directory prefix so paths are repo-relative."""
    if not p:
        return p
    wd = container_workdir.rstrip("/") + "/"
    if p.startswith(wd):
        return p[len(wd):]
    for pref in _CONTAINER_PREFIXES:
        if p.startswith(pref):
            return p[len(pref):]
    return p.lstrip("/")


# ---------- parsing --------------------------------------------------------

# `str_replace_editor` raw_action format:
#   str_replace_editor <op> <path>   [--flag1 'value1' --flag2 'value2' ...]
# Values may contain newlines and embedded single quotes (escaped as '"'"').

def parse_raw_action(raw: str, container_workdir: str = "/app") -> ParsedAction | None:
    """Parse a raw_action string into a ParsedAction. Returns None on unknown shape."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    tool = tokens[0]
    if tool != "str_replace_editor":
        # v1: handle only str_replace_editor. Other tools (bash, etc.) are skipped.
        return ParsedAction(tool=tool, operation="", path=None, flags={})

    if len(tokens) < 2:
        return ParsedAction(tool=tool, operation="", path=None, flags={})

    operation = tokens[1]
    path = tokens[2] if len(tokens) > 2 and not tokens[2].startswith("--") else None
    rel_path = _to_repo_relative(path, container_workdir) if path else None

    flags: dict = {}
    i = 3 if path else 2
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            name = tok[2:]
            # Value is the next non-flag token if any. Allow flags without values.
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                flags[name] = tokens[i + 1]
                i += 2
            else:
                flags[name] = True
                i += 1
        else:
            i += 1

    return ParsedAction(tool=tool, operation=operation, path=rel_path, flags=flags)


# ---------- application ----------------------------------------------------


class FileEditHistory:
    """Per-file stack of (prior_content) snapshots, for `undo_edit` support."""

    def __init__(self) -> None:
        self._stacks: dict[str, list[str | None]] = {}

    def record_before(self, path: str, prior: str | None) -> None:
        self._stacks.setdefault(path, []).append(prior)

    def pop_prior(self, path: str) -> tuple[bool, str | None]:
        stack = self._stacks.get(path)
        if not stack:
            return False, None
        return True, stack.pop()


def _read_text(p: Path) -> str | None:
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="utf-8", errors="replace")


def _write_text(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def apply_parsed(parsed: ParsedAction, workdir: Path,
                 history: FileEditHistory) -> ApplyResult:
    """Apply one parsed action to workdir. Tolerant: failures are reported."""
    if parsed.tool != "str_replace_editor":
        return ApplyResult(success=True, reason="tool-skipped", op=parsed.operation)

    op = parsed.operation
    if op in ("view", "view_range", ""):
        return ApplyResult(success=True, reason="readonly", op=op)

    if not parsed.path:
        return ApplyResult(success=False, reason="missing-path", op=op)

    target = workdir / parsed.path

    if op == "create":
        body = parsed.flags.get("file_text", "")
        if not isinstance(body, str):
            body = ""
        prior = _read_text(target)
        history.record_before(parsed.path, prior)
        _write_text(target, body)
        return ApplyResult(success=True, op=op, changed_files=[parsed.path])

    if op == "str_replace":
        old = parsed.flags.get("old_str", "")
        new = parsed.flags.get("new_str", "")
        if not isinstance(old, str) or not isinstance(new, str):
            return ApplyResult(success=False, reason="bad-args", op=op)
        prior = _read_text(target)
        if prior is None:
            return ApplyResult(success=False, reason="file-not-found", op=op)
        if old and old not in prior:
            return ApplyResult(success=False, reason="old-str-not-found", op=op)
        new_content = prior.replace(old, new, 1) if old else (prior + new)
        history.record_before(parsed.path, prior)
        _write_text(target, new_content)
        return ApplyResult(success=True, op=op, changed_files=[parsed.path])

    if op == "insert":
        new = parsed.flags.get("new_str", "")
        line_str = parsed.flags.get("insert_line", "0")
        try:
            line = int(line_str)
        except (TypeError, ValueError):
            return ApplyResult(success=False, reason="bad-insert-line", op=op)
        if not isinstance(new, str):
            return ApplyResult(success=False, reason="bad-args", op=op)
        prior = _read_text(target)
        if prior is None:
            return ApplyResult(success=False, reason="file-not-found", op=op)
        lines = prior.split("\n")
        # SWE-Bench `insert_line=N` inserts AFTER line N (1-indexed). N=0 → top.
        insert_at = max(0, min(line, len(lines)))
        new_lines = lines[:insert_at] + new.split("\n") + lines[insert_at:]
        new_content = "\n".join(new_lines)
        history.record_before(parsed.path, prior)
        _write_text(target, new_content)
        return ApplyResult(success=True, op=op, changed_files=[parsed.path])

    if op == "undo_edit":
        ok, prior = history.pop_prior(parsed.path)
        if not ok:
            return ApplyResult(success=False, reason="no-prior-edit", op=op)
        if prior is None:
            # File was created by the popped edit → undo means delete.
            if target.exists():
                target.unlink()
        else:
            _write_text(target, prior)
        return ApplyResult(success=True, op=op, changed_files=[parsed.path])

    return ApplyResult(success=False, reason=f"unknown-op:{op}", op=op)


def apply_raw(raw_action: str, workdir: Path, history: FileEditHistory,
               container_workdir: str = "/app") -> ApplyResult:
    """Convenience: parse + apply in one call."""
    parsed = parse_raw_action(raw_action, container_workdir=container_workdir)
    if parsed is None:
        return ApplyResult(success=False, reason="parse-failed")
    return apply_parsed(parsed, workdir, history)


# ---------- replay fidelity ------------------------------------------------

# Tools whose effects don't change repo SOURCE state — skipping them loses no
# fidelity for static analysis (env setup, perms, directory scaffolding).
_FIDELITY_NEUTRAL_TOOLS = {
    "mkdir", "chmod", "npm", "pip", "pip3", "apt-get", "apt", "apt-cache",
}
_FIDELITY_NEUTRAL_OPS = {"mkdir", "chmod", "install"}


def skip_loses_fidelity(action: dict) -> bool:
    """True if skipping this (non-str_replace_editor) action means the replayed
    workdir diverges from the agent's real repo state — i.e. static signals
    computed afterward may be incomplete.

    Examples that lose fidelity: `apply_patch`, `rm`, `mv`, `cp`,
    `echo/printf > file`, `cat > file`, `touch`. Examples that don't: `mkdir`,
    `chmod`, `npm/pip install`, plain reads.
    """
    n = action.get("normalized") or {}
    tool = (n.get("tool") or "").lower()
    op = (n.get("operation") or "").lower()
    cls = (n.get("action_class") or "").lower()
    if tool == "str_replace_editor":
        return False  # handled natively
    if tool in _FIDELITY_NEUTRAL_TOOLS or op in _FIDELITY_NEUTRAL_OPS:
        return False
    # A write/patch-class action from any other tool mutates repo state we
    # cannot reproduce in v1.
    return cls in ("write", "patch")
