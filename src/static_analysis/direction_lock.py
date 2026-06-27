"""DirectionLock analyzer — primary Lock.A signal provider.

Detects "premature direction commitment" by checking that the post-action
file is syntactically valid. The dominant Lock.A FN pattern from §2.3 v5
diagnosis was: agent inserts code at the wrong line, corrupting the file's
syntax — e.g., a struct/function definition inserted INSIDE another function
body. The LLM, reading the inserted text in isolation, judges it as
semantically correct ("standard Go functional options pattern") and misses
that the structural placement broke the file.

Approach (fast, local-scope, Python-first):
  - On a `.py` action, run `ast.parse(file_contents)`. If it raises
    SyntaxError, the post-action file is broken → strong Lock.A signal.
  - Also flag if the inserted code lives within an existing function body
    when it ought to be at module scope (struct/class/top-level def patterns).
  - Cheap, deterministic, no LLM, no test execution.

Phase 2.4c v1 limits: Python only. Go/TS extension goes to Phase 3.3.
"""

from __future__ import annotations

import ast
from typing import Any

from static_analysis.base import AnalyzerInput, is_python_file


# Patterns that are SUPPOSED to live at module scope. If we see them ending up
# inside another function body, that's structural confusion.
_MODULE_SCOPE_DEF_HINTS = (
    "def ",        # top-level function
    "class ",      # top-level class
    "@dataclass",  # decorator suggesting class
    "ImportOpt =",
)


class DirectionLockAnalyzer:
    """Outputs structural-validity signals for the current action's target file."""

    name = "direction_lock"

    def analyze(self, inp: AnalyzerInput) -> dict[str, Any]:
        out: dict[str, Any] = {
            "post_file_parses": None,         # True | False | None (not python)
            "syntax_error_msg": None,
            "inserted_at_wrong_scope": False,
            "candidate_paths_analyzed": [],
        }

        targets = inp.snapshot.changed_files or []
        if not targets:
            return out

        py_targets = [p for p in targets if is_python_file(p)]
        out["candidate_paths_analyzed"] = py_targets
        if not py_targets:
            return out

        # Analyze the first python target (most actions touch one file).
        path = py_targets[0]
        content = inp.snapshot.read_file(path)
        if content is None:
            return out

        try:
            tree = ast.parse(content)
            out["post_file_parses"] = True
        except SyntaxError as e:
            out["post_file_parses"] = False
            out["syntax_error_msg"] = f"line {e.lineno}: {e.msg}"
            return out

        # Heuristic: if the action is an `insert`, peek at the inserted text
        # and see if it contains module-scope-only constructs. Then check if
        # the inserted code ended up inside some function body in the parsed
        # AST. We approximate by checking the textual line range.
        nm = inp.action.get("normalized") or {}
        if nm.get("operation") != "insert":
            return out
        raw = inp.action.get("raw_action") or ""
        if not any(hint in raw for hint in _MODULE_SCOPE_DEF_HINTS):
            return out

        # Find the line number the action inserted at. Pull from raw_action's
        # --insert_line flag via the normalized payload (we don't re-parse).
        # Cheap textual scan: look for "--insert_line <N>" in raw.
        line_no = _extract_insert_line(raw)
        if line_no is None:
            return out

        # Walk all function/method bodies; if the insert line falls strictly
        # inside one (between its def's lineno and its body's last line),
        # that's a structural placement violation.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_start = node.lineno + 1
                body_end = getattr(node, "end_lineno", node.lineno) or node.lineno
                if body_start <= line_no <= body_end:
                    out["inserted_at_wrong_scope"] = True
                    break

        return out


def _extract_insert_line(raw: str) -> int | None:
    """Find --insert_line N in the raw action text."""
    tok = "--insert_line"
    i = raw.find(tok)
    if i < 0:
        return None
    rest = raw[i + len(tok):].strip()
    if not rest:
        return None
    # Number runs until whitespace or end.
    num = ""
    for ch in rest:
        if ch.isdigit():
            num += ch
        else:
            break
    try:
        return int(num) if num else None
    except ValueError:
        return None
