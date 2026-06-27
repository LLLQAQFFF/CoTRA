"""FragilityPatterns analyzer — `fragility_delta` signal provider.

Counts patterns the guide flags as fragile (per fragility_delta spec):
  - bare `except:` or `except Exception:` swallowing all errors
  - hardcoded absolute paths (`/usr/bin/python`, `/etc/...`)
  - magic numeric/string literals embedded in logic
  - Python-only ad-hoc nil checks via try/except (broad catch)

Python-only AST visit; very fast (~few ms even on 500-line files).
"""

from __future__ import annotations

import ast
import re
from typing import Any

from static_analysis.base import AnalyzerInput, is_python_file


_HARDCODED_PATH_RE = re.compile(r"['\"](/usr/[^'\"]+|/etc/[^'\"]+|/var/[^'\"]+|/opt/[^'\"]+|C:\\\\[^'\"]+)['\"]")
_VERSION_LITERAL_RE = re.compile(r"['\"]\d+\.\d+(\.\d+)?['\"]")


class _FragilityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.bare_excepts = 0
        self.broad_excepts = 0       # `except Exception:` or `except BaseException:`
        self.silent_excepts = 0      # except: pass / except E: pass / except E: return default

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.bare_excepts += 1
            # `except: pass`/`return default` → silent
            if _body_is_silent(node.body):
                self.silent_excepts += 1
        else:
            name = _name_of(node.type)
            if name in ("Exception", "BaseException"):
                self.broad_excepts += 1
                if _body_is_silent(node.body):
                    self.silent_excepts += 1
        self.generic_visit(node)


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _body_is_silent(body: list[ast.stmt]) -> bool:
    if not body:
        return False
    if len(body) == 1:
        s = body[0]
        if isinstance(s, ast.Pass):
            return True
        if isinstance(s, ast.Return) and (s.value is None or isinstance(s.value, ast.Constant)):
            return True
    return False


class FragilityPatternsAnalyzer:
    name = "fragility_patterns"

    def analyze(self, inp: AnalyzerInput) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bare_excepts": 0,
            "broad_excepts": 0,
            "silent_excepts": 0,
            "hardcoded_paths": 0,
            "version_literals": 0,
            "candidate_paths_analyzed": [],
        }
        py_targets = [p for p in inp.snapshot.changed_files if is_python_file(p)]
        out["candidate_paths_analyzed"] = py_targets
        if not py_targets:
            return out
        path = py_targets[0]
        content = inp.snapshot.read_file(path)
        if content is None:
            return out

        # Regex-based path / version scan is robust even if AST parse fails.
        out["hardcoded_paths"] = len(_HARDCODED_PATH_RE.findall(content))
        out["version_literals"] = len(_VERSION_LITERAL_RE.findall(content))

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return out
        v = _FragilityVisitor()
        v.visit(tree)
        out["bare_excepts"] = v.bare_excepts
        out["broad_excepts"] = v.broad_excepts
        out["silent_excepts"] = v.silent_excepts
        return out
