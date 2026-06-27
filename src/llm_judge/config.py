"""Runtime configuration for the human-target v2 LLM judge."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_cfg: dict = {}
_cfg_path = _PROJECT_ROOT / "llm_config.py"
if _cfg_path.exists():
    exec(_cfg_path.read_text(), _cfg)
_local_cfg_path = _PROJECT_ROOT / "llm_config.local.py"
if _local_cfg_path.exists():
    exec(_local_cfg_path.read_text(), _cfg)
_extra_cfg = os.environ.get("LLM_CONFIG_EXTRA")
if _extra_cfg:
    _extra_cfg_path = Path(_extra_cfg)
    if not _extra_cfg_path.is_absolute():
        _extra_cfg_path = _PROJECT_ROOT / _extra_cfg_path
    exec(_extra_cfg_path.read_text(), _cfg)

API_BASE_URL: str = os.environ.get("LLM_BASE_URL") or _cfg.get("LLM_BASE_URL", "https://api.deepseek.com")
API_KEY: str = (
    os.environ.get("LLM_API_KEY")
    or os.environ.get("DEEPSEEK_API_KEY")
    or _cfg.get("LLM_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)
PRICING_CURRENCY: str = os.environ.get("LLM_PRICING_CURRENCY") or _cfg.get("PRICING_CURRENCY", "USD")
PRICING_PER_M_TOKENS: dict = (
    _cfg.get("PRICING_PER_M_TOKENS")
    or _cfg.get("PRICING_USD_PER_M_TOKENS")
    or {}
)
# Backward-compatible alias for older callers.
PRICING_USD_PER_M_TOKENS: dict = PRICING_PER_M_TOKENS if PRICING_CURRENCY == "USD" else {}

DEFAULT_JUDGE_MODEL: str = os.environ.get("LLM_JUDGE_MODEL") or _cfg.get("LLM_JUDGE_MODEL", "glm-5.1")
JUDGE_FOR_CLAUDE: str = os.environ.get("LLM_JUDGE_FOR_CLAUDE") or DEFAULT_JUDGE_MODEL
JUDGE_FOR_NON_CLAUDE: str = os.environ.get("LLM_JUDGE_FOR_NON_CLAUDE") or DEFAULT_JUDGE_MODEL
_reasoning_env = os.environ.get("LLM_REASONING_EFFORT")
if _reasoning_env is not None:
    LLM_REASONING_EFFORT: str | None = None if _reasoning_env.lower() in {"", "none", "null"} else _reasoning_env
else:
    LLM_REASONING_EFFORT = _cfg.get("LLM_REASONING_EFFORT")
_pricing_raw = os.environ.get("LLM_PRICING_PER_M_TOKENS_JSON")
if _pricing_raw:
    try:
        PRICING_PER_M_TOKENS = json.loads(_pricing_raw)
    except json.JSONDecodeError:
        pass
if PRICING_CURRENCY == "USD":
    PRICING_USD_PER_M_TOKENS = PRICING_PER_M_TOKENS

_extra_body_raw = os.environ.get("LLM_EXTRA_BODY_JSON")
if _extra_body_raw:
    try:
        LLM_EXTRA_BODY: dict | None = json.loads(_extra_body_raw)
    except json.JSONDecodeError:
        LLM_EXTRA_BODY = None
else:
    LLM_EXTRA_BODY = _cfg.get("LLM_EXTRA_BODY")

MAX_TOKENS_ACTION: int = int(os.environ.get("LLM_MAX_TOKENS_ACTION") or _cfg.get("MAX_TOKENS_ACTION", 2048))
MAX_TOKENS_TRAJECTORY: int = int(os.environ.get("LLM_MAX_TOKENS_TRAJECTORY") or _cfg.get("MAX_TOKENS_TRAJECTORY", 2048))
MAX_TOKENS_E2E: int = int(os.environ.get("LLM_MAX_TOKENS_E2E") or _cfg.get("MAX_TOKENS_E2E", 8192))
# Backward-compatible names for older callers; the v2 judge uses ACTION/TRAJECTORY.
MAX_TOKENS_SCALAR: int = MAX_TOKENS_ACTION
MAX_TOKENS_LOCK: int = MAX_TOKENS_TRAJECTORY
TEMPERATURE: float = 0.0

ACTION_CONTENT_MAX_CHARS: int = 8000
CONTEXT_ACTION_MAX_CHARS: int = 600
CONTEXT_PRIOR_COUNT: int = 3

CONCURRENCY_PER_TRAJECTORY: int = int(os.environ.get("LLM_CONCURRENCY_PER_TRAJECTORY") or _cfg.get("CONCURRENCY_PER_TRAJECTORY", 6))

CACHE_DIR: Path = _PROJECT_ROOT / ".llm_judge_cache"

RETRY_MAX_ATTEMPTS: int = 4
RETRY_BACKOFF_SECONDS: float = 1.5
REQUEST_TIMEOUT_SECONDS: float = 180.0
ENCODER_REQUEST_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_ENCODER_REQUEST_TIMEOUT_SECONDS") or _cfg.get("ENCODER_REQUEST_TIMEOUT_SECONDS", 60.0))
ENCODER_RETRY_MAX_ATTEMPTS: int = int(os.environ.get("LLM_ENCODER_RETRY_MAX_ATTEMPTS") or _cfg.get("ENCODER_RETRY_MAX_ATTEMPTS", 1))
E2E_REQUEST_TIMEOUT_SECONDS: float = float(os.environ.get("LLM_E2E_REQUEST_TIMEOUT_SECONDS") or _cfg.get("E2E_REQUEST_TIMEOUT_SECONDS", 120.0))
E2E_RETRY_MAX_ATTEMPTS: int = int(os.environ.get("LLM_E2E_RETRY_MAX_ATTEMPTS") or _cfg.get("E2E_RETRY_MAX_ATTEMPTS", 1))
SCOPE_REVIEW_MAX_CANDIDATES: int = 40


@dataclass(frozen=True)
class JudgeRouting:
    claude_judge: str = JUDGE_FOR_CLAUDE
    non_claude_judge: str = JUDGE_FOR_NON_CLAUDE

    def for_family(self, family: str | None) -> str:
        """输入：可选的模型家族名。输出：评审模型名。作用：选择跨家族评审路由。"""
        if family is None:
            return self.non_claude_judge
        return self.claude_judge if family.lower() == "claude" else self.non_claude_judge


def get_routing() -> JudgeRouting:
    """输入：无。输出：JudgeRouting。作用：返回默认的评审路由配置。"""
    return JudgeRouting()
