"""Token-usage and cost accounting for LLM judge runs.

Pricing source:
  Pulled from `llm_config.py` -> `PRICING_PER_M_TOKENS` if defined; otherwise
  cost is reported as None and only token counts / call counts are surfaced.
  Prices are read from configuration rather than hardcoded, because the
  effective per-token price depends on the API provider/endpoint in use.

Format expected in llm_config.py:
    PRICING_CURRENCY = "CNY"
    PRICING_PER_M_TOKENS = {
        "deepseek-v4-pro": {"input_cached": 0.025, "input": 3.0, "output": 6.0},
    }
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional

from llm_judge import config
from llm_judge.client import CallResult


_NEED_QUOTA_RE = re.compile(r"need quota:\s*[＄$]?\s*([0-9]+(?:\.[0-9]+)?)")
_REMAIN_QUOTA_RE = re.compile(r"token remain quota:\s*[＄$]?\s*([0-9]+(?:\.[0-9]+)?)")


def _pricing_for(model: str) -> Optional[dict]:
    """输入：模型名。输出：价格配置或 None。作用：查找每 token 价格。"""
    table = (
        getattr(config, "PRICING_PER_M_TOKENS", None)
        or getattr(config, "PRICING_USD_PER_M_TOKENS", None)
        or {}
    )
    entry = table.get(model)
    if not entry:
        return None
    if "input" not in entry or "output" not in entry:
        return None
    return entry


def compute_call_cost(
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_prompt_tokens: int | None = None,
) -> float | None:
    """输入：模型名和 token 数。输出：配置币种成本或 None。作用：计算一次非缓存 LLM 调用费用。"""
    p = _pricing_for(model)
    if p is None or prompt_tokens is None or completion_tokens is None:
        return None
    cached = max(0, min(cached_prompt_tokens or 0, prompt_tokens))
    uncached = max(0, prompt_tokens - cached)
    return (
        (uncached / 1_000_000.0) * p["input"]
        + (cached / 1_000_000.0) * p.get("input_cached", p["input"])
        + (completion_tokens / 1_000_000.0) * p["output"]
    )


@dataclass
class ModelStats:
    n_calls: int = 0
    n_cached: int = 0
    n_failed: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_amount: float = 0.0
    cost_unknown: bool = False  # True if any uncached call had no pricing
    failed_quota_required_usd: float = 0.0
    failed_quota_remaining_usd_last: float | None = None

    def as_dict(self) -> dict:
        """输入：模型统计实例。输出：可序列化字典。作用：导出四舍五入后的成本统计。"""
        return {
            "n_calls": self.n_calls,
            "n_cached": self.n_cached,
            "n_failed": self.n_failed,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "currency": config.PRICING_CURRENCY,
            "cost_amount": None if self.cost_unknown else round(self.cost_amount, 6),
            "cost_usd": (
                None
                if self.cost_unknown or config.PRICING_CURRENCY != "USD"
                else round(self.cost_amount, 6)
            ),
            "failed_quota_required_usd": round(self.failed_quota_required_usd, 6),
            "failed_quota_remaining_usd_last": (
                None
                if self.failed_quota_remaining_usd_last is None
                else round(self.failed_quota_remaining_usd_last, 6)
            ),
        }


class CostTracker:
    def __init__(self) -> None:
        """输入：无。输出：无。作用：初始化线程安全的逐模型成本计数器。"""
        self._lock = threading.Lock()
        self._by_model: dict[str, ModelStats] = {}

    def record(self, call: CallResult) -> None:
        """输入：已完成的调用结果。输出：无。作用：累计 token、缓存和成本数据。"""
        with self._lock:
            stats = self._by_model.setdefault(call.model, ModelStats())
            stats.n_calls += 1
            if call.cached:
                stats.n_cached += 1
                return
            if call.prompt_tokens is None and call.completion_tokens is None:
                # Provider returned no usage block. Don't fabricate.
                return
            stats.prompt_tokens += call.prompt_tokens or 0
            stats.cached_prompt_tokens += call.cached_prompt_tokens or 0
            stats.completion_tokens += call.completion_tokens or 0
            cost = compute_call_cost(
                call.model,
                call.prompt_tokens,
                call.completion_tokens,
                call.cached_prompt_tokens,
            )
            if cost is None:
                stats.cost_unknown = True
            else:
                stats.cost_amount += cost

    def record_failure(self, model: str, error: BaseException | str | None = None) -> None:
        """输入：模型名和可选错误。输出：无。作用：记录一次失败请求及网关额度提示。"""
        with self._lock:
            stats = self._by_model.setdefault(model, ModelStats())
            stats.n_failed += 1
            quota = extract_failure_quota(error)
            if quota["required_usd"] is not None:
                stats.failed_quota_required_usd += quota["required_usd"]
            if quota["remaining_usd"] is not None:
                stats.failed_quota_remaining_usd_last = quota["remaining_usd"]

    def snapshot(self) -> dict:
        """输入：无。输出：聚合统计字典。作用：为日志或输出捕获当前成本。"""
        with self._lock:
            return {
                "by_model": {m: s.as_dict() for m, s in self._by_model.items()},
                "totals": self._totals_locked(),
            }

    def _totals_locked(self) -> dict:
        """输入：已加锁的统计状态。输出：总计字典。作用：汇总所有模型的统计值。"""
        total_calls = sum(s.n_calls for s in self._by_model.values())
        total_cached = sum(s.n_cached for s in self._by_model.values())
        total_failed = sum(s.n_failed for s in self._by_model.values())
        total_pt = sum(s.prompt_tokens for s in self._by_model.values())
        total_cached_pt = sum(s.cached_prompt_tokens for s in self._by_model.values())
        total_ct = sum(s.completion_tokens for s in self._by_model.values())
        any_unknown = any(s.cost_unknown for s in self._by_model.values())
        total_cost = sum(s.cost_amount for s in self._by_model.values())
        failed_required = sum(s.failed_quota_required_usd for s in self._by_model.values())
        remaining_values = [
            s.failed_quota_remaining_usd_last
            for s in self._by_model.values()
            if s.failed_quota_remaining_usd_last is not None
        ]
        return {
            "n_calls": total_calls,
            "n_cached": total_cached,
            "n_failed": total_failed,
            "prompt_tokens": total_pt,
            "cached_prompt_tokens": total_cached_pt,
            "completion_tokens": total_ct,
            "currency": config.PRICING_CURRENCY,
            "cost_amount": None if any_unknown else round(total_cost, 6),
            "cost_usd": (
                None
                if any_unknown or config.PRICING_CURRENCY != "USD"
                else round(total_cost, 6)
            ),
            "failed_quota_required_usd": round(failed_required, 6),
            "failed_quota_remaining_usd_last": (
                round(remaining_values[-1], 6) if remaining_values else None
            ),
        }

    def fork(self) -> "CostTracker":
        """输入：无。输出：空的 CostTracker。作用：创建单条轨迹的成本统计器。"""
        return CostTracker()

    def merge(self, other: "CostTracker") -> None:
        """输入：另一个统计器。输出：无。作用：将子统计器合并到当前统计器。"""
        with self._lock, other._lock:
            for model, s in other._by_model.items():
                target = self._by_model.setdefault(model, ModelStats())
                target.n_calls += s.n_calls
                target.n_cached += s.n_cached
                target.n_failed += s.n_failed
                target.prompt_tokens += s.prompt_tokens
                target.cached_prompt_tokens += s.cached_prompt_tokens
                target.completion_tokens += s.completion_tokens
                target.cost_amount += s.cost_amount
                target.failed_quota_required_usd += s.failed_quota_required_usd
                if s.failed_quota_remaining_usd_last is not None:
                    target.failed_quota_remaining_usd_last = s.failed_quota_remaining_usd_last
                if s.cost_unknown:
                    target.cost_unknown = True


def extract_failure_quota(error: BaseException | str | None) -> dict[str, float | None]:
    """输入：异常或错误文本。输出：额度字段。作用：解析网关失败消息中的额度信息。"""
    if error is None:
        text = ""
    elif isinstance(error, str):
        text = error
    else:
        text = f"{type(error).__name__}: {error}"
    need = _NEED_QUOTA_RE.search(text)
    remain = _REMAIN_QUOTA_RE.search(text)
    return {
        "required_usd": float(need.group(1)) if need else None,
        "remaining_usd": float(remain.group(1)) if remain else None,
    }


def format_summary(snap: dict) -> str:
    """输入：成本快照字典。输出：可读文本。作用：汇总运行成本统计。"""
    lines = []
    currency = snap.get("totals", {}).get("currency") or config.PRICING_CURRENCY
    for model, s in snap["by_model"].items():
        cost = s.get("cost_amount", s.get("cost_usd"))
        cost_str = f"{cost:.4f} {currency}" if cost is not None else f"unknown {currency} (no pricing)"
        lines.append(
            f"  {model}: {s['n_calls']} calls ({s['n_cached']} cached, "
            f"{s['n_failed']} failed) | "
            f"in={s['prompt_tokens']} cached_in={s.get('cached_prompt_tokens', 0)} "
            f"out={s['completion_tokens']} tok | {cost_str}"
        )
        if s.get("failed_quota_required_usd", 0.0):
            lines.append(
                f"    failed quota required: ${s['failed_quota_required_usd']:.4f}"
            )
    t = snap["totals"]
    total_cost = t.get("cost_amount", t.get("cost_usd"))
    total_str = f"{total_cost:.4f} {currency}" if total_cost is not None else f"unknown {currency}"
    lines.append(
        f"  TOTAL: {t['n_calls']} calls ({t['n_cached']} cached, "
        f"{t['n_failed']} failed) | "
        f"in={t['prompt_tokens']} cached_in={t.get('cached_prompt_tokens', 0)} "
        f"out={t['completion_tokens']} tok | {total_str}"
    )
    if t.get("failed_quota_required_usd", 0.0):
        lines.append(
            f"  TOTAL failed quota required: ${t['failed_quota_required_usd']:.4f}"
        )
    return "\n".join(lines)
