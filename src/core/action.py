"""轨迹与动作的核心数据结构。

这些类型是 Phase 0 的稳定接口：加载器负责把原始 SWE-Bench Pro
轨迹转换为这些对象，后续评分器只依赖这里的字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.snapshot import CodebaseSnapshot


EFFECTFUL_ACTION_CLASSES = frozenset({"write", "patch", "install"})
FILESYSTEM_MUTATION_TAG = "filesystem_mutation"


@dataclass(frozen=True)
class ActionContext:
    """动作执行时可观察到的任务上下文。"""

    task_description: str
    agent_reasoning: str | None = None
    test_results_before: dict[str, Any] | None = None
    test_results_after: dict[str, Any] | None = None


@dataclass(frozen=True)
class NormalizedAction:
    """单步原始动作归一化后的稳定字段。"""

    step_index: int
    tool: str
    operation: str
    action_class: str
    target_file: str | None
    target_kind: str
    payload_type: str
    risk_tags: tuple[str, ...]
    raw_action: str
    wrapper_tool: str | None = None
    shell_command: str | None = None
    command_parts: tuple[str, ...] = ()

    @property
    def is_effectful(self) -> bool:
        """判断动作是否改变系统状态，供后续筛选 effectful action 使用。"""

        return (
            self.action_class in EFFECTFUL_ACTION_CLASSES
            or FILESYSTEM_MUTATION_TAG in self.risk_tags
        )


@dataclass(frozen=True)
class EvalSummary:
    """一次轨迹最终评测结果的简要统计。"""

    status_counts: dict[str, int] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def total(self) -> int:
        return sum(self.status_counts.values())

    @property
    def outcome(self) -> str:
        """把细粒度测试状态压缩成轨迹级结果。"""

        if self.source_path is None:
            return "NO_EVAL"
        if self.total == 0:
            return "NO_TESTS"
        if self.status_counts.get("FAILED", 0) or self.status_counts.get("ERROR", 0):
            return "FAIL"
        if self.status_counts.get("PASSED", 0):
            return "PASS"
        return "UNKNOWN"


@dataclass(frozen=True)
class EffectfulAction:
    """轨迹中的一步动作。

    名称保留为 EffectfulAction 是为了贴合论文术语；实际对象会保留
    read/search/test 等非 effectful 步骤，并通过 normalized.is_effectful 区分。
    """

    action_id: str
    action_type: str
    pre_state: CodebaseSnapshot
    post_state: CodebaseSnapshot
    context: ActionContext
    normalized: NormalizedAction
    raw_step: dict[str, Any]

    @property
    def is_effectful(self) -> bool:
        return self.normalized.is_effectful


@dataclass(frozen=True)
class Trajectory:
    """一个模型在一个 SWE-Bench Pro instance 上的完整轨迹。"""

    task_id: str
    sample_id: str
    model: str
    model_group: str
    actions: tuple[EffectfulAction, ...]
    eval_summary: EvalSummary
    source_traj: Path
    source_eval: Path | None = None

    @property
    def eval_outcome(self) -> str:
        return self.eval_summary.outcome
