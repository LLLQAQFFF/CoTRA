"""六维风险向量的基础类型。

Phase 0 只定义数据形状，不实现任何维度评分逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RiskVector:
    """后续 RiskScorer 输出的统一结构。"""

    task_advancement: float
    debt_density: float
    fragility_delta: float
    regression_surface: float
    observability_loss: float
    wrong_abstraction: dict[str, Any] = field(default_factory=dict)
    action_myopia_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
