"""跨模块共享的协议接口。

接口层只描述调用约定，避免加载器、静态分析和评分器互相耦合。
"""

from __future__ import annotations

from typing import Protocol

from core.action import ActionContext
from core.risk_vector import RiskVector
from core.snapshot import CodebaseSnapshot


class RiskScorer(Protocol):
    """风险评分器协议，后续六维评分实现需要满足该接口。"""

    def score(
        self,
        pre: CodebaseSnapshot,
        post: CodebaseSnapshot,
        ctx: ActionContext,
    ) -> RiskVector:
        ...
