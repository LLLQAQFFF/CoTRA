"""代码库快照的轻量表示。

Phase 0 暂时不保存完整文件内容，只保留后续风险评分需要对齐的
最小元数据。真正的 pre/post 文件快照会在后续数据采集阶段补齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodebaseSnapshot:
    """某一时刻的代码库状态占位对象。"""

    root_path: str
    commit_hash: str | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)
