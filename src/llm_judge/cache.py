"""Client-side response cache keyed by sha256(model + system + user).

Gateway-side Anthropic cache_control was tested empirically and does not yield
cache hits on poloai (cache_read_input_tokens stays 0 across repeat requests).
We instead persist verbatim responses on disk so that re-running prelabeling on
the same (model, system, user) tuple does not re-burn tokens.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


class ResponseCache:
    def __init__(self, cache_dir: Path):
        """输入：缓存目录路径。输出：无。作用：初始化磁盘缓存存储。"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _key(model: str, system: str, user: str) -> str:
        """输入：模型名、系统提示和用户提示。输出：sha256 缓存键。作用：唯一标识一次请求。"""
        h = hashlib.sha256()
        for part in (model, system, user):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        """输入：缓存键。输出：分片后的 JSON 文件路径。作用：定位磁盘上的缓存响应。"""
        # 2-level prefix sharding to avoid one big directory
        return self.cache_dir / key[:2] / key[2:4] / f"{key}.json"

    def get(self, model: str, system: str, user: str) -> str | None:
        """输入：请求标识字段。输出：缓存响应或 None。作用：读取已保存的 LLM 回复。"""
        path = self._path(self._key(model, system, user))
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, model: str, system: str, user: str, response: str) -> None:
        """输入：请求标识和响应文本。输出：无。作用：原子化持久化 LLM 回复。"""
        key = self._key(model, system, user)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model, "response": response}
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(payload, ensure_ascii=False))
            tmp.replace(path)
