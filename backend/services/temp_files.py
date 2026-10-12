"""上传暂存：临时文件的写入与清理

原先这两个函数定义在 `backend/api/upload.py`，`ingest_service` 为了在
worker 里清理临时文件反过来 import 了 api 层 —— service 依赖协议层，
依赖方向倒置。它们是 I/O 工具，与 HTTP 无关，放在 service 层两边共用。

两个函数都是阻塞调用，异步调用方必须 `asyncio.to_thread`。
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def write_temp_file(content: bytes, suffix: str) -> str:
    """把上传内容落到临时文件，返回路径。"""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as fp:
        fp.write(content)
        return fp.name


def remove_temp_file(path: str) -> None:
    """删除临时文件。不存在时静默（清理可能被多处兜底调用）。"""
    Path(path).unlink(missing_ok=True)
