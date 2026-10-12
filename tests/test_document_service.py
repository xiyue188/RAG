"""backend/services/document_service.py 单元测试

重点是 `group_chunks_by_file` —— 它是从 `routes.py` 的 HTTP 端点里搬出来的
纯函数。搬出来的直接收益就是这份测试：原先它嵌在 `async def` 的 `try` 块里，
只能靠起服务发请求才能验证。

Service 的方法用假 VectorDB 注入验证，不起真的 ChromaDB —— 这里要测的是
"分组与 offload 的逻辑对不对"，不是 ChromaDB 本身。
"""

from __future__ import annotations

import asyncio

import pytest

from backend.services.document_service import (
    DocumentService,
    group_chunks_by_file,
)


class FakeCollection:
    """够用的假 collection。记录调用参数供断言。"""

    name = "test_collection"

    def __init__(self, payload: dict | None = None):
        self.payload = payload or {"ids": [], "documents": [], "metadatas": []}
        self.get_calls: list[dict] = []
        self.deleted_ids: list[str] | None = None

    def get(self, ids=None, where=None, include=None):
        self.get_calls.append({"where": where, "include": include})
        if where is None:
            return self.payload
        # 按 file 过滤
        file = where.get("file")
        keep = [
            i for i, m in enumerate(self.payload["metadatas"])
            if (m or {}).get("file") == file
        ]
        return {
            "ids": [self.payload["ids"][i] for i in keep],
            "documents": [self.payload["documents"][i] for i in keep],
            "metadatas": [self.payload["metadatas"][i] for i in keep],
        }

    def delete(self, ids=None):
        self.deleted_ids = list(ids or [])

    def count(self):
        return len(self.payload["ids"])


class FakeVectorDB:
    def __init__(self, collection: FakeCollection):
        self._collection = collection

    def get_collection(self):
        return self._collection


def _payload():
    return {
        "ids": ["a_0", "a_1", "b_0"],
        "documents": ["甲一", "甲二", "乙一"],
        "metadatas": [
            {"file": "a.md", "category": "policy"},
            {"file": "a.md", "category": "policy"},
            {"file": "b.md", "category": "manual"},
        ],
    }


# ============================================================
# group_chunks_by_file（纯函数）
# ============================================================

class TestGrouping:
    def test_groups_by_file_and_counts(self):
        docs = group_chunks_by_file(
            **{k: v for k, v in _payload().items()}, include_chunks=False
        )
        by_file = {d["file"]: d for d in docs}
        assert by_file["a.md"]["chunk_count"] == 2
        assert by_file["b.md"]["chunk_count"] == 1

    def test_without_chunks_omits_content(self):
        """`include_chunks=False` 时不返回正文 —— 列表页不需要，
        拉全库正文的代价随语料线性增长。"""
        docs = group_chunks_by_file(**_payload(), include_chunks=False)
        assert all(d["chunks"] is None for d in docs)

    def test_with_chunks_includes_content_and_index(self):
        docs = group_chunks_by_file(**_payload(), include_chunks=True)
        a = next(d for d in docs if d["file"] == "a.md")
        assert [c["content"] for c in a["chunks"]] == ["甲一", "甲二"]
        assert [c["index"] for c in a["chunks"]] == [0, 1], "序号是文件内位置，从 0 起"

    def test_index_is_per_file_not_global(self):
        """b.md 的第一块序号是 0，不是 2 —— 序号是文件内的位置。"""
        docs = group_chunks_by_file(**_payload(), include_chunks=True)
        b = next(d for d in docs if d["file"] == "b.md")
        assert b["chunks"][0]["index"] == 0

    def test_empty_input(self):
        assert group_chunks_by_file([], [], [], include_chunks=True) == []

    def test_missing_documents_list_tolerated(self):
        """`include=["metadatas"]` 时 ChromaDB 不返回 documents，
        此时正文补空串而不是崩在 zip 上。"""
        p = _payload()
        docs = group_chunks_by_file(p["ids"], [], p["metadatas"], include_chunks=True)
        assert all(c["content"] == "" for d in docs for c in d["chunks"])

    def test_missing_metadata_fields_default_to_unknown(self):
        """元数据缺字段不该让整个列表接口 500。"""
        docs = group_chunks_by_file(["x"], ["内容"], [{}], include_chunks=False)
        assert docs[0]["file"] == "unknown"
        assert docs[0]["category"] == "unknown"

    def test_none_metadata_tolerated(self):
        docs = group_chunks_by_file(["x"], ["内容"], [None], include_chunks=False)
        assert docs[0]["file"] == "unknown"


# ============================================================
# DocumentService
# ============================================================

class TestDocumentService:
    @pytest.mark.asyncio
    async def test_list_documents_requests_documents_only_when_needed(self):
        """`include_chunks=False` 时不向 ChromaDB 要正文。

        这条守的是那个"拉全库正文"的性能问题：列表页默认不该付这个代价。
        """
        collection = FakeCollection(_payload())
        service = DocumentService(FakeVectorDB(collection))

        await service.list_documents(include_chunks=False)
        assert collection.get_calls[-1]["include"] == ["metadatas"]

        await service.list_documents(include_chunks=True)
        assert collection.get_calls[-1]["include"] == ["metadatas", "documents"]

    @pytest.mark.asyncio
    async def test_get_chunks_filters_by_file(self):
        collection = FakeCollection(_payload())
        service = DocumentService(FakeVectorDB(collection))
        chunks = await service.get_chunks("a.md")
        assert [c["content"] for c in chunks] == ["甲一", "甲二"]
        assert collection.get_calls[-1]["where"] == {"file": "a.md"}

    @pytest.mark.asyncio
    async def test_delete_returns_count_and_removes_ids(self):
        collection = FakeCollection(_payload())
        service = DocumentService(FakeVectorDB(collection))
        deleted = await service.delete_document("a.md")
        assert deleted == 2
        assert collection.deleted_ids == ["a_0", "a_1"]

    @pytest.mark.asyncio
    async def test_delete_missing_returns_zero_not_raise(self):
        """"不存在"返回 0，不抛 HTTPException —— 状态码是协议层的事，
        Service 抛 HTTP 异常会让 CLI 调用方也得去 catch 它。"""
        collection = FakeCollection(_payload())
        service = DocumentService(FakeVectorDB(collection))
        assert await service.delete_document("不存在.md") == 0
        assert collection.deleted_ids is None

    @pytest.mark.asyncio
    async def test_invalidate_bumps_kb_version(self, monkeypatch):
        """文档变更的唯一入口必须自增 kb_version。

        四条变更路径（三种上传 + 删除）都经过这里；此前只有异步上传自增。
        """
        from unittest.mock import AsyncMock

        from backend.cache import redis_client
        import backend.services.chat_service as chat_service

        bump = AsyncMock(return_value=1)
        monkeypatch.setattr(redis_client, "bump_kb_version", bump)
        # 不让测试去构造真实 ChatService（会加载 embedding 模型）
        monkeypatch.setattr(chat_service, "get_chat_service", lambda: None)

        service = DocumentService(FakeVectorDB(FakeCollection(_payload())))
        await service.invalidate_retrieval_caches()
        bump.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stats(self):
        collection = FakeCollection(_payload())
        service = DocumentService(FakeVectorDB(collection))
        assert await service.stats() == {
            "total_chunks": 3,
            "vectordb_name": "test_collection",
        }

    @pytest.mark.asyncio
    async def test_calls_do_not_block_event_loop(self):
        """ChromaDB 调用必须 offload —— 阻塞期间事件循环要能跑别的协程。

        用一个会 sleep 的假 collection：如果调用没走线程池，
        `ticker` 一次都跑不动。
        """
        import time

        class SlowCollection(FakeCollection):
            def get(self, ids=None, where=None, include=None):
                time.sleep(0.2)
                return super().get(ids=ids, where=where, include=include)

        service = DocumentService(FakeVectorDB(SlowCollection(_payload())))

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(ticker())
        await service.list_documents()
        task.cancel()

        assert ticks > 3, f"事件循环被阻塞了，只跑了 {ticks} 次"
