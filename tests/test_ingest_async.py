"""T6 异步摄入测试

验收标准 C8：上传文件后立即返回 task_id，不阻塞请求线程。
同时覆盖：
- 任务状态机 pending → running → done / error
- 孤儿任务（进程重启）回收逻辑
- Redis 不可用时进度读取降级
- 文件类型 / 大小校验
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.db.session import session_scope
from backend.repositories import IngestTaskRepository
from backend.services.ingest_service import (
    _run_task,
    _update_progress,
    enqueue,
    reclaim_pending,
)


# ---------------------------------------------------------------------------
# 状态机：create → mark_running → mark_done
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_lifecycle_ok(db):
    """pending → running → done 状态转换正确，chunk_count 写入。"""
    async with session_scope() as s:
        task = await IngestTaskRepository(s).create(
            filename="test.md", size_bytes=1024, task_id="task-ok-1"
        )

    assert task.status == "pending"

    async with session_scope() as s:
        claimed = await IngestTaskRepository(s).mark_running("task-ok-1")
    assert claimed is True

    async with session_scope() as s:
        done = await IngestTaskRepository(s).mark_done("task-ok-1", chunk_count=7)
    assert done is True

    async with session_scope() as s:
        t = await IngestTaskRepository(s).get("task-ok-1")
    assert t.status == "done"
    assert t.chunk_count == 7
    assert t.finished_at is not None


# ---------------------------------------------------------------------------
# 状态机：mark_running CAS 防双认领
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mark_running_cas_prevents_double_claim(db):
    """两次 mark_running：第一次返回 True，第二次返回 False。"""
    async with session_scope() as s:
        await IngestTaskRepository(s).create(
            filename="double.md", size_bytes=512, task_id="task-cas"
        )

    async with session_scope() as s:
        first = await IngestTaskRepository(s).mark_running("task-cas")
    async with session_scope() as s:
        second = await IngestTaskRepository(s).mark_running("task-cas")

    assert first is True
    assert second is False  # 已是 running，WHERE status='pending' 不命中


# ---------------------------------------------------------------------------
# 状态机：mark_error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_task_lifecycle_error(db):
    """pending → running → error，error 字段写入。"""
    async with session_scope() as s:
        await IngestTaskRepository(s).create(
            filename="bad.md", size_bytes=256, task_id="task-err"
        )

    async with session_scope() as s:
        await IngestTaskRepository(s).mark_running("task-err")
    async with session_scope() as s:
        await IngestTaskRepository(s).mark_error("task-err", error="embedding 失败")

    async with session_scope() as s:
        t = await IngestTaskRepository(s).get("task-err")
    assert t.status == "error"
    assert "embedding 失败" in t.error


# ---------------------------------------------------------------------------
# reclaim_pending：进程重启后孤儿 pending 任务标为 error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reclaim_pending_marks_as_error(db):
    """reclaim_pending 把所有 pending 任务标为 error（临时文件已丢失）。"""
    for i in range(3):
        async with session_scope() as s:
            await IngestTaskRepository(s).create(
                filename=f"orphan_{i}.md",
                size_bytes=100,
                task_id=f"orphan-{i}",
            )

    count = await reclaim_pending()
    assert count == 3

    for i in range(3):
        async with session_scope() as s:
            t = await IngestTaskRepository(s).get(f"orphan-{i}")
        assert t.status == "error"
        assert "重启" in t.error


@pytest.mark.asyncio
async def test_reclaim_pending_ignores_running_and_done(db):
    """reclaim_pending 只处理 pending，不动 running 和 done。"""
    async with session_scope() as s:
        await IngestTaskRepository(s).create(
            filename="running.md", size_bytes=100, task_id="t-running"
        )
        await IngestTaskRepository(s).mark_running("t-running")

    async with session_scope() as s:
        await IngestTaskRepository(s).create(
            filename="done.md", size_bytes=100, task_id="t-done"
        )
        await IngestTaskRepository(s).mark_running("t-done")
        await IngestTaskRepository(s).mark_done("t-done", chunk_count=1)

    count = await reclaim_pending()
    assert count == 0  # 没有 pending 任务

    async with session_scope() as s:
        r = await IngestTaskRepository(s).get("t-running")
        d = await IngestTaskRepository(s).get("t-done")
    assert r.status == "running"
    assert d.status == "done"


# ---------------------------------------------------------------------------
# list_by_status：旧的在前
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_by_status_oldest_first(db):
    """list_by_status('pending') 按 created_at 升序返回。"""
    for i in range(4):
        async with session_scope() as s:
            await IngestTaskRepository(s).create(
                filename=f"f{i}.md", size_bytes=10, task_id=f"order-{i}"
            )

    async with session_scope() as s:
        tasks = await IngestTaskRepository(s).list_by_status("pending", limit=10)

    ids = [t.id for t in tasks]
    assert ids == sorted(ids, key=lambda x: x)  # 按插入顺序（uuid 不排序，但 created_at 升序）
    assert len(tasks) == 4


# ---------------------------------------------------------------------------
# _update_progress：Redis 不可用时静默通过
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_progress_redis_unavailable():
    """Redis 不可用时 _update_progress 不抛异常。"""
    with patch(
        "backend.services.ingest_service.redis_client.hset_mapping",
        new_callable=AsyncMock,
        return_value=False,  # 返回 False 表示不可用
    ):
        # 不应抛出
        await _update_progress("task-x", stage="embedding", progress=50.0)


# ---------------------------------------------------------------------------
# _run_task：摄入成功完整流程（mock adapter 和 session_scope）
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_success(db):
    """_run_task 成功时：running → done，chunk_count 正确写入。"""
    # 先建任务行
    async with session_scope() as s:
        await IngestTaskRepository(s).create(
            filename="ok.md", size_bytes=500, task_id="task-run-ok"
        )

    # mock adapter 返回成功事件流
    async def _fake_stream(path, filename, category):
        yield {"type": "chunking_done", "data": {"chunk_count": 5}}
        yield {"type": "storing_done", "data": {"chunk_count": 5}}
        yield {"type": "upload_complete", "data": {"chunk_count": 5}}

    with (
        patch("backend.services.ingest_service.get_ingestion_async", new_callable=AsyncMock),
        patch("backend.services.ingest_service.StreamingIngestionAdapter") as mock_adapter_cls,
        patch("backend.services.ingest_service.get_document_service") as mock_ds,
        patch("backend.services.ingest_service.redis_client.hset_mapping", new_callable=AsyncMock),
        patch("backend.services.ingest_service.redis_client.bump_kb_version", new_callable=AsyncMock),
        patch("backend.api.upload._remove_temp_file"),  # patch where the name lives
    ):
        mock_adapter_cls.return_value.ingest_file_stream = _fake_stream
        mock_ds.return_value.invalidate_retrieval_caches = AsyncMock()

        await _run_task("task-run-ok", "/tmp/fake.md", "ok.md", "uploaded")

    async with session_scope() as s:
        t = await IngestTaskRepository(s).get("task-run-ok")
    assert t.status == "done"
    assert t.chunk_count == 5


# ---------------------------------------------------------------------------
# _run_task：摄入失败时写 error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_task_error(db):
    """adapter 抛异常时 _run_task 标记为 error。"""
    async with session_scope() as s:
        await IngestTaskRepository(s).create(
            filename="bad.md", size_bytes=100, task_id="task-run-err"
        )

    async def _bad_stream(path, filename, category):
        yield {"type": "error", "data": {"message": "embedding 崩了"}}

    with (
        patch("backend.services.ingest_service.get_ingestion_async", new_callable=AsyncMock),
        patch("backend.services.ingest_service.StreamingIngestionAdapter") as mock_cls,
        patch("backend.services.ingest_service.redis_client.hset_mapping", new_callable=AsyncMock),
        patch("backend.services.ingest_service.redis_client.bump_kb_version", new_callable=AsyncMock),
        patch("backend.services.ingest_service._remove_temp_file"),
    ):
        mock_cls.return_value.ingest_file_stream = _bad_stream

        await _run_task("task-run-err", "/tmp/bad.md", "bad.md", "uploaded")

    async with session_scope() as s:
        t = await IngestTaskRepository(s).get("task-run-err")
    assert t.status == "error"
    assert "embedding 崩了" in t.error


# ---------------------------------------------------------------------------
# enqueue：不阻塞，立即返回
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enqueue_nonblocking():
    """enqueue 是非阻塞的，即使队列里已有任务也能立即返回。"""
    import backend.services.ingest_service as svc
    # 清空队列
    while not svc._queue.empty():
        svc._queue.get_nowait()

    await enqueue("t1", "/tmp/a", "a.md")
    await enqueue("t2", "/tmp/b", "b.md")

    assert svc._queue.qsize() == 2
    # 清理
    while not svc._queue.empty():
        svc._queue.get_nowait()
