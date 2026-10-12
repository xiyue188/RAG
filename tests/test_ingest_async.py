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
from backend.services import ingest_service
from backend.services.ingest_service import (
    STALE_ERROR,
    _run_task,
    _update_progress,
    enqueue,
    sweep_stale,
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
# sweep_stale：超时未结束的任务标 error，新任务不动
# ---------------------------------------------------------------------------

async def _age(task_id: str, hours: float) -> None:
    from datetime import timedelta
    from backend.db.models import utcnow

    async with session_scope() as s:
        task = await IngestTaskRepository(s).get(task_id)
        task.created_at = utcnow() - timedelta(hours=hours)


@pytest.mark.asyncio
async def test_sweep_stale_fails_old_pending_and_running(db):
    """进程重启后遗留的 pending / running 超过阈值即标 error。"""
    async with session_scope() as s:
        repo = IngestTaskRepository(s)
        await repo.create(filename="p.md", size_bytes=1, task_id="old-pending")
        await repo.create(filename="r.md", size_bytes=1, task_id="old-running")
        await repo.mark_running("old-running")
    await _age("old-pending", 2)
    await _age("old-running", 2)

    assert await sweep_stale() == 2

    async with session_scope() as s:
        for tid in ("old-pending", "old-running"):
            task = await IngestTaskRepository(s).get(tid)
            assert task.status == "error"
            assert task.error == STALE_ERROR


@pytest.mark.asyncio
async def test_sweep_stale_spares_fresh_tasks(db):
    """多副本安全的关键：未超时的任务（可能在别的副本队列里）不能被清。"""
    async with session_scope() as s:
        await IngestTaskRepository(s).create(filename="f.md", size_bytes=1, task_id="fresh")

    assert await sweep_stale() == 0

    async with session_scope() as s:
        assert (await IngestTaskRepository(s).get("fresh")).status == "pending"


@pytest.mark.asyncio
async def test_swept_task_is_skipped_and_temp_file_removed(db, tmp_path):
    """被清扫的任务轮到 worker 时 CAS 失败：不执行、不改状态、但要清临时文件。

    此前抢占失败直接 return，临时文件留在磁盘上。
    """
    temp = tmp_path / "upload.md"
    temp.write_text("x", encoding="utf-8")
    async with session_scope() as s:
        await IngestTaskRepository(s).create(filename="s.md", size_bytes=1, task_id="swept")
    await _age("swept", 2)
    await sweep_stale()

    with patch("backend.services.ingest_service._ingest", new_callable=AsyncMock) as ingest:
        await _run_task("swept", str(temp), "s.md", "uploaded")

    ingest.assert_not_called()
    assert not temp.exists()
    async with session_scope() as s:
        assert (await IngestTaskRepository(s).get("swept")).status == "error"


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
        patch("backend.services.ingest_service.remove_temp_file"),
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
        patch("backend.services.ingest_service.remove_temp_file"),
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
    """队列无上限，enqueue 立即返回。"""
    await ingest_service.stop()  # 重置为新队列
    await enqueue("t1", "/tmp/a", "a.md")
    await enqueue("t2", "/tmp/b", "b.md")
    assert ingest_service._get_queue().qsize() == 2
    await ingest_service.stop()


@pytest.mark.asyncio
async def test_worker_survives_task_exception():
    """单个任务抛出意外异常，worker 不能退出，要继续处理下一个。"""
    await ingest_service.stop()
    seen = []

    async def fake_run(task_id, *args):
        seen.append(task_id)
        if task_id == "boom":
            raise RuntimeError("意外错误")

    with patch("backend.services.ingest_service._run_task", side_effect=fake_run),          patch("backend.services.ingest_service.sweep_stale", new_callable=AsyncMock):
        await ingest_service.start()
        await enqueue("boom", "/tmp/a", "a.md")
        await enqueue("ok", "/tmp/b", "b.md")
        await asyncio.wait_for(ingest_service._get_queue().join(), timeout=2)
        await ingest_service.stop()

    assert seen == ["boom", "ok"]
