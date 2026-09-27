"""后台摄入服务

**设计原则**：上传端点立即返回 task_id，实际摄入在后台 worker 里跑。

用 `asyncio.Queue` + 常驻 worker，不引入 Celery/RQ：
- 摄入任务本就在同一进程里，没有分布式调度的需求
- 常驻 worker 比每次请求新建 `asyncio.create_task` 更可控：
  可以限制并发数、可以在关闭时优雅等待当前任务完成
- 进程重启后 Queue 会空掉，但 `list_by_status('pending')` 可以
  把未完成的任务重新入队 —— 这是启动时的 reclaim 步骤

**并发上限**：`MAX_CONCURRENT_WORKERS = 1`。
摄入是 CPU+IO 密集型（embedding 是 torch forward pass），
多个 worker 会争线程池槽位且无法提速，一个 worker 顺序处理最稳定。
多文件上传时各自排队，而不是并行跑。

**进度写 Redis hash**：key = `ingest:{task_id}`，TTL = 1h。
前端轮询 `GET /api/v1/documents/tasks/{task_id}` 即可拿到当前阶段。
Redis 不可用时只是无法追踪进度，任务照常跑，终态仍写 PG。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from backend.cache import redis_client
from backend.db.session import session_scope
from backend.repositories import IngestTaskRepository
from backend.services.document_service import get_ingestion_async, get_document_service
from backend.adapters import StreamingIngestionAdapter
from backend.api.upload import _remove_temp_file

logger = logging.getLogger(__name__)

# 单例 Queue：进程内所有上传端点共用同一条队列
_queue: asyncio.Queue[str] = asyncio.Queue()

# worker 任务句柄，便于 lifespan 关闭时 cancel
_worker_task: Optional[asyncio.Task] = None

# Redis hash 里各字段的语义：
#   stage      当前阶段（parsing / chunking / embedding / storing / done / error）
#   progress   当前进度百分比（字符串，如 "42.0"）
#   message    最新状态描述
_PROGRESS_TTL = 3600  # 1 小时，任务结束后进度条不必永久保留


async def _update_progress(
    task_id: str, *, stage: str, progress: float = 0.0, message: str = ""
) -> None:
    """把进度写进 Redis hash，不可用时静默忽略。"""
    await redis_client.hset_mapping(
        f"ingest:{task_id}",
        {"stage": stage, "progress": str(round(progress, 1)), "message": message},
        ttl=_PROGRESS_TTL,
    )


async def _run_task(task_id: str, temp_path: str, filename: str, category: str) -> None:
    """执行一个摄入任务的全部步骤，更新 PG 状态和 Redis 进度。"""
    # 1. pending → running（CAS，防止双认领）
    async with session_scope() as s:
        ok = await IngestTaskRepository(s).mark_running(task_id)
    if not ok:
        logger.warning("ingest task %s already claimed, skipping", task_id)
        return

    await _update_progress(task_id, stage="running", progress=0.0, message="开始处理")

    try:
        ingestion = await get_ingestion_async()

        adapter = StreamingIngestionAdapter(ingestion)

        chunk_count = 0
        async for event in adapter.ingest_file_stream(temp_path, filename, category):
            etype = event.get("type", "")
            data = event.get("data", {})

            if etype == "embedding_progress":
                pct = data.get("percentage", 0.0)
                await _update_progress(
                    task_id, stage="embedding", progress=pct,
                    message=f"向量化 {data.get('current', 0)}/{data.get('total', 0)}"
                )
            elif etype == "chunking_done":
                await _update_progress(
                    task_id, stage="chunking", progress=30.0,
                    message=f"分块完成，{data.get('chunk_count', 0)} 个切片"
                )
            elif etype == "storing_done":
                chunk_count = data.get("chunk_count", 0)
                await _update_progress(
                    task_id, stage="storing", progress=95.0, message="写入向量库"
                )
            elif etype == "upload_complete":
                chunk_count = data.get("chunk_count", chunk_count)
            elif etype == "error":
                raise RuntimeError(data.get("message", "摄入失败"))

        # 2. running → done
        async with session_scope() as s:
            await IngestTaskRepository(s).mark_done(task_id, chunk_count=chunk_count)

        await _update_progress(
            task_id, stage="done", progress=100.0,
            message=f"完成，共 {chunk_count} 个切片"
        )

        # 3. 刷新检索侧缓存（BM25 索引 + kb_version）
        await get_document_service().invalidate_retrieval_caches()
        await redis_client.bump_kb_version()

        logger.info("ingest task %s done, %d chunks", task_id, chunk_count)

    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        logger.error("ingest task %s failed: %s", task_id, error_msg, exc_info=True)

        async with session_scope() as s:
            await IngestTaskRepository(s).mark_error(task_id, error=error_msg)

        await _update_progress(task_id, stage="error", progress=0.0, message=error_msg)

    finally:
        # 不论成败都清掉临时文件
        import asyncio as _asyncio
        from backend.api.upload import _remove_temp_file
        await _asyncio.to_thread(_remove_temp_file, temp_path)


async def _worker_loop() -> None:
    """常驻 worker。从队列取 (task_id, temp_path, filename, category) 元组并执行。"""
    logger.info("ingest worker started")
    while True:
        try:
            item = await _queue.get()
            # item 是 (task_id, temp_path, filename, category) 元组
            task_id, temp_path, filename, category = item
            await _run_task(task_id, temp_path, filename, category)
        except asyncio.CancelledError:
            logger.info("ingest worker cancelled")
            break
        except Exception:  # noqa: BLE001
            logger.exception("ingest worker unexpected error, continuing")
        finally:
            try:
                _queue.task_done()
            except Exception:
                pass


async def enqueue(
    task_id: str, temp_path: str, filename: str, category: str = "uploaded"
) -> None:
    """把一个摄入任务放进队列。调用方负责提前在 PG 建好任务行。"""
    await _queue.put((task_id, temp_path, filename, category))
    logger.info("ingest task %s enqueued (queue size=%d)", task_id, _queue.qsize())


async def start_worker() -> None:
    """启动后台 worker，在 lifespan 里调用一次。"""
    global _worker_task
    _worker_task = asyncio.create_task(_worker_loop(), name="ingest-worker")
    logger.info("ingest background worker started")


async def stop_worker() -> None:
    """优雅停止：取消 worker，等待当前任务结束。"""
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await asyncio.wait_for(_worker_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    _worker_task = None
    logger.info("ingest background worker stopped")


async def reclaim_pending() -> int:
    """进程重启后，把 PG 里 pending 的任务重新塞进队列。

    队列在内存里，进程重启就空了；但 pending 任务行还在 PG，
    不 reclaim 它们就永远不会被处理。

    只 reclaim pending，不 reclaim running：running 任务可能是上次
    进程被杀时中断的，IngestTaskRepository.reclaim_stale_running 负责
    把超时的 running 打回 pending，两者分开调用。
    """
    async with session_scope() as s:
        tasks = await IngestTaskRepository(s).list_by_status("pending", limit=100)

    # pending 任务没有 temp_path 了（重启时临时文件消失），
    # 直接标 error 并告知原因 —— 用户需要重新上传
    count = 0
    for task in tasks:
        async with session_scope() as s:
            await IngestTaskRepository(s).mark_error(
                task.id, error="进程重启，临时文件已丢失，请重新上传"
            )
        count += 1

    if count:
        logger.warning("reclaimed %d orphaned pending tasks as error", count)
    return count
