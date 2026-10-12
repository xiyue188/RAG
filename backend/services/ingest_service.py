"""后台摄入服务

上传端点立即返回 task_id，实际摄入在这里的后台 worker 里跑。

## 组成

- **队列**：进程内 `asyncio.Queue`，元素是 `(task_id, temp_path, filename, category)`
- **worker**：一个常驻协程，顺序消费队列
- **清扫器**：一个常驻协程，定期把超时未结束的任务标 error

## 为什么不用 Celery / RQ

摄入发生在接收上传的同一进程里，临时文件也在本机磁盘上，没有跨机调度
的需求。为此引入 broker + 独立 worker 进程，换来的是一套额外的部署与
监控面。代价是本方案**只适合单副本**，见下方"已知限制"。

## 为什么只有一个 worker

embedding 是 torch 前向计算，跑在默认线程池里。多个 worker 并发只会
争同一批线程与 CPU，总吞吐不升反降，还会把检索请求要用的线程池占满。
多文件上传时依次排队。

## 状态的两个存放处

- **PG `ingest_tasks`**：状态机 pending → running → done / error，权威终态。
  Redis 丢了，"这个任务完成没有"仍然答得上来。
- **Redis hash `ingest:{task_id}`**：阶段与百分比，每批 embedding 更新一次。
  高频写不该打数据库；它丢了只是进度条不动，不影响正确性。

## 已知限制（单副本设计）

临时文件路径只存在于内存队列里、没有落库，所以：

1. 进程重启后，排队中与执行中的任务都无法继续 —— 只能标 error 让用户重传。
2. 多副本部署时，上传落到哪个副本就只能由哪个副本处理。

要支持多副本，需要把文件放到共享存储（对象存储）并把路径落库，再把队列
换成数据库队列（`SELECT ... FOR UPDATE SKIP LOCKED`）。不在 M2 范围。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

import config
from backend.adapters import StreamingIngestionAdapter
from backend.cache import redis_client
from backend.db.session import session_scope
from backend.repositories import IngestTaskRepository
from backend.services.document_service import get_document_service, get_ingestion_async
from backend.services.temp_files import remove_temp_file

logger = logging.getLogger(__name__)

# (task_id, temp_path, filename, category)
QueueItem = Tuple[str, str, str, str]

# 惰性创建：在模块 import 时创建会绑定到当时（可能根本不存在）的事件循环，
# 测试里每个用例一个事件循环时会报 "bound to a different event loop"
_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None
_sweeper_task: Optional[asyncio.Task] = None

_PROGRESS_TTL = 3600  # 进度只在任务进行中有意义，结束 1 小时后自然过期

STALE_ERROR = "任务超时未完成（进程可能已重启），临时文件已丢失，请重新上传"


def _get_queue() -> asyncio.Queue:
    global _queue
    if _queue is None:
        _queue = asyncio.Queue()
    return _queue


async def _update_progress(
    task_id: str, *, stage: str, progress: float = 0.0, message: str = ""
) -> None:
    """把进度写进 Redis hash。Redis 不可用时静默（进度不影响正确性）。"""
    await redis_client.hset_mapping(
        f"ingest:{task_id}",
        {"stage": stage, "progress": str(round(progress, 1)), "message": message},
        ttl=_PROGRESS_TTL,
    )


async def _run_task(task_id: str, temp_path: str, filename: str, category: str) -> None:
    """执行一个摄入任务：抢占 → 摄入 → 写终态 → 刷新检索缓存。

    临时文件在 finally 里清理，**包括抢占失败的分支** —— 此前抢占失败
    直接 return，临时文件就留在磁盘上了。
    """
    try:
        # pending → running。WHERE status='pending' 让它成为一次 CAS：
        # 任务若已被清扫器标 error，或被别处抢走，这里返回 False
        async with session_scope() as s:
            claimed = await IngestTaskRepository(s).mark_running(task_id)
        if not claimed:
            logger.warning("ingest task %s 不是 pending（已被抢占或已超时），跳过", task_id)
            return

        await _update_progress(task_id, stage="running", message="开始处理")
        try:
            chunk_count = await _ingest(task_id, temp_path, filename, category)
        except Exception as exc:  # noqa: BLE001 - 任何失败都要落成 error 终态
            logger.error("ingest task %s 失败: %s", task_id, exc, exc_info=True)
            async with session_scope() as s:
                await IngestTaskRepository(s).mark_error(task_id, error=str(exc))
            await _update_progress(task_id, stage="error", message=str(exc))
            return

        async with session_scope() as s:
            await IngestTaskRepository(s).mark_done(task_id, chunk_count=chunk_count)
        await _update_progress(
            task_id, stage="done", progress=100.0, message=f"完成，共 {chunk_count} 个切片"
        )

        # 文档变了：BM25 索引按全库语料构建需要重建；kb_version 自增让
        # 将来带版本号的检索缓存自然失效
        await get_document_service().invalidate_retrieval_caches()
        await redis_client.bump_kb_version()
        logger.info("ingest task %s 完成，%d 个切片", task_id, chunk_count)
    finally:
        await asyncio.to_thread(remove_temp_file, temp_path)


async def _ingest(task_id: str, temp_path: str, filename: str, category: str) -> int:
    """跑一遍摄入流水线，边跑边写进度，返回切片数。失败抛异常。

    复用 SSE 流式上传的同一个适配器：两条上传路径写进向量库的 metadata
    结构必须一致，不能各有一套摄入实现。
    """
    adapter = StreamingIngestionAdapter(await get_ingestion_async())

    chunk_count = 0
    async for event in adapter.ingest_file_stream(temp_path, filename, category):
        etype = event.get("type", "")
        data = event.get("data", {})

        if etype == "chunking_done":
            await _update_progress(
                task_id, stage="chunking", progress=30.0,
                message=f"分块完成，{data.get('chunk_count', 0)} 个切片",
            )
        elif etype == "embedding_progress":
            await _update_progress(
                task_id, stage="embedding", progress=data.get("percentage", 0.0),
                message=f"向量化 {data.get('current', 0)}/{data.get('total', 0)}",
            )
        elif etype == "storing_done":
            chunk_count = data.get("chunk_count", 0)
            await _update_progress(task_id, stage="storing", progress=95.0, message="写入向量库")
        elif etype == "upload_complete":
            chunk_count = data.get("chunk_count", chunk_count)
        elif etype == "error":
            # 适配器把异常转成了 error 事件，这里还原成异常交给调用方统一处理
            raise RuntimeError(data.get("message", "摄入失败"))

    return chunk_count


async def _worker_loop() -> None:
    """常驻 worker：顺序消费队列。单个任务出错不能让 worker 退出。"""
    queue = _get_queue()
    logger.info("ingest worker 已启动")
    while True:
        item: QueueItem = await queue.get()
        try:
            await _run_task(*item)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - _run_task 已处理业务异常，这里兜底
            logger.exception("ingest worker 意外错误，继续处理下一个任务")
        finally:
            queue.task_done()


async def sweep_stale() -> int:
    """把超时仍未结束的任务标 error，返回处理数量。

    阈值 `INGEST_STALE_SECONDS` 必须大于"排队 + 单个摄入"的最长耗时，
    否则会把还在排队的活任务清掉。被清掉的任务轮到 worker 时 CAS 失败、
    跳过并清理临时文件，不会被执行两次。
    """
    async with session_scope() as s:
        count = await IngestTaskRepository(s).fail_stale(
            config.INGEST_STALE_SECONDS, error=STALE_ERROR
        )
    if count:
        logger.warning("清扫了 %d 个超时摄入任务", count)
    return count


async def _sweeper_loop() -> None:
    """常驻清扫器：启动时立刻扫一次，之后按间隔扫。

    只在启动时扫不够：崩溃前刚创建的任务启动时还没到阈值，
    要等下一次重启才会被清理。定期扫让它在阈值过后即被清理。
    """
    while True:
        try:
            await sweep_stale()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 数据库暂时不可用不该让清扫器退出
            logger.warning("清扫超时摄入任务失败，下个周期重试", exc_info=True)
        await asyncio.sleep(config.INGEST_SWEEP_INTERVAL_SECONDS)


async def enqueue(
    task_id: str, temp_path: str, filename: str, category: str = "uploaded"
) -> None:
    """放进队列。调用方负责先在 PG 建好 pending 任务行。

    队列无上限，put 不会阻塞；排队过久的任务由清扫器兜底。
    """
    queue = _get_queue()
    await queue.put((task_id, temp_path, filename, category))
    logger.info("ingest task %s 入队（队列长度 %d）", task_id, queue.qsize())


async def start() -> None:
    """启动 worker 与清扫器。在 lifespan 启动阶段调用一次。"""
    global _worker_task, _sweeper_task
    _worker_task = asyncio.create_task(_worker_loop(), name="ingest-worker")
    _sweeper_task = asyncio.create_task(_sweeper_loop(), name="ingest-sweeper")


async def stop() -> None:
    """停止 worker 与清扫器。在 lifespan 关闭阶段调用。

    直接 cancel 而不等当前摄入跑完：摄入可能持续数分钟，阻塞关闭会让
    容器编排超时后强杀。被打断的任务停在 running，由下次启动后的清扫器
    标 error。
    """
    global _queue, _worker_task, _sweeper_task
    for task in (_worker_task, _sweeper_task):
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    _worker_task = _sweeper_task = None
    _queue = None
