"""摄入任务的读写

**进度写 Redis，终态写这里。** 进度是高频更新（每批 embedding 一次），
打数据库不划算；但 Redis 可丢弃，所以"这个任务到底完成没有"必须落 PG。

T6 的异步摄入依赖本模块：upload 建任务后立即返回 task_id，
后台 worker 取任务、跑摄入、写终态。
"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select, update

from ..db.models import IngestTask, utcnow
from .base import BaseRepository

# 任务状态机：pending → running → done / error
# 只允许单向前进，不允许 done 回到 running
TASK_STATUSES = ("pending", "running", "done", "error")


class IngestTaskRepository(BaseRepository):
    """文档摄入任务。"""

    async def create(
        self,
        *,
        filename: str,
        size_bytes: int,
        category: str = "uploaded",
        task_id: Optional[str] = None,
    ) -> IngestTask:
        """建一个 pending 任务，返回它（含 id）。

        task_id 允许传入，理由同 Run.trace_id：调用方要能先把 id 写进
        日志与响应，再落库。
        """
        task = IngestTask(
            id=task_id or str(uuid.uuid4()),
            filename=filename,
            size_bytes=size_bytes,
            category=category,
            status="pending",
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def mark_running(self, task_id: str) -> bool:
        """pending → running。返回是否真的转换了。

        **WHERE 里带 status='pending' 是关键**：这让状态转换成为一次
        原子的 compare-and-set。两个 worker 同时取到同一任务时，
        只有一个的 rowcount 是 1，另一个是 0 —— 靠数据库判谁抢到，
        不需要锁。这与移除分布式锁是同一条推理。
        """
        result = await self.session.execute(
            update(IngestTask)
            .where(IngestTask.id == task_id, IngestTask.status == "pending")
            .values(status="running")
        )
        return result.rowcount > 0

    async def mark_done(self, task_id: str, *, chunk_count: int) -> bool:
        """running → done，记录切片数。"""
        result = await self.session.execute(
            update(IngestTask)
            .where(IngestTask.id == task_id)
            .values(status="done", chunk_count=chunk_count, finished_at=utcnow())
        )
        return result.rowcount > 0

    async def mark_error(self, task_id: str, *, error: str) -> bool:
        """标记失败。

        error 存全文不截断：摄入失败的原因往往在异常链末尾，
        截断会把最有用的部分切掉。列类型是 Text 不是 varchar。
        """
        result = await self.session.execute(
            update(IngestTask)
            .where(IngestTask.id == task_id)
            .values(status="error", error=error, finished_at=utcnow())
        )
        return result.rowcount > 0

    async def get(self, task_id: str) -> Optional[IngestTask]:
        return await self.session.get(IngestTask, task_id)

    async def list_by_status(
        self,
        status: str,
        *,
        limit: int = 50,
    ) -> Sequence[IngestTask]:
        """按状态列任务，旧的在前。

        走 ix_ingest_tasks_status 复合索引 (status, created_at)。
        主要用途是进程重启后捞回 pending 任务 —— 队列在内存里，
        重启就空了，但 PG 里的 pending 行还在。
        """
        result = await self.session.execute(
            select(IngestTask)
            .where(IngestTask.status == status)
            .order_by(IngestTask.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def fail_stale(self, older_than_seconds: int, *, error: str) -> int:
        """把超时仍未结束（pending / running）的任务标为 error，返回处理数量。

        **为什么是标 error 而不是打回 pending 重跑**：此前这里是
        `reclaim_stale_running`（running → pending），理由是"摄入幂等，重跑
        安全"。幂等本身成立，但重跑需要原始文件，而临时文件路径只存在于
        进程内的队列里、没有落库 —— 进程一死路径就丢了，打回 pending 的
        任务永远不会有人处理。唯一诚实的终态是 error，让用户重新上传。

        **为什么按年龄而不是"本进程启动前创建的"**：多副本时，副本 B 启动
        若清掉所有早于自己启动的任务，会误杀副本 A 队列里正常排队的任务。
        按年龄判断，只要阈值大于"排队 + 摄入"的最长耗时，就不会误伤活任务。

        一条 UPDATE 完成，不先查再逐行改：避免 N+1，也避免查询与更新之间
        任务状态已变的窗口。
        """
        from datetime import timedelta

        cutoff = utcnow() - timedelta(seconds=older_than_seconds)
        result = await self.session.execute(
            update(IngestTask)
            .where(
                IngestTask.status.in_(("pending", "running")),
                IngestTask.created_at < cutoff,
            )
            .values(status="error", error=error, finished_at=utcnow())
        )
        return result.rowcount
