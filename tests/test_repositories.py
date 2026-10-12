"""backend/repositories 单元测试

除了各方法的行为，这份测试重点守住 Repository 层的三条硬约束 ——
它们是最容易在后续开发中被无意破坏的：

1. **不 commit**。任何一个方法自己提交，跨表原子性就没了。
   `TestNoCommit` 用"抛异常后检查库里什么都没有"来证明。
2. **集合关系必须预加载**。async 下懒加载抛 MissingGreenlet，
   `TestEagerLoading` 在 session 关闭后访问关系来证明确实加载了。
3. **并发下的唯一性靠数据库**，不靠"先查再写"。

不测 PostgreSQL 特有行为（JSONB 按字段查询），那需要真实 PG 实例。
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, MissingGreenlet
from sqlalchemy.orm.exc import DetachedInstanceError

from backend.db.models import Evidence, IngestTask, Message, Run, Session, utcnow
from backend.repositories import (
    EvidenceRepository,
    IngestTaskRepository,
    RunRepository,
    SessionRepository,
)


# ============================================================
# RunRepository
# ============================================================

class TestRunCreate:
    @pytest.mark.asyncio
    async def test_create_returns_object_with_id(self, db):
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="宠物政策是什么")
            assert run.id is not None
            assert run.status == "running"
            assert len(run.trace_id) == 36

    @pytest.mark.asyncio
    async def test_accepts_caller_supplied_trace_id(self, db):
        """允许传 trace_id：调用方要能先写日志再落库，两边 id 对得上。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q", trace_id="fixed-trace")
            assert run.trace_id == "fixed-trace"

    @pytest.mark.asyncio
    async def test_duplicate_trace_id_rejected_by_db(self, db):
        """唯一性由数据库保证，Repository 不做"先查再写"。"""
        async with db.session_scope() as s:
            await RunRepository(s).create(query="q1", trace_id="dup")

        with pytest.raises(IntegrityError):
            async with db.session_scope() as s:
                await RunRepository(s).create(query="q2", trace_id="dup")


class TestRunFinish:
    @pytest.mark.asyncio
    async def test_finish_sets_status_and_metrics(self, db):
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            ok = await repo.finish(
                run.id, status="ok", total_ms=1234, first_token_ms=210,
                prompt_tokens=800, completion_tokens=120,
            )
            assert ok is True
            run_id = run.id

        async with db.session_scope() as s:
            got = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one()
            assert got.status == "ok"
            assert got.total_ms == 1234
            assert got.first_token_ms == 210
            assert got.finished_at is not None

    @pytest.mark.asyncio
    async def test_zero_ms_is_preserved_not_dropped(self, db):
        """total_ms=0 与 first_token_ms=0 是合法值（缓存命中），不能被当成"没传"。

        回归测试：用 `if total_ms:` 或字典推导过滤假值会把 0 丢掉，
        这与 CLAUDE.md 禁止用 `or` 链回退数值是同一类问题。
        """
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            await repo.finish(run.id, total_ms=0, first_token_ms=0)
            run_id = run.id

        async with db.session_scope() as s:
            got = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one()
            assert got.total_ms == 0
            assert got.first_token_ms == 0

    @pytest.mark.asyncio
    async def test_finish_missing_run_returns_false(self, db):
        """不存在的 run 返回 False 而非抛异常 —— 调用方据此决定要不要告警。"""
        async with db.session_scope() as s:
            assert await RunRepository(s).finish(999999) is False

    @pytest.mark.asyncio
    async def test_finish_with_error_records_message(self, db):
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            await repo.finish(run.id, status="error", error="LLM 超时")
            run_id = run.id

        async with db.session_scope() as s:
            got = (await s.execute(select(Run).where(Run.id == run_id))).scalar_one()
            assert got.status == "error"
            assert got.error == "LLM 超时"


class TestRunSteps:
    @pytest.mark.asyncio
    async def test_add_step_stores_snapshot(self, db):
        snapshot = {"node": "retrieve", "hits": [{"chunk_id": "c1", "score": 0.87}]}
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            step = await repo.add_step(
                run.id, seq=0, node="retrieve", ms=42, state_snapshot=snapshot
            )
            assert step.id is not None
            assert step.state_snapshot == snapshot

    @pytest.mark.asyncio
    async def test_step_without_snapshot_gets_empty_dict(self, db):
        """{} 是"这步没快照"，NULL 是"没记录过"，不能混。"""
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            step = await repo.add_step(run.id, seq=0, node="route")
            assert step.state_snapshot == {}

    @pytest.mark.asyncio
    async def test_duplicate_seq_raises(self, db):
        """同一 run 内 seq 重复必须炸，不能静默吞掉。

        轨迹完整性是本项目的卖点，缺一步比报错更糟。
        """
        with pytest.raises(IntegrityError):
            async with db.session_scope() as s:
                repo = RunRepository(s)
                run = await repo.create(query="q")
                await repo.add_step(run.id, seq=0, node="a")
                await repo.add_step(run.id, seq=0, node="b")

    @pytest.mark.asyncio
    async def test_tool_call_recorded_with_idempotency_key(self, db):
        """idempotency_key 目前只是存下来，没有判重逻辑（见 ADR-003）。"""
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            call = await repo.add_tool_call(
                run.id, seq=0, tool="calculate",
                args={"expr": "1+1"}, result_summary="2", ms=3,
                idempotency_key="k1",
            )
            assert call.args == {"expr": "1+1"}
            assert call.ok is True
            assert call.idempotency_key == "k1"

    @pytest.mark.asyncio
    async def test_failed_tool_call_recorded(self, db):
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            call = await repo.add_tool_call(
                run.id, seq=0, tool="web_search", ok=False,
                result_summary="超时",
            )
            assert call.ok is False


class TestEagerLoading:
    """预加载不是性能优化，是"能不能跑"的问题。"""

    @pytest.mark.asyncio
    async def test_get_by_trace_loads_all_collections(self, db):
        """在 session 关闭后仍能访问三个集合关系。

        这是验证预加载的正确方式：如果只在 session 里访问，
        懒加载也能"成功"（它会偷偷发一条 SQL），测不出区别。
        """
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q", trace_id="t-eager")
            await repo.add_step(run.id, seq=0, node="route")
            await repo.add_step(run.id, seq=1, node="retrieve")
            await repo.add_tool_call(run.id, seq=0, tool="search_knowledge_base")
            await EvidenceRepository(s).bulk_insert(
                run.id, [{"chunk_id": "c1", "relevance": 0.9}]
            )

        async with db.session_scope() as s:
            fetched = await RunRepository(s).get_by_trace("t-eager")

        # session 已关闭。未预加载的话下面每一行都会抛 MissingGreenlet
        assert fetched is not None
        assert [step.node for step in fetched.steps] == ["route", "retrieve"]
        assert len(fetched.tool_calls) == 1
        assert len(fetched.evidence) == 1

    @pytest.mark.asyncio
    async def test_list_recent_does_not_preload(self, db):
        """列表页故意不预加载 —— 只显示标量字段，拉关联是纯浪费。

        这条测试把"没预加载"钉成有意的设计而非遗漏：
        哪天要在列表页显示步骤数，得先改 Repository 再改这条测试。
        """
        async with db.session_scope() as s:
            repo = RunRepository(s)
            run = await repo.create(query="q")
            await repo.add_step(run.id, seq=0, node="route")

        async with db.session_scope() as s:
            runs = await RunRepository(s).list_recent()
            assert len(runs) == 1
            fetched = runs[0]

        # 两种异常都表示"没预加载"：session 还在时是 MissingGreenlet
        # （异步下发不出懒加载的 SQL），已关闭时是 DetachedInstanceError
        # （SQLAlchemy 在更早一步就发现对象脱管）
        with pytest.raises((DetachedInstanceError, MissingGreenlet)):
            _ = fetched.steps[0]

    @pytest.mark.asyncio
    async def test_get_by_trace_missing_returns_none(self, db):
        async with db.session_scope() as s:
            assert await RunRepository(s).get_by_trace("不存在") is None


class TestRunQueries:
    @pytest.mark.asyncio
    async def test_list_recent_is_newest_first(self, db):
        async with db.session_scope() as s:
            repo = RunRepository(s)
            for i in range(3):
                run = await repo.create(query=f"q{i}", trace_id=f"t{i}")
                # 手工拉开时间，否则同一毫秒内创建的顺序不确定
                run.created_at = utcnow() + timedelta(seconds=i)

        async with db.session_scope() as s:
            runs = await RunRepository(s).list_recent(limit=2)
            assert [r.trace_id for r in runs] == ["t2", "t1"]

    @pytest.mark.asyncio
    async def test_list_unfinished_finds_stale_running(self, db):
        """进程崩溃留下的 running 记录能被捞出来。"""
        async with db.session_scope() as s:
            repo = RunRepository(s)
            stale = await repo.create(query="崩溃的", trace_id="stale")
            stale.created_at = utcnow() - timedelta(hours=2)
            fresh = await repo.create(query="刚开始的", trace_id="fresh")
            done = await repo.create(query="已完成", trace_id="done")
            done.created_at = utcnow() - timedelta(hours=2)
            await s.flush()
            await repo.finish(done.id)

        async with db.session_scope() as s:
            found = await RunRepository(s).list_unfinished(
                older_than=utcnow() - timedelta(hours=1)
            )
            assert [r.trace_id for r in found] == ["stale"]

    @pytest.mark.asyncio
    async def test_count_by_status(self, db):
        async with db.session_scope() as s:
            repo = RunRepository(s)
            ok1 = await repo.create(query="a", trace_id="a")
            ok2 = await repo.create(query="b", trace_id="b")
            await repo.create(query="c", trace_id="c")
            await repo.finish(ok1.id, status="ok")
            await repo.finish(ok2.id, status="ok")

        async with db.session_scope() as s:
            counts = await RunRepository(s).count_by_status()
            assert counts == {"ok": 2, "running": 1}


# ============================================================
# EvidenceRepository
# ============================================================

class TestEvidence:
    @pytest.mark.asyncio
    async def test_bulk_insert_assigns_rank_by_order(self, db):
        """rank 由列表顺序定义，不从 results 里取 —— 避免两者不一致。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            rows = await EvidenceRepository(s).bulk_insert(run.id, [
                {"chunk_id": "c1", "file": "a.md", "relevance": 0.91},
                {"chunk_id": "c2", "file": "b.md", "relevance": 0.72},
            ])
            assert [r.rank for r in rows] == [0, 1]

    @pytest.mark.asyncio
    async def test_relevance_zero_preserved(self, db):
        """0.0 是合法分数（极低相关），不能被当成缺失存成 NULL。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            rows = await EvidenceRepository(s).bulk_insert(
                run.id, [{"chunk_id": "c1", "relevance": 0.0}]
            )
            assert rows[0].relevance is not None
            assert float(rows[0].relevance) == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_missing_relevance_becomes_null(self, db):
        """没有分数与分数为 0 是两回事。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            rows = await EvidenceRepository(s).bulk_insert(
                run.id, [{"chunk_id": "c1"}]
            )
            assert rows[0].relevance is None

    @pytest.mark.asyncio
    async def test_retrieved_by_list_joined(self, db):
        """检索层给 list，存成逗号分隔字符串（ARRAY 在 SQLite 上不可用）。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            rows = await EvidenceRepository(s).bulk_insert(
                run.id, [{"chunk_id": "c1", "retrieved_by": ["vector", "bm25"]}]
            )
            assert rows[0].retrieved_by == "vector,bm25"

    @pytest.mark.asyncio
    async def test_empty_insert_is_noop(self, db):
        """检索无结果时不该炸，也不该白跑一次 flush。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            assert await EvidenceRepository(s).bulk_insert(run.id, []) == []

    @pytest.mark.asyncio
    async def test_mark_used_only_marks_cited(self, db):
        """区分"检索到"与"被答案用上"。"""
        async with db.session_scope() as s:
            repo = EvidenceRepository(s)
            run = await RunRepository(s).create(query="q")
            await repo.bulk_insert(run.id, [
                {"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"},
            ])
            updated = await repo.mark_used(run.id, ["c1", "c3"])
            assert updated == 2
            run_id = run.id

        async with db.session_scope() as s:
            rows = await EvidenceRepository(s).list_for_run(run_id)
            assert {r.chunk_id: r.used_in_answer for r in rows} == {
                "c1": True, "c2": False, "c3": True,
            }

    @pytest.mark.asyncio
    async def test_mark_used_empty_is_noop(self, db):
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            assert await EvidenceRepository(s).mark_used(run.id, []) == 0

    @pytest.mark.asyncio
    async def test_mark_used_scoped_to_run(self, db):
        """同一 chunk 可能出现在多次 run 里，标记不能串台。"""
        async with db.session_scope() as s:
            repo = EvidenceRepository(s)
            r1 = await RunRepository(s).create(query="q1", trace_id="t1")
            r2 = await RunRepository(s).create(query="q2", trace_id="t2")
            await repo.bulk_insert(r1.id, [{"chunk_id": "shared"}])
            await repo.bulk_insert(r2.id, [{"chunk_id": "shared"}])
            await repo.mark_used(r1.id, ["shared"])
            r2_id = r2.id

        async with db.session_scope() as s:
            rows = await EvidenceRepository(s).list_for_run(r2_id)
            assert rows[0].used_in_answer is False

    @pytest.mark.asyncio
    async def test_usage_stats(self, db):
        async with db.session_scope() as s:
            repo = EvidenceRepository(s)
            run = await RunRepository(s).create(query="q")
            await repo.bulk_insert(run.id, [{"chunk_id": f"c{i}"} for i in range(5)])
            await repo.mark_used(run.id, ["c0", "c1"])
            stats = await repo.usage_stats(run.id)
            assert stats == {"retrieved": 5, "used": 2}

    @pytest.mark.asyncio
    async def test_usage_stats_on_empty_run(self, db):
        """零行时 SUM 返回 NULL，必须被 coalesce 成 0 而不是漏给调用方。"""
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            stats = await EvidenceRepository(s).usage_stats(run.id)
            assert stats == {"retrieved": 0, "used": 0}


# ============================================================
# SessionRepository
# ============================================================

class TestSessionGetOrCreate:
    @pytest.mark.asyncio
    async def test_creates_when_missing(self, db):
        async with db.session_scope() as s:
            sess = await SessionRepository(s).get_or_create("s1")
            assert sess.id == "s1"

    @pytest.mark.asyncio
    async def test_returns_existing(self, db):
        async with db.session_scope() as s:
            await SessionRepository(s).get_or_create("s1", meta={"ua": "chrome"})

        async with db.session_scope() as s:
            sess = await SessionRepository(s).get_or_create("s1")
            assert sess.meta == {"ua": "chrome"}

    @pytest.mark.asyncio
    async def test_concurrent_create_does_not_duplicate(self, db):
        """并发建同一会话不产生重复行，也不让任何一方失败。

        靠的是 SAVEPOINT + 主键约束，不是应用层加锁 ——
        与"不引入分布式锁"是同一条推理。
        """
        async def create_one():
            async with db.session_scope() as s:
                await SessionRepository(s).get_or_create("shared")

        await asyncio.gather(*(create_one() for _ in range(5)))

        async with db.session_scope() as s:
            count = (await s.execute(
                select(func.count()).select_from(Session)
            )).scalar_one()
            assert count == 1

    @pytest.mark.asyncio
    async def test_savepoint_rollback_keeps_outer_transaction_usable(self, db):
        """内层冲突回滚后，同一事务里的其它写入不受影响。

        这是为什么用 begin_nested 而不是 session.rollback() ——
        后者会把调用方的其它写入一起丢掉，而 Repository 无权决定这件事。
        """
        async with db.session_scope() as s:
            await SessionRepository(s).get_or_create("existing")

        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="不该被回滚", trace_id="keep")
            await SessionRepository(s).get_or_create("existing")   # 走冲突分支
            assert run.id is not None

        async with db.session_scope() as s:
            got = await RunRepository(s).get_by_trace("keep")
            assert got is not None
            assert got.query == "不该被回滚"


class TestSessionMessages:
    @pytest.mark.asyncio
    async def test_append_and_read_history_in_order(self, db):
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            await repo.get_or_create("s1")
            await repo.append_message("s1", role="user", content="问题一")
            await repo.append_message("s1", role="assistant", content="回答一")

        async with db.session_scope() as s:
            history = await SessionRepository(s).history("s1")
            assert [(m.role, m.content) for m in history] == [
                ("user", "问题一"), ("assistant", "回答一"),
            ]

    @pytest.mark.asyncio
    async def test_history_limit_takes_latest_but_returns_ascending(self, db):
        """截断从旧的那头砍，返回仍是时间正序 —— 上下文要按时间读。"""
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            await repo.get_or_create("s1")
            for i in range(5):
                await repo.append_message("s1", role="user", content=f"m{i}")

        async with db.session_scope() as s:
            history = await SessionRepository(s).history("s1", limit=2)
            assert [m.content for m in history] == ["m3", "m4"]

    @pytest.mark.asyncio
    async def test_user_message_has_no_run_id(self, db):
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            await repo.get_or_create("s1")
            msg = await repo.append_message("s1", role="user", content="问题")
            assert msg.run_id is None

    @pytest.mark.asyncio
    async def test_assistant_message_links_to_run(self, db):
        async with db.session_scope() as s:
            run = await RunRepository(s).create(query="q")
            repo = SessionRepository(s)
            await repo.get_or_create("s1")
            msg = await repo.append_message(
                "s1", role="assistant", content="答案", run_id=run.id
            )
            assert msg.run_id == run.id

    @pytest.mark.asyncio
    async def test_append_refreshes_last_active(self, db):
        """活跃时间与消息在同一事务里更新，不会出现"有消息但会话看着过期"。"""
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            sess = await repo.get_or_create("s1")
            sess.last_active_at = utcnow() - timedelta(hours=5)
            await s.flush()

        async with db.session_scope() as s:
            await SessionRepository(s).append_message("s1", role="user", content="嗨")

        async with db.session_scope() as s:
            sess = await s.get(Session, "s1")
            assert (utcnow() - sess.last_active_at).total_seconds() < 60


class TestSessionCleanup:
    @pytest.mark.asyncio
    async def test_clear_removes_session_and_messages(self, db):
        """消息由外键 CASCADE 连带删除，不在应用层逐条删。"""
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            await repo.get_or_create("s1")
            await repo.append_message("s1", role="user", content="x")

        async with db.session_scope() as s:
            assert await SessionRepository(s).clear("s1") is True

        async with db.session_scope() as s:
            assert (await s.execute(
                select(func.count()).select_from(Message)
            )).scalar_one() == 0

    @pytest.mark.asyncio
    async def test_clear_missing_returns_false(self, db):
        async with db.session_scope() as s:
            assert await SessionRepository(s).clear("不存在") is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_only_removes_stale(self, db):
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            old = await repo.get_or_create("old")
            old.last_active_at = utcnow() - timedelta(hours=3)
            await repo.get_or_create("fresh")
            await s.flush()

        async with db.session_scope() as s:
            removed = await SessionRepository(s).cleanup_expired(timeout_seconds=3600)
            assert removed == 1

        async with db.session_scope() as s:
            remaining = (await s.execute(select(Session.id))).scalars().all()
            assert remaining == ["fresh"]

    @pytest.mark.asyncio
    async def test_cleanup_cascades_to_messages(self, db):
        """批量 DELETE 时 ORM 级联不生效，靠的是数据库外键 ondelete=CASCADE。

        这两个是不同机制，这条测试证明我们依赖的是后者。
        """
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            old = await repo.get_or_create("old")
            await repo.append_message("old", role="user", content="x")
            old.last_active_at = utcnow() - timedelta(hours=3)
            await s.flush()

        async with db.session_scope() as s:
            await SessionRepository(s).cleanup_expired(timeout_seconds=3600)

        async with db.session_scope() as s:
            assert (await s.execute(
                select(func.count()).select_from(Message)
            )).scalar_one() == 0

    @pytest.mark.asyncio
    async def test_count_active(self, db):
        async with db.session_scope() as s:
            repo = SessionRepository(s)
            old = await repo.get_or_create("old")
            old.last_active_at = utcnow() - timedelta(hours=3)
            await repo.get_or_create("fresh")
            await s.flush()

        async with db.session_scope() as s:
            assert await SessionRepository(s).count_active() == 1

    @pytest.mark.asyncio
    async def test_touch_updates_and_reports_hit(self, db):
        async with db.session_scope() as s:
            await SessionRepository(s).get_or_create("s1")

        async with db.session_scope() as s:
            repo = SessionRepository(s)
            assert await repo.touch("s1") is True
            assert await repo.touch("不存在") is False


# ============================================================
# IngestTaskRepository
# ============================================================

class TestIngestTask:
    @pytest.mark.asyncio
    async def test_create_is_pending(self, db):
        async with db.session_scope() as s:
            task = await IngestTaskRepository(s).create(
                filename="手册.pdf", size_bytes=5_242_880
            )
            assert task.status == "pending"
            assert task.chunk_count is None
            assert task.finished_at is None

    @pytest.mark.asyncio
    async def test_lifecycle_pending_running_done(self, db):
        async with db.session_scope() as s:
            repo = IngestTaskRepository(s)
            task = await repo.create(filename="a.md", size_bytes=100)
            assert await repo.mark_running(task.id) is True
            assert await repo.mark_done(task.id, chunk_count=12) is True
            task_id = task.id

        async with db.session_scope() as s:
            got = await IngestTaskRepository(s).get(task_id)
            assert got.status == "done"
            assert got.chunk_count == 12
            assert got.finished_at is not None

    @pytest.mark.asyncio
    async def test_mark_running_is_compare_and_set(self, db):
        """第二次 mark_running 返回 False —— WHERE 里带 status='pending'
        让状态转换成为原子的 CAS。两个 worker 抢同一任务时只有一个成功。
        """
        async with db.session_scope() as s:
            repo = IngestTaskRepository(s)
            task = await repo.create(filename="a.md", size_bytes=1)
            assert await repo.mark_running(task.id) is True
            assert await repo.mark_running(task.id) is False

    @pytest.mark.asyncio
    async def test_concurrent_claim_only_one_wins(self, db):
        """并发抢任务，只有一个 worker 拿到。不需要分布式锁。"""
        async with db.session_scope() as s:
            task = await IngestTaskRepository(s).create(
                filename="a.md", size_bytes=1, task_id="task-race"
            )

        async def claim() -> bool:
            async with db.session_scope() as s:
                return await IngestTaskRepository(s).mark_running("task-race")

        results = await asyncio.gather(*(claim() for _ in range(5)))
        assert sum(results) == 1

    @pytest.mark.asyncio
    async def test_mark_error_stores_full_message(self, db):
        """error 不截断：失败原因往往在异常链末尾。"""
        long_error = "Traceback...\n" * 200
        async with db.session_scope() as s:
            repo = IngestTaskRepository(s)
            task = await repo.create(filename="a.md", size_bytes=1)
            await repo.mark_error(task.id, error=long_error)
            task_id = task.id

        async with db.session_scope() as s:
            got = await IngestTaskRepository(s).get(task_id)
            assert got.status == "error"
            assert got.error == long_error

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, db):
        async with db.session_scope() as s:
            assert await IngestTaskRepository(s).get("不存在") is None

    @pytest.mark.asyncio
    async def test_list_by_status_oldest_first(self, db):
        """重启后捞回 pending 任务，先来的先处理。"""
        async with db.session_scope() as s:
            repo = IngestTaskRepository(s)
            for i in range(3):
                task = await repo.create(
                    filename=f"{i}.md", size_bytes=1, task_id=f"t{i}"
                )
                task.created_at = utcnow() + timedelta(seconds=i)
            await s.flush()
            await repo.mark_running("t1")

        async with db.session_scope() as s:
            pending = await IngestTaskRepository(s).list_by_status("pending")
            assert [t.id for t in pending] == ["t0", "t2"]

    @pytest.mark.asyncio
    async def test_fail_stale_marks_old_unfinished_as_error(self, db):
        """超时的 pending / running 标 error；新任务与已结束任务不动。

        不打回 pending：临时文件路径只在内存队列里，进程死后无法重跑。
        """
        async with db.session_scope() as s:
            repo = IngestTaskRepository(s)
            old_pending = await repo.create(filename="a.md", size_bytes=1, task_id="old-p")
            old_running = await repo.create(filename="b.md", size_bytes=1, task_id="old-r")
            old_done = await repo.create(filename="c.md", size_bytes=1, task_id="old-d")
            await repo.create(filename="d.md", size_bytes=1, task_id="fresh")
            for t in (old_pending, old_running, old_done):
                t.created_at = utcnow() - timedelta(hours=2)
            await s.flush()
            await repo.mark_running("old-r")
            await repo.mark_running("old-d")
            await repo.mark_done("old-d", chunk_count=3)

        async with db.session_scope() as s:
            assert await IngestTaskRepository(s).fail_stale(1800, error="超时") == 2

        async with db.session_scope() as s:
            repo = IngestTaskRepository(s)
            for tid in ("old-p", "old-r"):
                task = await repo.get(tid)
                assert task.status == "error"
                assert task.error == "超时"
                assert task.finished_at is not None
            assert (await repo.get("old-d")).status == "done"
            assert (await repo.get("fresh")).status == "pending"


# ============================================================
# 硬约束：Repository 不 commit
# ============================================================

class TestNoCommit:
    """任何一个方法自己 commit，跨表原子性就没了。

    这是最容易被无意破坏的约束 —— 加一行 `await self.session.commit()`
    能让某个用例"跑通"，但同时悄悄毁掉所有跨表事务。
    """

    @pytest.mark.asyncio
    async def test_exception_rolls_back_everything(self, db):
        """写完 run、step、evidence 后抛异常，库里必须什么都没有。"""
        class Boom(Exception):
            pass

        with pytest.raises(Boom):
            async with db.session_scope() as s:
                run = await RunRepository(s).create(query="q", trace_id="rollback")
                await RunRepository(s).add_step(run.id, seq=0, node="route")
                await EvidenceRepository(s).bulk_insert(
                    run.id, [{"chunk_id": "c1"}]
                )
                raise Boom()

        async with db.session_scope() as s:
            for model in (Run, Evidence):
                count = (await s.execute(
                    select(func.count()).select_from(model)
                )).scalar_one()
                assert count == 0, f"{model.__tablename__} 有残留 —— 有人在 commit"

    @pytest.mark.asyncio
    async def test_cross_repository_atomicity(self, db):
        """跨 Repository 的写入共享一个事务，一个失败全部回滚。

        这正是 Repository 不该自己管事务的原因：原子性需求是业务级的。
        """
        class Boom(Exception):
            pass

        with pytest.raises(Boom):
            async with db.session_scope() as s:
                await SessionRepository(s).get_or_create("s1")
                run = await RunRepository(s).create(query="q")
                await SessionRepository(s).append_message(
                    "s1", role="assistant", content="答案", run_id=run.id
                )
                await IngestTaskRepository(s).create(filename="a.md", size_bytes=1)
                raise Boom()

        async with db.session_scope() as s:
            for model in (Session, Run, Message, IngestTask):
                count = (await s.execute(
                    select(func.count()).select_from(model)
                )).scalar_one()
                assert count == 0, f"{model.__tablename__} 有残留"

    # 源码级的 commit 检查已迁到 tests/test_architecture.py::
    # TestRepositoriesDoNotCommit，并从字符串查找改为 AST ——
    # 原先的 `".commit()" in source` 会把注释和 docstring 里提到
    # "不 commit" 的文字也算成违规。上面两条行为测试保留：
    # 它们验的是"跨表原子性成立"，与"源码里没有 commit 调用"是
    # 两个不同层面的证据。
