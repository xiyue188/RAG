# 交接文档 — 2026-09-27

> 覆写此文件时用 `docs/handoffs/LATEST.md`，旧内容不保留。
> 事实来源：代码与 `git log`；本文件仅导航。

## M2 完成 ✓

**311 passed，`vue-tsc` exit 0。** 全部 8 个任务完成，M2 收尾。

| 任务 | 提交 | 产出 |
|---|---|---|
| T0a–T0e | 9bf186f | 阈值单一真相源、删可答性副本、I/O 边缘化、AST 架构测试、文档更正 |
| T5 | 786dec3 | 会话落库 + run 轨迹落库，6 条测试 |
| T4 | 374bb75 | 限流→Redis、无界内存修复、XFF 校验，11 条测试 |
| T6 | 156d87f, 9dd84b2 | 后台摄入服务 + 异步上传端点，10 条测试 |

## T6 架构说明

**两个新端点：**
- `POST /api/v1/documents/upload/async` — 校验文件、写临时盘、建 `ingest_tasks` 行，立即返回 `{tasks: [{task_id, status: "pending"}]}`（202）
- `GET /api/v1/documents/tasks/{task_id}` — 从 PG 读权威终态，从 Redis 读实时进度（Redis 不可用时 `progress: {}`）

**[backend/services/ingest_service.py](../../backend/services/ingest_service.py) — 后台 worker：**
- 进程级 `asyncio.Queue`，lifespan 启动时 `start_worker()`
- worker 取 `(task_id, temp_path, filename, category)` 元组，调 `StreamingIngestionAdapter` 跑摄入
- 进度写 `ingest:{task_id}` Redis hash（TTL 1h），不可用时静默
- 成功：`mark_done`；失败：`mark_error`；完成后 `invalidate_retrieval_caches + bump_kb_version`
- `reclaim_pending`：进程重启时把孤儿 pending 任务标为 error（临时文件已丢，需重新上传）

**设计决策：**
- 不引入 Celery/RQ — 摄入在同进程内，无分布式需求，asyncio.Queue 够用
- 并发上限 1 个 worker — embedding 是 torch forward pass，多 worker 争线程池槽位无法提速
- 孤儿任务标 error 而非入队 — 临时文件随重启消失，重跑必然失败，告知用户重新上传更诚实

## 未验证

C8 的端到端（上传真实 5MB 文件、/health 不中断）未实测，只有单元测试。
端对端验证需要起服务 + ChromaDB + 嵌入模型，属于 M5 集成测试范围。

## 下一格：M3

**下一阶段**：M3 Agentic 编排（状态机、路由、四个工具、dispatcher、citation_verify）

M3 开始前建议先做：
1. `answerability.py` — 第一步（top1-top2 分差规则，无需训练数据）
2. M3 计划书（`docs/plans/M3-agent.md`）

详见 [ROADMAP.md](../../docs/ROADMAP.md) M3 节与 [STATUS.md](../../STATUS.md)。
