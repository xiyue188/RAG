# STATUS

> 导航文件。与代码冲突时以代码和 `git log` 为准。

## 当前位置

**阶段**：M2 基础设施 —— **进行中**（T1、T2、T3、T0、T5、T4 完成）
**计划书**：`docs/plans/M2-infrastructure.md`

## 下一步：单一有序清单

**一次只做一格，做完提交，不并行。** 这张表是唯一的待办来源 ——
其它文档里的待办项都应该在这里有对应行，没有的说明还没排期。

| # | 做什么 | 验收 | 状态 |
|---|---|---|---|
| 1 | **T0a** 阈值单一真相源 | 改 `.env` 阈值后刷新页面，前端配色随之变 | **完成** |
| 2 | **T0b** 删可答性判断副本 | `chat_service.py` 不再出现 `ANSWERABLE_MIN_RELEVANCE` | **完成** |
| 3 | **T0c** I/O 边缘化 | `backend/api/` 不再 import `rag`，阻塞调用全部 offload | **完成** |
| 4 | **T0d** 架构约束测试 | 注入违规必失败并指出行号（已实测） | **完成** |
| 5 | **T0e** 文档更正 | ADR-004 阶段 1、本清单 | **完成** |
| 6 | **T5** 会话落库 + run 轨迹落库 | C3/C4/C5：一次问答在 `runs` 留一行，重启后数据仍在 | **完成** |
| 7 | **T4** 限流迁移 + 缓存防御 | C6/C7/C9 | **完成** |
| — | **← 停下来 review（当前位置）** | 301 passed；`vue-tsc` exit 0 | |
| 8 | **T6** 异步摄入 | C8：上传 5MB PDF 立即返回 task_id | 2h |
| — | **← M2 完成** | | |
| 9 | M3 工具 `Protocol` + dispatcher | 加新工具不需改 dispatcher | M3 |
| 10 | M3 answerability 重做 | `rag/eval/` 能直接跑可答性评测 | M3 |
| 11 | M5 `LLMProvider` Port + fake | CI 跑通端到端测试且无真实 API 调用 | M5 |

T0 是 T3 完成后审计出五处规则漂移而插入的，详见计划书 T0 节
与 [ADR-004](docs/decisions/ADR-004-layering-and-rule-placement.md)。
**五处已全部修完**，且每一处都留下了对应的架构测试规则防复发：

| 漂移 | 形态 | 修法 | 防复发规则 |
|---|---|---|---|
| 前端两组件阈值不一致（且与后端不符） | 副本漂移，**已产生用户可见错误** | `/config/thresholds` 运行时下发 | 规则 5 |
| `chat_service` 手抄可答性判断 | 副本漂移 | 删副本，调 `assess_context` | 规则 4 |
| `routes.py` 内联 60 行分组逻辑 | 位置漂移 | 搬进 `DocumentService` | 规则 2 |
| `api/` 直接 import `rag` 并操作 ChromaDB | 位置漂移 | 经 service，全部 offload | 规则 2 |
| 每请求新建 `VectorDB()` | 资源误用 | 进程级单例 | —— |

## M2 任务状态

| 任务 | 状态 | 实际产出 |
|---|---|---|
| T1 依赖与容器 | **完成** | compose 加 PG16 + Redis7 含健康检查；requirements 补 6 个依赖；`.env.example` 同步 |
| T2 ORM 与迁移 | **完成** | 11 张表 + Alembic 初始化 + 首个迁移；修掉 SQLite 主键与时区两处缺陷 |
| T3 Repository 层 | **完成** | 4 个 Repository + 54 条测试；修掉 SQLite 外键不强制、SAVEPOINT 失效、并发写死锁三处 |
| T4 Redis + 限流迁移 | 一半 | `redis_client.py` 已有并有测试；`rate_limit.py` 仍是内存态。**范围已扩写**（XFF 信任、无界增长、缓存三大问题防御） |
| T5 会话落库 | 未开始 | Repository 已就绪，剩下改 `chat_service.py` |
| T6 异步摄入 | 未开始 | `IngestTaskRepository` 已就绪；摄入路径的阻塞调用已修 |

## Repository 层的三条硬约束

写在 `backend/repositories/base.py`，`tests/test_repositories.py::TestNoCommit`
用行为测试 + 源码扫描双重守着：

1. **不 commit**。session 由调用方传入，事务边界归 service。
   任何方法自己提交，跨表原子性就没了。
2. **需要自增 id 时 flush 而非 commit**。
3. **查集合关系必须显式预加载**。async 下懒加载抛 `MissingGreenlet` /
   `DetachedInstanceError`，这不是性能优化是能不能跑。
   `get_by_trace` 用 `selectinload` 而非 `joinedload` —— 后者对三个
   集合会产生笛卡尔积。`list_recent` **故意不预加载**，列表页只要标量字段。

按**聚合**而非按表划分：`RunRepository` 同时管 runs / run_steps /
tool_calls，因为后两者不能脱离 run 存在。

## 验收标准进度

| # | 标准 | 状态 |
|---|---|---|
| C1 | `docker compose up` 起 4 服务且健康检查通过 | **部分** —— `postgres` 与 `redis` 已实测起来且 healthy；backend/frontend 镜像未构建（要装 torch，约 10-20 分钟，放 M5） |
| C2 | `alembic upgrade head` 建出全部表 | **通过** —— SQLite 与 **PostgreSQL 上都实测过**，`alembic check` 均无漂移，downgrade 也验过 |
| C3-C5 | runs/evidence 落库、重启后数据仍在 | 未开始（依赖 T3） |
| C6-C7 | 限流存 Redis、Redis 停掉仍放行 | 未开始（T4 后半） |
| C8-C9 | 异步上传、`kb_version` 自增 | 未开始（T6） |
| C10 | 测试通过 + 前端 `vue-tsc` 通过 | **通过** —— 228 passed；`vue-tsc --noEmit` exit 0 |

## 测试覆盖

```
tests/test_scoring.py      37 条  分数口径
tests/test_metrics.py      35 条  检索指标
tests/test_threshold.py    17 条  阈值校准
tests/test_db.py           36 条  ORM/约束/级联/事务边界/并发/SQLite PRAGMA
tests/test_redis_client.py 41 条  降级路径（三种不可用形态）
tests/test_migrations.py    8 条  迁移与模型不漂移、升降级往返
tests/test_repositories.py 54 条  四个 Repository + 三条硬约束
                          ----
                          228 passed（约 17s）
```

仍然零覆盖：`api/`、`services/chat_service.py`、`adapters/`、
`rate_limit.py`、`rag/retriever.py`、`rag/llm.py`。集成测试属 M5，
需要 fake LLM provider 才能不烧 token。

## M2 已修掉的缺陷

**五处，全部只在 SQLite 上出现**，而 SQLite 是默认 `DATABASE_URL` 与计划书
写明的失败回退路径 —— 不是边缘情况。共同点是"开发环境静默通过、换库或
上线才暴露"，**五处全部是补测试时才查出的**，此前 DB 层零覆盖，
b67a4c2 写的代码从未被执行过。

T2 期间（`models.py`）：

1. **`BigInteger` 主键不自增**。SQLite 的隐式自增要求列类型名恰好是
   `"INTEGER"`；`BIGINT` 有整数亲和性但不是 rowid 别名，插入报
   `NOT NULL constraint failed: messages.id`。9 张表全中。
   改用 `BigInteger().with_variant(Integer, "sqlite")`，PG 上仍是 BIGINT。
2. **`DateTime(timezone=True)` 不保留时区**。SQLite 无原生时间类型，
   读回来一律 naive，`created_at > utcnow()` 在 PG 上正常、在 SQLite 上抛
   `can't compare offset-naive and offset-aware`。加 `UtcDateTime`
   TypeDecorator 在类型层抹平。

T3 期间（`session.py::_apply_sqlite_pragmas`）：

3. **SQLite 默认不强制外键**。`ondelete="CASCADE"` 写了也白写。
   需每连接 `PRAGMA foreign_keys=ON`。
   **此前的级联测试是因为错误的原因通过的** —— 它用 `session.delete(obj)`，
   走的是 SQLAlchemy 在 Python 里逐个删的 ORM 级联，即使数据库根本不强制
   外键也会通过。现已补 `TestSqlitePragmas::test_database_level_cascade_works`
   用 DELETE 语句验数据库级联。
4. **pysqlite 的隐式事务管理破坏 SAVEPOINT**。`begin_nested()` 的保存点
   无法正确回滚，`SessionRepository.get_or_create` 的并发处理会失效。
   修法是 SQLAlchemy 文档的标准方案：`isolation_level = None` +
   自己发 BEGIN。
5. **并发写死锁**。修完 4 之后并发写撞 `database is locked`。
   默认的 deferred 事务延迟申请写锁，两个事务各持读锁再同时升级即死锁，
   **`busy_timeout` 对此无效**（那是死锁不是繁忙）。需
   `journal_mode=WAL` + `busy_timeout` + **`BEGIN IMMEDIATE`**，三者缺一不可。

## PostgreSQL 实测结果（"PG 特有行为未验证"已消除）

`docker compose up -d postgres redis` 起容器后实测：

- `alembic upgrade head` → `Context impl PostgresqlImpl` + **`transactional DDL`**
  （SQLite 上是 `SQLiteImpl` + `non-transactional`），`alembic check` exit 0
- `runs.id` = `bigint DEFAULT nextval('runs_id_seq')`，SQLite 上是 `INTEGER`
  —— `with_variant` 两边都对
- `created_at` = `timestamp with time zone`；`tool_calls.args` = **`jsonb`**
  （不是 `json`），`JSONField` 的方言分支生效
- Repository 读写往返全部正确：`total_ms=0` 未被吞、`created_at.tzinfo=UTC`、
  预加载在 session 关闭后可用、JSONB 中文键往返、`relevance=0.0` 未变 NULL、
  5 并发 `get_or_create` 只产生 1 行（SAVEPOINT 在 PG 上同样工作）

**宿主机侧端口是 15432 不是 5432**：开发机上已装有本地 PostgreSQL
（`postgres.exe` 作为系统服务占着 `0.0.0.0:5432`），容器绑 5432 会报
"An attempt was made to access a socket in a way forbidden by its access
permissions" 而起不来。换端口而非停服务，还顺带避免了**静默连错库** ——
连接串写 `localhost:5432` 会连到本机那个 PG，表和数据全对不上且不报错。

## 环境坑（本轮新踩）

- **`alembic.ini` 必须纯 ASCII**。configparser 以 `encoding="locale"` 读它，
  Windows 上即 GBK，文件里有中文在 alembic 启动前就抛 `UnicodeDecodeError`，
  报错栈指向 configparser 内部，看不出是编码问题。说明写进 `alembic/README.md`。
- **`alembic.ini` 里不能有 `timezone =` 空值**。空字符串被当作已配置，
  喂给 `ZoneInfo("")` 抛 ValueError。要留空就整行删掉。
- **`env.py` 不能用 `TextIOWrapper(sys.stdout.buffer)` 切 UTF-8**。
  它替换 `sys.stdout` 对象本身，在测试里以 API 方式调 alembic 会把 pytest
  的捕获流顶掉，第二次调用报 "I/O operation on closed file"。
  用 `stream.reconfigure(encoding="utf-8")` 就地改。
- **autogenerate 不会为自定义类型加 import**。它把类型渲染成
  `backend.db.models.JSONField()` 却不加 import，直接跑 NameError。
  这是"autogenerate 产物必须人工审阅"的典型例子。
- **对着已有表的库跑 autogenerate 会生成空迁移**。首次生成必须指向空库
  （`DATABASE_URL=... alembic revision --autogenerate`），
  否则 diff 为空而看不出问题。

## 未解决问题（移交 M3）

- **工具调用幂等尚未实现**。`ToolCall.idempotency_key` 字段存在但无任何写入方，
  且只有普通索引不是唯一约束 —— 靠它做"先查再写"判重在并发下两个请求会同时
  查空同时执行。键的组成已定，见
  [ADR-003](docs/decisions/ADR-003-tool-call-idempotency-key.md)：拆成 run 内
  步骤键（带 UNIQUE，先插占位行再执行）与跨 run 结果缓存键（Redis + TTL，
  只对声明只读的工具启用）。原 ROADMAP 那个全局键公式已废弃。
  注意本项目四个工具全部只读无外部副作用，**不要把它讲成解决了重复扣费**。
- **独立的 answerability 判断**：单一阈值已证明不可行。候选方向 ——
  LLM 生成前先判断"给定这些证据能否回答"、用 top1 与 top2 的分数差
  而非绝对值、引用校验前置。
- **rerank 可用性改造**：质量增益真实但 CPU P50 9.8 秒。
  待验证 GPU / 候选数削到 top-5 / 按路由选择性启用。
- **引用准确率与覆盖率**：依赖 M3 的 citation_verify。
- **BM25 分层验证**：T2Ranking 无 query 类型标注，
  无法验证"BM25 在关键词类查询上更强"。

## M1 结论（不要重复实验）

检索层（300 条 T2Ranking）：

| variant | R@5 | MRR@10 | nDCG@10 | P50 ms |
|---|---|---|---|---|
| vector_minilm | 0.023 | 0.058 | 0.029 | 16.5 |
| vector_bge | **0.707** | 0.907 | 0.854 | 15.1 |
| bm25 | 0.586 | 0.841 | 0.721 | 42.5 |
| rrf | 0.708 | 0.902 | 0.837 | 52.1 |
| rrf_rerank | **0.747** | **0.947** | **0.900** | 9841.7 |

1. **换中文 embedding 使 Recall@5 提升约 30 倍**（0.023 → 0.707）。
   MiniLM 的 MRR@10 仅 0.058，等同随机排序。
2. **RRF 在本数据集上无增益**（0.708 vs 0.707），延迟涨 3.5 倍。
   该数据集语义匹配为主，BM25 单独 0.586 无互补空间。
   不是实现错误，**不要试图"修好"它**。
3. **单一 relevance 阈值无法判定可答性**。有答案与无答案的 top1 分布
   严重重叠（无答案最小值 0.714 > 有答案最小值 0.697）。
   误拒率 ≤10% 时最佳阈值只能识别 19.1% 的无答案查询。

生成层（50 条）：忠实度均值 **0.916**，中位数 1.000，44 条有效 / 6 条判 None。
人工核验 20 条一致率 **85%**，不一致的 3 条全部偏紧、无偏松失效。
判分器自实现（不用 ragas，原因见报告第 7 节）。

## 向量库现状

```
eval_t2ranking__baai_bge_small_zh_v1_5   13536   评测（bge）
eval_t2ranking__all_minilm_l6_v2         13536   评测（MiniLM 基线）
techcorp_docs__baai_bge_small_zh_v1_5       59   生产默认
techcorp_docs__all_minilm_l6_v2             59   生产对比基线
techcorp_docs                               21   换指纹前旧库，可删
```

评测与生产物理隔离（前缀 `eval_`）。

## 活跃决策

- 编排层手写状态机，不引入 LangGraph（节点签名对齐 StateGraph 形状）
- 向量留在 ChromaDB，不引入 pgvector；PG 只存关系数据与轨迹
- 不做分布式锁（M2 实现过一版后移除）。摄入本身幂等，
  锁是性能优化而非正确性需求；真需要唯一性时用数据库唯一约束
- 检索指标自研；生成层判分也自研（ragas 在 GLM 上跑不通）
- 项目定位为通用可追溯 RAG 研究平台，`data/documents/` 仅为 demo

## 待办观察（边界外）

- `.env` 已补 `POSTGRES_DB/USER/PASSWORD`（compose 用 `${VAR:?}` 强制必填，
  不给默认弱密码）。`DATABASE_URL` 默认注释掉走 SQLite，要连 PG 就取消注释。
- `create_all` 与 Alembic 并存的漂移风险。已在 `alembic/README.md` 写明：
  `init_models()` 只用于测试，生产路径只有 `alembic upgrade head`。
  lifespan 里**没有**调 `create_all`。
- `rate_limit.py` 三个缺陷，**其中两个单机就存在**，T4 的范围应当覆盖：
  1. 内存态 → 重启归零、多副本配额翻 N 倍（原计划已知）
  2. **`request_history` 无界增长**。`defaultdict(deque)` 为每个见过的 IP
     建一个 deque 且**永不清理** —— 只有该 IP 再次访问才会清掉它自己的
     过期条目。被换 IP 扫描就会持续吃内存。
  3. **`x-forwarded-for` 无条件信任**（`rate_limit.py:35-39`）。攻击者
     每个请求伪造一个不同的 XFF 即可完全绕过限流，同时放大第 2 条。
     修法是只在可信代理后才采信该头（配置可信代理列表或跳数）。
- ~~`streaming_ingestion.py` 的 `asyncio.sleep(0)`~~ **已修**（08ec861）。
  记住这条结论：**`sleep(0)` 不能让阻塞调用变成非阻塞** —— 它只在调用
  之前让出一次控制权，紧随其后的同步计算照样霸占事件循环。
  六处改成了 `asyncio.to_thread`。
- **`upload.py:105` 的同步端点完全没有 offload**。`ingestion.ingest_file()`
  直接在 `async def` 里跑，10MB 文件的 embedding 期间整个进程停摆。
  T6 改异步摄入时这个端点会被替换，但在那之前它是最大的单点阻塞。
- **`routes.py` 的 ChromaDB 调用直接在 `async def` 里**（`VectorDB()`、
  `collection.get/delete/count`）。其中 `list_documents(include_chunks=True)`
  会拉全库文档内容，是同步阻塞调用。数据量小时无感，语料涨上去会明显。
- **`_sync_generator_to_async` 每个流式请求占用两份线程资源**
  （`chat_service.py:22-51`）：一个专属 `threading.Thread` 跑生产者，
  外加 `asyncio.to_thread(queue.get)` **阻塞占用默认线程池的一个 worker**
  等待数据。默认池只有 `min(32, cpu+4)` 个 worker，并发流式请求多了会
  把池占满，而 `to_thread` 的其它用途（检索、指代消解）也共用这个池。
- `Index("ix_runs_created_desc", "created_at")` 实际是升序索引，名字里的
  desc 名不副实。单列索引可反向扫描，功能上没问题，仅命名误导。
- `scripts/` 下 4 个历史调试脚本仍引用已删配置
  （`4_run_rag.py`、`5_test_hybrid_rag.py`、`9_test_hybrid.py`、`7_test_rerank.py`），
  运行会 ImportError。非产品代码，清理或归档。
- `chat_service.py` 的"查询 ≤20 字则注入 jieba 实体"补救逻辑，
  以及 Multi-Query 扩展，当初都是为弥补 MiniLM 失效而加的。
  换中文模型后必要性需重新评估。
- **requests 在 hf-mirror 上约 12 KB/s，curl 能跑 9 MB/s**（差 700 倍）。
  大文件必须用 curl，已在 `t2ranking.py` 实现并记入 ADR-002。
- 所有 `scripts/` 入口必须把 stdout 切 UTF-8，否则 Windows GBK 控制台
  遇中文即 UnicodeEncodeError。
- 智谱拒绝 `temperature=1e-8`（要求两位小数），但接受 `0.0` 与 `0.01`。
