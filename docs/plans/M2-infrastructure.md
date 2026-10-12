# M2 · 基础设施：PostgreSQL + Redis + 异步摄入

**目标**：让"可追溯"落到磁盘上 —— 执行轨迹、工具调用、证据、引用
全部可按 `trace_id` 查询与聚合。
**预算**：8-10 小时
**前置**：M1 完成（提供延迟与召回基线）
**阻塞**：M3（Agent 轨迹必须能落库，内存态轨迹在面试时站不住）

## 为什么现在做

当前有两处内存态，进程重启即丢、多副本各算一套：

| 位置 | 现状 | 问题 |
|---|---|---|
| `backend/rate_limit.py` | `defaultdict(deque)` 存 IP 时间戳 | 重启后限流归零；多副本时实际配额翻倍 |
| `chat_service.sessions` | `Dict[str, ConversationManager]` | 重启丢会话；`cleanup_old_sessions` 需手动调用 |

M3 会引入 Agent 的多步执行，每步的耗时、工具入参、证据来源都要能事后
查询。这些数据放内存等于没有 —— "可追溯"这个卖点必须有持久化支撑。

## 数据流与调用链

```
                      ┌──────────────── 写路径 ────────────────┐
HTTP POST /chat/stream
   │
   ├─ RateLimitMiddleware ──► Redis: INCR rl:{ip}:{window}  (原子，带 TTL)
   │                            └─ 超限 → 429，不进业务
   │
   ├─ TraceContext 生成 trace_id (uuid4)
   │
   ├─ RunRepository.create() ──► PG: INSERT runs (trace_id, query, status='running')
   │                                   ↓ 返回 run_id
   ├─ 检索
   │   ├─ Redis GET ret:{variant}:{sha1(query)}:{kb_version}   命中则跳过检索
   │   ├─ ChromaDB 向量检索 + BM25
   │   └─ EvidenceRepository.bulk_insert() ──► PG: INSERT evidence[]
   │
   ├─ 生成（SSE 逐 token 下发）
   │
   └─ RunRepository.finish() ──► PG: UPDATE runs SET status='ok', total_ms, tokens
                                     INSERT run_steps[]  (每节点一行)

                      ┌──────────────── 读路径 ────────────────┐
GET /api/v1/runs/{trace_id}
   └─ PG: SELECT runs JOIN run_steps JOIN tool_calls JOIN evidence
          → 面板回放的数据源（M4）
```

**上传的异步链路**：

```
POST /documents/upload
   ├─ 校验类型与大小 → 写临时文件
   ├─ IngestTaskRepository.create() ──► PG: INSERT ingest_tasks (status='pending')
   ├─ asyncio.Queue.put(task_id)      ← 立即返回，不阻塞 HTTP
   └─ 200 {task_id}

后台常驻 worker（lifespan 启动）
   ├─ queue.get() → 解析 → 分块 → embedding → 入库
   ├─ 每阶段 Redis SETEX ingest:{task_id} {progress}   (前端轮询/SSE 读这里)
   ├─ Redis INCR kb_version                            (检索缓存自然失效)
   └─ PG: UPDATE ingest_tasks SET status='done', chunk_count

GET /documents/tasks/{task_id}/events  → SSE，从 Redis 读进度
```

## 表结构

```sql
-- 会话与消息（替换 chat_service 的内存字典）
sessions(id uuid pk, created_at, last_active_at, meta jsonb)
messages(id bigserial pk, session_id fk, role, content, created_at,
         run_id fk null)                      -- 关联到产生该回答的 run

-- 可追溯的核心
runs(id bigserial pk, trace_id uuid unique, session_id fk null,
     route text, query text, status text,      -- running/ok/error/cancelled
     total_ms int, first_token_ms int,
     prompt_tokens int, completion_tokens int,
     error text null, created_at, finished_at null)

run_steps(id bigserial pk, run_id fk, seq int, node text,
          ms int, state_snapshot jsonb,        -- M4 回放的数据源
          unique(run_id, seq))

tool_calls(id bigserial pk, run_id fk, seq int, tool text,
           args jsonb, result_summary text, ms int, ok bool,
           idempotency_key text,               -- M3 才写入，见 ADR-003
           unique(run_id, seq))

evidence(id bigserial pk, run_id fk, chunk_id text, file text,
         relevance real, rank int, retrieved_by text[],
         used_in_answer bool default false)

citations(id bigserial pk, message_id fk, sentence_idx int,
          chunk_id text, verified bool null, verify_score real null)

-- 反馈与评测
feedback(id bigserial pk, message_id fk, rating smallint, comment text, created_at)
eval_runs(id bigserial pk, suite text, variant text, started_at, finished_at, meta jsonb)
eval_results(id bigserial pk, eval_run_id fk, qid text, metrics jsonb)

-- 异步摄入
ingest_tasks(id uuid pk, filename text, size_bytes int, category text,
             status text, chunk_count int null, error text null,
             created_at, finished_at null)
```

索引要点：
- `runs(trace_id)` unique —— 按 trace_id 查询是主路径
- `runs(created_at desc)` —— 面板列出最近的 run
- `evidence(run_id)`、`run_steps(run_id, seq)`、`tool_calls(run_id, seq)`
- `messages(session_id, created_at)`

`state_snapshot` 与 `args` 用 `jsonb` 而非 `text`：需要按字段查询
（如"找出所有调用了 calculate 且失败的 run"）。

## Redis key 设计

| key | 类型 | TTL | 用途 |
|---|---|---|---|
| `rl:{ip}:{window}` | string(int) | 窗口长度 | 限流计数，`INCR` + `EXPIRE` |
| `kb_version` | string(int) | 无 | 知识库版本，文档变更时 `INCR` |
| `ret:{variant}:{sha1(query)}:{kb_ver}` | string(json) | 1h | 检索结果缓存 |
| `emb:{model}:{sha1(text)}` | string(json) | 24h | 查询向量缓存 |
| `ingest:{task_id}` | hash | 1h | 摄入进度 |
| `cancel:{trace_id}` | string | 10min | 取消信号，节点边界检查 |

**`kb_version` 嵌进 key 而非用它做失效判断**：文档一变就 `INCR`，
旧 key 因前缀不匹配自然失效，不必遍历删除（`KEYS` 在生产是禁用操作，
`SCAN` 也要 O(n)）。这替换掉现在 `retriever.invalidate_bm25_cache()`
那套手工失效逻辑。

**Redis 不作为事实来源**：全部内容可丢弃后重建。
限流计数丢了最多放过一个窗口的请求；缓存丢了只是变慢。

## 任务清单

> **顺序**：T1 → T2 → T3 → **T0（插入）** → T5 → T4 → T6。
> T0 是后加的架构收口，插在 T5 之前 —— 它修的三条正在产生错误输出，
> 而它建立的机制会保护后面所有任务。

### T0 · 架构收口（插入，约 4h）

**为什么插队**：T3 完成后做了一次审计，发现五处规则漂移，其中一处
**正在给用户看错误信息**。更重要的是，`CLAUDE.md` 里的架构约束一直只是
文字，没有任何机制保障 —— 五处漂移全部发生在"约定存在但无人执行"的地方。

遵循三条原则（2026 年 FastAPI 社区的实际共识，不是某个具名架构）：

1. **依赖方向指向领域** —— 已基本满足（`rag/` 不 import `backend/`），
   但 `backend/api/` 直接 import `rag/` 绕过了 service
2. **I/O 集中在边缘** —— 检索路径已满足，**摄入与 `routes.py` 未满足**
3. **用工具而非文档保障前两条** —— 完全缺失，这是本次最高杠杆的一项

每格一个提交，**每个提交自洽、可回滚、测试全绿**。防漂移规则随对应修复
一起加，不先写一批红灯测试再慢慢修。

#### T0a · 阈值单一真相源（1h）

同一个概念现在有四个值，且两个前端组件互相矛盾：

| 位置 | 值 |
|---|---|
| `config.py:163` 默认值 | `0.50` |
| `.env:35` 实际生效 | **`0.75`** |
| `BrainPanel.vue:141-148` | `50`（注释声称"与后端 config 对齐"，已过期） |
| `ChatPanel.vue:585,591` | `0.75` |

后果：relevance = 0.60 时，仪表盘显示蓝色"足以支撑基于文档的回答"，
同一次回答的引用卡片显示橙色，而后端按 0.75 判定**根本没用这些证据**。

- 后端加 `GET /api/v1/config/thresholds`，返回 `{retrieval_min, answerable_min}`
- 前端启动时拉一次，删除两个组件里的硬编码与过期注释
- 统一 `config.py` 默认值与 `.env`（按 `docs/eval/threshold.md` 的校准结论定）
- 架构测试：`frontend-vue/src/` 不得出现领域阈值字面量

**验收**：改 `.env` 里的阈值，刷新页面后前端配色随之改变，无需改前端代码。

#### T0b · 删可答性判断的副本（0.5h）

`rag/llm.py:26 assess_context()` 是权威实现。`chat_service.py:304-329`
是它的手抄副本 —— `relevance_of` / `top_relevance = max(...)` /
`has_signal = any(...)` / 阈值比较，四部分逐一对应。

更糟的是 `chat_service.py:379` 又把 `answerable_min` 传进
`answer_smart_stream`，让 `rag/` 再算一遍。**同一判断一次请求执行两次。**

- 删掉 service 里的副本，改调 `assess_context`
- 架构测试：`backend/services/` 不得 import `ANSWERABLE_MIN_RELEVANCE`

**验收**：`chat_service.py` 不再出现该常量；SSE 的可答性日志字段不变。

#### T0c · I/O 边缘化（3h，拆成四个子提交）

"I/O 在边缘"有两层含义，本项目两层都缺。下面列出**全部**违规点，
一处不漏；每个子提交独立可回滚。

**第一层违规：阻塞调用跑在事件循环里。**

`async def` 里的同步调用不会让出控制权，期间**整个进程**对所有请求失去
响应 —— 包括 Docker 的健康检查，而健康检查超时会导致容器被重启。

| 文件:行 | 调用 | 阻塞类型 | 严重度 |
|---|---|---|---|
| `upload.py:104` | `ingestion.ingest_file()` | 解析+分块+embedding+写库，**零 offload** | **最高** —— 10MB 文件期间进程停摆 |
| `upload.py:89,227` | `NamedTemporaryFile().write()` | 磁盘 I/O | 中 |
| `upload.py:120,256` | `Path.unlink()` | 磁盘 I/O | 低 |
| `upload.py:147,267` | `invalidate_bm25_cache()` | 重建 BM25 索引 | 中 |
| `upload.py:32` | `VectorDB()` / `Embedder()` | 首次调用加载模型 | 中 |
| `routes.py:97` | `collection.get(include=[...,"documents"])` | **拉全库正文** | **高** —— 随语料线性增长 |
| `routes.py:130,162,191` | `collection.get(...)` | ChromaDB 查询 | 中 |
| `routes.py:205` | `collection.delete(ids=...)` | ChromaDB 写 | 中 |
| `routes.py:232` | `collection.count()` | ChromaDB 查询 | 低 |
| `routes.py:208` | `invalidate_bm25_cache()` | 重建 BM25 索引 | 中 |
| `routes.py:92,160,187,230` | `VectorDB()` | **每请求新建客户端** | 中 |

**第二层违规：领域数据的变换发生在协议层。**

协议层只该做两件事：把 HTTP 请求解析成参数、把结果组装成响应。
数据怎么变换、从哪取、要不要缓存，都不是它的事。

| 文件:行 | 问题 | 判定依据 |
|---|---|---|
| `routes.py:99-127` | 60 行"按文件分组 chunk"逻辑 | 换成 CLI 一字不变 → 应用逻辑 |
| `routes.py:132-147` | 另一份分组逻辑（`include_chunks=False` 分支） | 同上，且与上一条重复 |
| `routes.py:9` | `from rag import VectorDB` | api 不该知道向量库存在 |
| `routes.py:208` | `get_chat_service().retriever.invalidate_bm25_cache()` | 穿三层拿内部对象的内部方法 |
| `upload.py:16` | `from rag import DocumentIngestion, VectorDB, Embedder` | 同 `routes.py:9` |
| `upload.py:23,183-190` | 文件类型白名单 + 全量读入内存 | 白名单是领域规则；全量读入是资源策略 |

**第三层（附带发现）：资源与错误处理。**

- `upload.py:183-190` 把**所有**文件全量读进内存才开始处理。
  注释说是为了"避免 SSE 生成器内文件句柄已关闭"—— 理由成立，
  但 10 个 10MB 文件 = 100MB 常驻。改为逐个落临时盘再处理。
- `upload.py:104` 的 `ingest_file` 抛异常时，`finally` 里的 `unlink`
  会执行，但**临时文件在异常路径上仍可能残留**（写入一半时进程被杀）。
  这属于已知限制，记录不修。
- `routes.py` 六个端点各自 `try/except Exception` 后 `raise HTTPException`，
  与 `main.py` 的全局异常处理器重复。不在本次范围，记进待办观察。

**子提交划分：**

| # | 内容 | 验收 |
|---|---|---|
| c1 | 新建 `backend/services/document_service.py`：`VectorDB` 模块级单例 + 四个方法（`list_documents` / `get_chunks` / `delete_document` / `stats`），全部 `to_thread`，含分组逻辑 | 新增单测覆盖分组逻辑（含空库、单文件多 chunk、`include_chunks` 两分支） |
| c2 | `routes.py` 改为调用 `DocumentService`，删掉 `from rag import VectorDB` 与两处分组逻辑 | 六个端点行为不变；文件从 245 行降到 ~120 行 |
| c3 | `upload.py` 两个端点全部 offload，`get_ingestion()` 的初始化也走线程 | 上传 10MB 文件期间 `/health` 1 秒内返回 |
| c4 | `upload.py` 流式端点改为逐文件处理，不再全量读入内存 | 10 个文件上传期间内存增量 < 20MB |

**T0c 整体验收**：
1. 上传 10MB 文件期间，另一个终端 `curl /health` 在 1 秒内返回
2. `grep -rn "from rag import" backend/api/` 结果为空
3. `grep -c "async def" backend/api/routes.py` 的每个端点体内无同步 ChromaDB 调用
4. 228 条既有测试 + 新增 DocumentService 测试全绿

#### T0d · 架构约束测试（1h，规则随 T0a-T0c 分批加入）

**这是本次最高杠杆的一项。** 前几条约束在 `CLAUDE.md` 里写了很久，
五处漂移照样发生 —— **写在文档里的架构约束等于没有约束。**

##### 为什么用 AST 而不是 import-linter

| | `import-linter` | **AST（采纳）** |
|---|---|---|
| 依赖 | 新增一个包 + `.importlinter` 配置文件 | 只用标准库 `ast` |
| 能查 import | 是 | 是 |
| 能查"不得出现 `.commit()`" | **不能** | 能 |
| 能查前端 `.vue` 里的字面量 | **不能**（只认 Python） | 能（正则那一条） |
| 能查"同步调用在 async def 里" | **不能** | 能（AST 遍历函数体） |
| 失败信息 | 独立命令的输出 | 与 pytest 其余测试同一处 |

本项目要守的五条里有三条超出了 import 图的范围，而 `import-linter` 只解决
import 层次。为一条规则引入一个依赖 + 一份独立配置，另外三条还得再写
AST —— 不如统一用 AST。**已有先例**：`test_repositories.py::TestNoCommit`
的源码扫描就是这个思路，本次把它归并进来并扩大范围。

用 `ast.parse` 而非正则扫 Python：正则会把注释、docstring、字符串里的
`import rag` 当成真 import。前端 `.vue` 没有 Python AST 可用，那一条用正则
但排除注释行。

##### 五条规则

| # | 规则 | 现状 | 满足时机 |
|---|---|---|---|
| 1 | `rag/` 不得 import `backend/` | **已满足** | 钉住（CLAUDE.md 硬约束） |
| 2 | `backend/api/` 不得 import `rag` | 违反（`routes.py:9`、`upload.py:16`） | T0c |
| 3 | `backend/repositories/` 不得出现 `.commit()` | 已满足 | 从 `test_repositories` 迁入 |
| 4 | `backend/services/` 不得 import 领域阈值常量 | 违反（`chat_service.py:304`） | T0b |
| 5 | `frontend-vue/src/` 不得出现领域阈值字面量 | 违反（两个 `.vue`） | T0a |

**规则 4 的边界**：禁的是阈值常量（`ANSWERABLE_MIN_RELEVANCE` /
`RETRIEVAL_MIN_RELEVANCE`）—— 它们是领域规则的参数，用它们意味着在
service 层做领域判断。**不禁** service 层读 `config` 的其它值
（如 `SESSION_TIMEOUT`），那些是应用配置。

**规则 2 的例外**：`backend/api/` 可以 import `backend.schemas`
（那是协议层自己的 DTO）。测试要能区分 `rag` 与 `backend.*`。

##### 每条规则都要有"反向测试"

只断言"当前代码合规"是不够的 —— 规则写错了（比如永远返回通过）
也会绿。每条规则配一个用例：喂一段**故意违规的源码字符串**给检查函数，
断言它**返回违规**。这样测试本身的有效性也被覆盖。

**验收**：
1. `pytest tests/test_architecture.py` 全绿
2. 手工在 `backend/api/routes.py` 加一行 `from rag import VectorDB`，
   测试**必须失败**并指出文件与行号；删掉后恢复绿
3. 每条规则的反向测试都能捕获人造违规

#### T0e · 文档更正（0.5h）

- ADR-004 阶段 1 的描述写错了（说"搬家到 `rag/`"，实际是"删副本"）
- STATUS.md 的「下一步」改为单一有序清单

### T1 · 依赖与容器（1h）

- `docker-compose.yml` 加 `postgres:16-alpine` + `redis:7-alpine`
- 依赖：`sqlalchemy[asyncio]`、`asyncpg`、`alembic`、`redis`
- `config.py` 加 `DATABASE_URL`、`REDIS_URL`，`.env.example` 同步
- 健康检查：PG 用 `pg_isready`，Redis 用 `redis-cli ping`；
  backend 的 `depends_on` 改为 `condition: service_healthy`

### T2 · ORM 模型与迁移（2h）

- `backend/db/models.py` —— 上述全部表
- `backend/db/session.py` —— async engine + sessionmaker，
  FastAPI 依赖注入提供 session
- Alembic 初始化 + 首个迁移
- **不引入 pgvector**（ROADMAP 已定）

实施中发现并修掉的两处（原计划没预见）：

- **`BigInteger` 主键在 SQLite 上不自增**。SQLite 的隐式自增要求列类型名
  恰好是 `"INTEGER"`，`BIGINT` 有整数亲和性但不是 rowid 别名，插入直接报
  `NOT NULL constraint failed`。而 SQLite 正是默认配置与失败回退路径 ——
  等于 DB 层在默认配置下一行都写不进去。改用
  `BigInteger().with_variant(Integer, "sqlite")`。
- **`DateTime(timezone=True)` 在 SQLite 上不保留时区**。它没有原生时间类型，
  读回来一律 naive，于是 `created_at > utcnow()` 在 PG 上正常、
  在 SQLite 上抛 `can't compare offset-naive and offset-aware`。
  加 `UtcDateTime(TypeDecorator)` 在类型层抹平，手法同 `JSONField`
  抹平 JSONB/JSON。

### T3 · Repository 层（2h）

```
backend/repositories/
  runs.py       create / finish / add_step / get_by_trace
  evidence.py   bulk_insert / mark_used
  sessions.py   get_or_create / append_message / history
  ingest.py     create / update_status / get
```

Repository 而非在 service 里直接写 SQL：M4 的面板要复用同一套查询，
而 service 层已经很长（`chat_service.py` 500+ 行）。

### T4 · Redis 客户端与限流迁移（3h，原估 2h）

`redis_client.py` 已完成并有 41 条测试覆盖降级路径。剩余部分范围扩大 ——
代码核查发现原计划漏了几条，其中两条**单机就存在**。

#### T4a · 限流迁移

- `rate_limit.py` 改用 Redis `INCR`，**Redis 不可用时降级为放行**
  （限流是保护措施，不该因它自己挂掉而拒绝全部流量）。
  依赖 `incr()` 返回 `None` 而非 `0` 来区分"不知道"与"这个窗口还没请求"，
  该区分已有测试守着。

- **修 `x-forwarded-for` 无条件信任**（`rate_limit.py:35-39`）。
  XFF 是客户端可任意写的普通 HTTP 头，只有在自己的可信代理之后才有意义。
  当前代码直接取第一段，攻击者每个请求伪造一个不同 IP 即可**完全绕过限流**。
  修法：配置可信代理（列表或跳数），仅当请求来自可信代理时采信该头，
  否则用 `request.client.host`。

- **修 `request_history` 无界增长**。`defaultdict(deque)` 为每个见过的 IP
  建一条 deque 且永不清理 —— 清理逻辑只在该 IP 再次访问时触发。
  配合上一条可被打到 OOM。迁到 Redis 后由 `EXPIRE` 自然解决。

  **实际实现与本节原计划不同**：原计划是 Redis 不可用时直接放行，理由是
  "退回内存计数会把无界增长带回来"。实现时选择了退回**有界**内存计数
  （`OrderedDict` 上限 10000 个 IP，满了淘汰最早插入的）：无界增长的根因
  是没有上限，加上限即消除；而 Redis 挂掉时仍保留单机级别的保护，比完全
  放行更稳。代价是多副本下降级期间全局配额放大为 N 倍，见 STATUS 的说明。

#### T4b · 检索缓存接入 `kb_version` —— **已移出 M2**

收口时决定不在 M2 接入检索缓存（2026-10-12）。理由：

- M1 实测检索 P50 为 15–52ms（不开 rerank），缓存能省的时间远小于
  一次 LLM 生成，用户感知不到。
- 缓存会引入一条新的一致性风险（文档变更后返回旧结果），而目前没有任何
  测试覆盖"变更 → 失效"这条链路的端到端行为。
- 值得缓存的是 rerank（P50 9.8 秒），但 rerank 是否启用、如何启用本身
  还是 M3 的未决问题。缓存应在那之后按实际热点设计。

M2 保留的部分：`kb_version` 已在文档变更的唯一入口
（`DocumentService.invalidate_retrieval_caches`）自增，四条变更路径
（同步/流式/异步上传、删除）都会触发。将来接入缓存时 key 直接带上它即可。

#### 采用固定窗口而非滑动窗口（明确的 trade-off）

`rl:{ip}:{window}` + `INCR` + `EXPIRE` 是**固定窗口**。它有窗口边界突发：
配额 30/分钟时，在 00:59 发 30 个、01:01 再发 30 个，两秒内实际放过 60 个。

| 方案 | 实现 | 代价 |
|---|---|---|
| **固定窗口（选用）** | `INCR` + `EXPIRE` | 1 次往返；边界最坏 2 倍突发 |
| 滑动窗口 | ZSET，member=时间戳，`ZREMRANGEBYSCORE` + `ZCARD` | 每请求存一个 member，内存 O(配额)，需 Lua 保证原子 |
| 令牌桶 | Lua 维护 `{tokens, last_refill}` | 最平滑、支持突发额度，实现最复杂 |

选固定窗口：最坏放过 60 次/分钟对本项目不构成威胁，而 ZSET 方案要为每个
请求存一个 member 并写 Lua 保证三条命令的原子性。**知道自己选了较弱的方案
并说得出理由，比默认用了强方案却讲不清区别更有价值。**

#### 缓存三大问题的防御（随 T4b 移出 M2；接入缓存时一起交付，不事后补）

当前没有检索缓存，所以三个问题一个都不存在。
但一旦接入检索缓存就全都成立，且本项目的风险画像与典型 Web 应用不同 ——
**这里的"底层"不是一次廉价的数据库查询，而是 CPU 密集的 embedding
（外加可能 9.8 秒的 rerank）。**

| 问题 | 本项目形态 | 防御 |
|---|---|---|
| **穿透** | 攻击者构造无穷多随机 query，每个都 miss，每次都跑 embedding。**危害最大** | 限流（因此上升为保护 CPU 的必需品，不再是礼貌措施）+ **空/低质结果也缓存**，短 TTL。**布隆过滤器不适用** —— 它需要有限的合法 key 集合，而用户问题空间是开放的 |
| **击穿** | 热点 query 缓存过期瞬间，N 个并发请求同时重算 | 单飞（single-flight）：同一 key 只放一个去算。**用 `asyncio.Lock` per-key，不要重新引入分布式锁** —— 这是性能问题不是正确性问题，与移除分布式锁是同一条推理 |
| **雪崩 A**（Redis 挂） | **已解决**：全链路静默降级为穿透，变慢但结果正确 | 保持现状 |
| **雪崩 B**（批量过期） | 风险低：缓存随请求陆续写入，过期时间天然分散 | TTL 加随机抖动，成本极低 |
| **雪崩 B'**（`kb_version`） | `INCR` 使**全部**检索缓存一次性失效，等同一次可控雪崩 | 已知 trade-off，不是缺陷。文档变更不频繁；必要时 bump 后预热高频 query |

缓存空结果与「禁止无声降级」不冲突：缓存的是"检索不到"这个如实结论，
不是放宽阈值凑结果。

### T5 · 会话落库（1h）

`chat_service.sessions` 改为读写 PG。历史记录从 `messages` 表加载，
内存只留当前请求的上下文。

**实际实现**：每次问答结束时一个事务写 sessions / runs / evidence /
两条 messages（`chat_service._persist_run_async`）。内存里的
`ConversationManager` 作为热缓存保留，进程重启后首次访问某会话时
从 `messages` 表回填。

### T6 · 异步摄入（2h）

- `POST /documents/upload` 立即返回 `task_id`
- `asyncio.Queue` + lifespan 启动的常驻 worker（不引入 Celery ——
  单机部署，Celery 要多一个 broker 和一套部署复杂度）
- 进度写 Redis，`GET /documents/tasks/{id}/events` 读
- 完成后 `INCR kb_version`

**实际实现与计划的差异**：

- 新增独立端点 `POST /documents/upload/async`，原同步与 SSE 上传端点保留
  （前端在用，替换属于前端改造）。
- 进度用轮询 `GET /documents/tasks/{id}` 而非 SSE `/events`：任务状态以 PG
  为准、进度以 Redis 为准，一个 JSON 端点合并两者更简单；SSE 留给 M4。
- 加了清扫器：`pending` / `running` 超过 `INGEST_STALE_SECONDS` 标 error。
  临时文件路径只在内存队列里，进程重启后任务无法继续，只能让用户重传。
- **单副本设计**。多副本需要共享存储 + 数据库队列，见 `ingest_service.py`。

## 验收标准

| # | 标准 | 验证方式 |
|---|---|---|
| C1 | `docker compose up` 起 4 个服务且健康检查通过 | `docker compose ps` |
| C2 | `alembic upgrade head` 建出全部表 | `\dt` 输出 |
| C3 | 一次问答在 `runs` 表留下一行，含 trace_id 与耗时 | SQL 查询 |
| C4 | `evidence` 表记录本次用到的 chunk 与 relevance | SQL 查询 |
| C5 | 重启 backend 后 `runs` 数据仍在、会话历史仍可读 | 重启实测 |
| C6 | 限流计数存于 Redis，重启不归零 | `redis-cli GET rl:*` |
| C7 | Redis 停掉后请求仍能成功（降级放行） | 停容器实测 |
| C8 | 上传 5MB PDF 立即返回 task_id，不阻塞 | 计时实测 |
| C9 | 文档变更后 `kb_version` 自增（检索缓存部分随 T4b 移出 M2） | `redis-cli GET kb_version` |
| C10 | 全部测试通过，前端 `vue-tsc` 通过 | 命令输出 |

## 明确不做

- 不引入 pgvector（向量留在 ChromaDB）
- 不引入 Celery（`asyncio.Queue` 够用）
- **不做分布式锁**。曾实现过一版后移除：摄入逻辑本身幂等
  （`ingest_text` 与流式路径都先 `delete_by_file` 再插入），
  并发重复摄入不会破坏数据，最坏只是白算一次 embedding。
  锁在这里只是性能优化而非正确性需求，为此引入一套需要理解
  `SET NX PX` + Lua 释放 + 主从切换边界的机制不划算。
  真需要唯一性时正确做法是数据库唯一约束。
- 不做用户系统与鉴权（M5 视情况）
- 不做 Agent 编排（M3）
- 不改前端（M4 统一改）

## 失败条件与回退

- **Docker 起不来 PG** → 退回 SQLite（`aiosqlite`），
  同一套 SQLAlchemy 模型与 Alembic 迁移，只改 `DATABASE_URL`。
  在 README 注明"当前用 SQLite，多副本需换 PG"。
- **Redis 不可用** → 限流降级放行、缓存直接穿透。
  这条是设计要求，不是回退：把 Redis 当强依赖，
  系统可用性上限就等于 Redis 的可用性。
- **异步摄入导致进度事件丢失** → 保留现有同步 SSE 上传端点作为兜底，
  两个端点并存一个版本周期。
