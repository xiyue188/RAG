# 总体规划

> 唯一的全局路线图。各阶段细节见 `docs/plans/M{n}-*.md`，
> 当前进度见 `STATUS.md`（那是导航，这里是蓝图）。

## 目标

把项目做成**可追溯的 RAG 知识库研究 Agent**：不只给答案，
还能展示每一步决策、每一条证据、每一次工具调用，并且这些都有指标支撑。

`data/documents/` 下的海事文档只是 demo，主评测走公开数据集。

## 三层结构

```
第一层  把 RAG 做正确        M0 · M1
第二层  加真正有用的 Agentic   M2 · M3 · M4
第三层  做成生产式项目        M5
```

## 阶段总览

| 阶段 | 内容 | 预算 | 状态 | 计划书 |
|---|---|---|---|---|
| M0 | 地基修正：分数口径、中文模型、RRF、chunk 定位 | 6-8h | **完成** | [M0](plans/M0-foundation.md) |
| M1 | 评测集与检索实验 | 14-18h | **完成** | [M1](plans/M1-evaluation.md) |
| M2 | 基础设施：PG + Redis + 异步摄入 | 8-10h | **进行中** | [M2](plans/M2-infrastructure.md) |
| M3 | Agentic 编排：路由、工具、引用校验 | 18-22h | 未开始 | 待写 |
| M4 | 大脑面板改造：轨迹、证据、校验视图 | 8-10h | 未开始 | 待写 |
| M5 | 测试、Docker、README、安全 | 8-10h | 未开始 | 待写 |

累计约 60-80 小时。

## 依赖关系

```
M0 ──► M1 ──► M2 ──► M3 ──► M4 ──► M5
        │              ▲
        └── 提供延迟与召回基线 ┘
```

- **M1 阻塞 M3**：路由设计需要知道快速 RAG 的延迟与召回基线，
  否则无法判断"什么问题值得进 Agent"。
- **M2 阻塞 M3**：Agent 的执行轨迹要落库才能"可追溯"，
  内存态轨迹在面试时站不住。
- **M3 阻塞 M4**：面板展示的是 Agent 的产出，Agent 不存在则无可展示。
- **M4 与 M3 紧邻**：Agent 做完若无对应展示，连自己都无法验证跑得对不对。

## 各阶段要交付什么

### M0 · 地基修正（已完成）

修掉三个使指标失效的问题：embedding 语言不匹配、双义阈值、融合尺度不可比。
交付：`rag/scoring.py` 作为全系统分数口径唯一来源。

关键结果：换中文模型使 Recall@5 从 0.023 到 0.707。

### M1 · 评测集与检索实验（已完成）

用公开人工精标数据集量化 5 个检索方案，产出可写进简历的对比报告。

已完成：
- T2Ranking 接入（300 query / 13,536 段语料）
- 检索指标自研（Recall/MRR/nDCG/延迟分位，35 条单测）
- 5 方案实验与报告
- 无答案阈值校准 —— 证明单一阈值不可行

生成层：忠实度 0.916（44 条有效），人工核验一致率 85%。
判分器自实现而非用 ragas —— 后者按 temperature=1e-8 调用（智谱拒绝），
且英文 prompt 使 GLM 把 JSON 包在代码块里导致断言抽取失败。

### M2 · 基础设施

PG 存关系数据与可追溯轨迹，Redis 存缓存与短期状态。向量继续留在 ChromaDB。

表结构要点：

```
users / documents / sessions / messages
runs        (trace_id, route, query, total_ms, first_token_ms, tokens, status)
run_steps   (run_id, seq, node, ms, state_snapshot jsonb)   ← 面板回放的数据源
tool_calls  (run_id, seq, tool, args jsonb, ms, ok, idempotency_key)
evidence    (run_id, chunk_id, file, relevance, used_in_answer)
citations   (message_id, sentence_idx, chunk_id, verified, verify_score)
feedback / eval_runs / eval_results / ingest_tasks
```

Redis 的 key 设计要点：检索缓存带 `kb_version`，文档变更时 `INCR` 一次，
旧 key 自然失效，不必遍历删除。

文档解析异步化：`upload → 建 task → 立即返回 task_id → 后台 worker →
进度写 Redis → GET /tasks/{id}/events`。用 `asyncio.Queue` + 常驻 worker，
不引入 Celery。

验收：`docker compose up` 起 4 服务；上传 5MB PDF 立即返回不阻塞；
重启后 runs 表数据仍在。

### M3 · Agentic 编排（核心）

手写状态机，`AgentState` dataclass + 每个节点 `async def node(state, emit) -> str`。
节点签名对齐 LangGraph 的 StateGraph 形状，但不引入该依赖
（理由见下方「已定决策」）。

流程：

```
route → direct_rag / clarify / agent
        agent → plan → act(工具) → evidence_check
                          ↑            │
                          └── rewrite_query（最多 2 次）
                                       ↓
                         generate_with_citations → citation_verify
```

**路由不要全用 LLM**：先走规则（问候语、超短查询、明确单跳意图 → direct_rag），
规则拿不准才调小模型做结构化分类。理由是成本、延迟，以及路由准确率
可以单独进评测集。目标 `direct_rag` 占比 ≥ 60%。

四个工具，统一 Protocol + Pydantic 入参校验：

| 工具 | 要点 |
|---|---|
| `search_knowledge_base` | 走 M1 选出的最优方案 |
| `get_document_context` | 按 `doc_key + seq` 取相邻 chunk（M0 已备好字段） |
| `calculate` | **绝不用 eval**。`ast.parse` + 节点白名单 |
| `web_search` | 默认关闭需授权。SSRF 防护：仅 https、拒私网 IP、禁跨主机重定向、响应限 512KB |

横切关注点放在 dispatcher 做一次，不散进每个工具：超时、幂等、
重试、取消检查、落库、SSE 推送。

**幂等分两个键，不共用一个字段**（见 [ADR-003](decisions/ADR-003-tool-call-idempotency-key.md)）：

- `idempotency_key = sha1(run_id + seq + tool + canonical_json(args))`
  —— run 内防重放，**带 UNIQUE 约束**，先插占位行再执行（并发下靠
  约束冲突判重，不靠"先查再写"，后者两个请求会同时查到空）
- `ret:{tool}:{sha1(args)}:{kb_version}` —— 跨 run 结果缓存，放 Redis 带 TTL，
  **只对声明 `side_effects=False` 的只读工具启用**

原先定的全局键 `sha1(tool + args + kb_version)` 已废弃：它不含 run_id，
会把"两次都该生效的相同请求"判成重复。本项目四个工具恰好都只读，
所以这个错误暂时看不出后果 —— 但形状错了就该改。

防 Agent 死循环**不用幂等键做**，用每 run 的工具调用次数上限 +
同一 `(tool, args)` 在本 run 内的重复次数上限。那是控制流问题。

**citation_verify 是简历上最能打的一环**：答案按标点切句，
每个带 `[doc_X]` 的句子用 cross-encoder 算 (句子, 被引 chunk) 的支持度，
低于阈值标 `verified=false`。产出引用准确率与引用覆盖率两个指标。

**独立的 answerability 判断**（M1 遗留）：单一 relevance 阈值已证明不可行
（有答案与无答案的 top1 分布严重重叠，误拒率 ≤10% 时只能识别 19.1% 的无答案查询）。
根本原因：**语义相似 ≠ 包含答案**，用单一相关性分数做可答性判断是错误的建模。

**演进路径（三步，按阶段推进）**：

第一步（M3 范围，无需训练数据，立刻可做）：把 `assess_context` 从"top1 绝对值 > 阈值"
改为多信号联合判断：
- top1 分数超阈值（保留现有逻辑）
- **top1 - top2 分差足够大**（新信号）：分差大说明 top1 是真正命中而非"同领域但无答案"；
  分差小但 top1 很高说明召回结果都差不多好、不能确认有清晰答案
- **top-k 衰减斜率**（可选增强）：知识库有答案时多个文档应同时命中，
  分布应平坦（`[0.85, 0.82, 0.79]`）；无答案时只有一个偶然命中，
  随后迅速下降（`[0.82, 0.41, 0.30]`）

第二步（M3 后期，有计算成本但最直接）：对"灰色区间"（top1 分数在阈值附近的请求），
发一次廉价 LLM 调用："给定以下文档片段，这个问题能否被回答？是/否"。
直接绕过"语义相似 ≠ 包含答案"的根本问题，不需要任何训练数据。

第三步（M5 范围，需要线上积累数据后）：把线上可答/不可答判断结果收集标注，
训练轻量分类器（逻辑回归或 LightGBM）替代规则，多特征联合决策：
`[top1_relevance, top1_top2_diff, top3_avg, decay_slope, query_length, ...]`。

重做时**统一搬到 `rag/answerability.py`**，输出带 `reason` 与 `signals`：
拒答时说得出为什么，对齐"禁止无声降级"。当前的 `assess_context`（`rag/llm.py:26`）
是中间过渡实现，M3 第一步完成后替换掉它。

降级链：主模型 → 重试 1 次（指数退避 + jitter）→ 降级小模型 →
仍失败则返回已检索到的证据 + 明确失败说明。绝不静默返回空。

### M4 · 大脑面板改造

现在是单一日志流，Agent 的多步执行在里面会糊成一团。改成三个 tab：

- **执行轨迹**：节点时间轴、每步耗时、当前节点高亮、重试标记
- **工具与证据**：工具卡片（名称/参数摘要/耗时/成败）+ 证据表
  （chunk、file、各路分数、是否被答案采用）
- **引用校验**：答案句 ↔ 原文映射，未通过校验的标红

底部固定栏：路由结果 · 总 token · 端到端延迟 · trace_id（可复制）

**明确不展示**：模型隐藏思维链、完整系统提示词、API Key。
这一点写进 README —— 多数模型 API 不提供真实内部推理，
不宣传"展示完整思维链"本身是专业信号。

新增 SSE 事件（旧事件全部保留）：`run_start / node_start / node_end /
route_decided / plan_created / tool_start / tool_end / evidence_collected /
retry / citation_verified / run_end`

SSE 断线恢复：客户端带 `Last-Event-ID`，服务端从 Redis Stream 补发；
每 15s 心跳；重连后若 run 已终结直接补发终态。

### M5 · 生产化收尾

- 单测：chunker、RRF、metrics、`calculate` 白名单（含恶意输入）、
  SSRF 判定、幂等键、路由规则
- 集成测试：完整 direct_rag / agent 链路、取消、超时降级。
  用 fake LLM provider，CI 不烧真 token
- GitHub Actions：lint + test
- README 按 18 节结构，首屏是「这是什么 / 解决什么问题 / 在线地址 /
  截图 / 如何启动」
- 视觉打磨（配色、间距、空状态）放在这一步之后

## 已定决策

| 决策 | 结论 | 理由 |
|---|---|---|
| 编排框架 | 手写状态机，不用 LangGraph | 节点签名对齐 StateGraph 形状即可讲清模型；checkpointer 的价值在 10-30 秒的 run 上有限；换来的是事件推送与落库时机的精确控制。详见 README 计划中的「为什么没用 LangGraph」一节 |
| 向量库 | ChromaDB，不迁 pgvector/Milvus | 数据量用不到其优势；省下的时间投到 citation_verify 与评测收益更高 |
| 关系库 | PostgreSQL | 轨迹要能按 trace_id 查询与聚合，内存态站不住 |
| 缓存 | Redis，不作为事实来源 | 只放缓存/限流/取消信号/事件流 |
| 检索融合 | RRF，无权重参数 | 两路分数量纲不同，加权相加无物理意义。见 [ADR-001](decisions/ADR-001-rrf-over-weighted-fusion.md) |
| 工具幂等 | 拆成 run 内步骤键 + 跨 run 缓存键 | 一个字段塞两种语义会让全局键把合法的重复请求判成重复。唯一性交给数据库约束而非应用层"先查再写"。见 [ADR-003](decisions/ADR-003-tool-call-idempotency-key.md) |
| 分层与端口 | 混合：只在 LLM、工具、可答性三处引入端口，其余保持直接依赖 | 多数外部依赖只有一个实现且已决定不换，为它们写 Protocol 得到的是"一个接口一个实现"的空抽象。但规则漂移已真实发生（可答性判断漂进了编排层），关键边界需要接口挡住。见 [ADR-004](decisions/ADR-004-layering-and-rule-placement.md) |
| 评测数据 | T2Ranking + CRUD-RAG，不自己标注 | 规模、避免循环论证、可比性。见 [ADR-002](decisions/ADR-002-eval-dataset-choice.md) |
| 指标实现 | 检索层自研，生成层用 Ragas | 确定性运算应可单测；语义判断才交给 LLM，且必须人工抽检报一致率 |

## 跨阶段的未决问题

- **无答案识别**：单一阈值不可行（M1 已证明），M3 需引入独立判断
- **rerank 可用性**：质量增益真实（MRR@10 0.947 最高）但 CPU 延迟 9.8 秒。
  待验证：GPU / 削减候选数 / 按路由结果选择性启用
- **BM25 的价值未被证实**：T2Ranking 不带 query 类型标注，
  无法验证"BM25 在关键词类查询上更强"。需要分层数据或自建小型分类子集
- **历史补丁的必要性**：`chat_service` 的 jieba 实体注入、Multi-Query 扩展
  都是为弥补失效 encoder 而加的，换模型后是否还需要，待评测验证
