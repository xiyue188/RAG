# M3 · 证据质量与 Agentic 编排

**目标**：让系统**知道自己什么时候不该回答**，并在需要多步检索时能可追溯地
执行 —— 每一步决策、每一次工具调用、每一条引用都落库、有指标。
**预算**：30-36 小时（ROADMAP 原估 18-22h，未含移交进来的遗留项，见下方清单）
**前置**：M2 完成（轨迹表、会话落库、Redis）
**阻塞**：M4（面板展示的是本阶段的产出）

## 为什么这是重心

M0–M2 把 RAG 做对、把数据落盘。但 M1 留下了一个没解决的根本问题：

> 单一 relevance 阈值无法判定可答性 —— 误拒率 ≤10% 时只能识别 19.1% 的
> 无答案查询。系统会拿无关文档自信地编答案。

可追溯 Agent 的卖点是"每一步都有证据"。如果系统连"证据够不够"都判断不了，
那么后面的路由、工具调用、引用校验都只是在给一个错误的结论加上更多的过程
记录。**所以 M3 先做证据质量，再做编排。**

## 本计划起草时的新发现（2026-10-13）

重读 M1 的运行记录（`docs/eval/runs/*.jsonl`）后发现三件事，改变了本阶段的设计：

1. **M1 基线的误拒率口径偏差。** 校准脚本对有答案样本取 `relevances[0]`
   （RRF 排第一那条），生产代码 `assess_context` 取 `max`。按生产口径重算：
   识别率仍是 19.1%，**误拒率从 5.7% 降到 3.0%**。本阶段一律按生产口径。
2. **rerank 分数本身就能把识别率提到 58.9%**（`rrf_rerank`，同样单阈值法，
   阈值 0.65，误拒 7.7%）—— 是向量分的 3 倍。M1 只校准了 `rrf`，没报告这个。
   cross-encoder 让 query 与文档相互注意，比双塔余弦更接近"能否回答"。
   **代价是 CPU 上 P50 9.8 秒。** 这把"可答性"与"rerank 可用性"两个遗留项
   绑在了一起，必须一起设计。
3. **生产阈值不分口径。** `ANSWERABLE_MIN_RELEVANCE=0.75` 只有一个值，但
   relevance 有两种口径（余弦 `1-d/2` 与 rerank `sigmoid(logit)`）。开启
   rerank 后同一个 0.75 的含义完全变了。阈值必须按 `relevance_basis` 分开。

另一个观察：语料是 300 条 query 的 gold 段落 + 8 倍**随机**干扰段落，剔除
gold 后剩下的基本是无关文档，余弦分却仍在 0.7–0.8。这是 bge 向量的
**各向异性**：任意两段中文的余弦都挤在一个窄的高位区间，**绝对值几乎不携带
信息**。由此推测相对特征（top1 比背景高出多少）比绝对阈值更有效 ——
这是 A1 要验证的假设，不是结论。

## 移交进 M3 的遗留项

每一项都在下方任务清单里有落点。没有落点的项不允许静默消失。

| 遗留项 | 来源 | 落点 |
|---|---|---|
| 单一阈值无法判定可答性 | M1 结论 3、ROADMAP 跨阶段未决 | **A1** → A3 → A4 |
| 可答性判断要搬到 `rag/` 并带 `reason`/`signals` | ADR-004 阶段 1 遗留、ROADMAP M3 节 | **A1** |
| 阈值不分 relevance 口径 | 本计划新发现 3 | **A1**（配置层）、A2（rerank 侧取值） |
| rerank 可用性：质量真实但 P50 9.8s | M1 结论、STATUS 未解决问题 | **A2** |
| 检索缓存（T4b）推迟到 rerank 策略定下后 | M2 收口决议 | **A2** 末尾决定做或不做 |
| BM25 价值未被证实 | ROADMAP 跨阶段未决、STATUS | **A5**（离线分层分析） |
| jieba 实体注入 / Multi-Query 是否仍必要 | ROADMAP 跨阶段未决、STATUS 待办观察 | **A5** |
| 工具幂等：步骤键 + UNIQUE + 先占位后执行 | ADR-003（实现推迟到 M3） | **B4** |
| 工具统一 Protocol（第一个真正的端口） | ADR-004 阶段 2 | **B4** |
| `run_steps` / `tool_calls` / `citations` 三张表无写入方 | M2 建表、M3 填充 | B2 / B4 / C1 |
| 取消信号 `cancel:{trace_id}` | M2 计划书 Redis key 表 | **B2** |
| `_sync_generator_to_async` 每个流式请求占两份线程资源 | STATUS 待办观察 | **B1**（Agent 多节点流式会放大它） |
| 引用准确率与覆盖率 | STATUS 未解决问题 | **C1 / C2** |
| 降级链：主模型→重试→小模型→返回证据+失败说明 | ROADMAP M3 节 | **B6** |

**明确留在 M5 的**：LLMProvider Port + fake provider（ADR-004 阶段 3）、Redis
熔断、端到端集成测试、会话保留策略。M3 的 LLM 相关单测用函数级替身，不建端口。

## 三条线与顺序

```
A 证据质量线（先做，决定后两条线的输入）
  A1 可答性 v1：规则 + 评测框架        ← 本阶段第一格
  A2 rerank 可用性                   ← 依赖 A1 的评测框架
  A3 可答性 v2：灰区 LLM 校验          ← 依赖 A1、A2 的结论
  A4 生产接入与阈值分口径               ← 依赖 A1-A3
  A5 历史补丁与 BM25 的离线评估         ← 独立，可穿插

B 编排线（A4 完成后）
  B1 AgentState + 节点 runner + 流式桥接重做
  B2 run_steps 落库 + 节点事件 + 取消
  B3 路由（规则优先）+ 路由评测集
  B4 Tool Protocol + dispatcher + 幂等（ADR-003）
  B5 四个工具
  B6 plan → act → evidence_check → rewrite 循环 + 降级链

C 引用校验线（B6 完成后）
  C1 citation_verify
  C2 引用准确率 / 覆盖率评测
```

**停下来 review 的点**：A1 完成、A4 完成、B4 完成、C2 完成（M3 结束）。

为什么 A 在 B 前面：B3 的路由要知道"直接 RAG 能答多少"，B6 的
`evidence_check` 节点本身就是可答性判断 —— A 线的产物是 B 线的零件。
先搭状态机再回头补判断逻辑，会让编排层先长出一份临时的判断规则，
那正是 ADR-004 记录过的规则漂移。

---

## A1 · 可答性 v1：规则版 + 评测框架（4-5h）

### 做什么

1. **`rag/answerability.py`**，纯函数、无 I/O：

   ```python
   @dataclass(frozen=True)
   class AnswerabilitySignals:
       basis: str                 # "cosine" | "rerank" | "none"
       n_scored: int              # 有相关性信号的条目数
       top1: float | None
       top2: float | None
       gap12: float | None        # top1 - top2
       gap_bg: float | None       # top1 - 背景分（候选池中位数）
       ...

   @dataclass(frozen=True)
   class AnswerabilityDecision:
       answerable: bool
       reason: str                # 人可读：为什么判为不可答
       signals: AnswerabilitySignals

   def extract_signals(scores, basis) -> AnswerabilitySignals
   def decide(signals, policy) -> AnswerabilityDecision
   ```

   **两个函数分开**：特征提取是事实，判定规则是策略。评测要对同一组特征
   试多种策略，生产只用一种 —— 不分开就得把特征提取抄进评测脚本。

2. **无信号条目不当 0 分。** BM25 独占命中的 relevance 是"缺失"，不是"0 分"。
   `extract_signals` 只用 `has_relevance_signal` 为真的条目；评测侧
   jsonl 丢失了这个标记，按 `0.0 ⇒ 缺失` 还原（余弦口径下真实 0.0 需要
   两向量反向，实际不出现 —— 这条假设写进代码注释）。

3. **`rag/eval/answerability.py`** 评测框架：
   - 从 run 记录构造配对样本：每条 query 产出一个有答案样本（原列表）
     和一个无答案样本（剔除全部 gold 后的列表）
   - 两种可见性：**top-3**（生产当前交给判断函数的条数）与 **top-20**
     （检索内部候选池）。前者回答"不改接口能做到多少"，后者回答
     "把候选分数传下去值不值"
   - **按 qid 分组的 5 折交叉验证**：在 4 折上选参数（误拒率 ≤10% 约束下
     识别率最高），在留出折上计分。同一 query 的两个样本必须在同一折，
     否则参数会在测试样本的配对样本上被调过
   - **bootstrap 95% 置信区间**（按 qid 重采样）：300 条样本上 2-3 个百分点
     的差异可能只是噪声，不报区间就无法判断"提升"是否成立

4. **候选规则族**（全部在同一协议下评测，不预设赢家）：

   | 规则 | 判可答条件 | 假设 |
   |---|---|---|
   | R0 基线 | `top1 ≥ a` | 即 M1 的单阈值 |
   | R1 分差 | `top1 ≥ a 且 gap12 ≥ b` | 计划原定方案 |
   | R2 背景差 | `gap_bg ≥ c` | 各向异性下相对值比绝对值有效 |
   | R3 组合 | `top1 ≥ a 且 gap_bg ≥ c` | |

   **关于 R1 的预期**：有答案的 query 平均有 ~5 条 gold，top1 与 top2
   常常都是 gold、分数相近，gap12 可能反而**小**。R1 可能不如基线 ——
   如果实测如此，如实报告，这正是要测而不是要猜的原因。

5. **报告** `docs/eval/answerability.md` + 脚本 `scripts/eval_answerability.py`。

### 验收

- `rag/answerability.py` 不 import `backend`、不做 I/O（架构测试规则 1 已覆盖）
- 单测覆盖：无信号、全缺失、单条结果、分数相等、basis 混合
- 报告给出每条规则在 top-3 / top-20 两种可见性下的**交叉验证**识别率、
  误拒率与 95% CI，以及与基线（生产口径 19.1% / 3.0%）的差值
- 在 `rrf`（生产默认）、`vector_bge`、`rrf_rerank` 三个方案上都跑

### 失败条件与回退

- **没有规则在 CI 意义上显著优于基线** → 报告负结果，生产保持 R0，
  直接进 A2/A3。负结果本身有价值：它证明"只靠检索分数做不了"，
  为 A3 的 LLM 校验提供依据
- **只有 top-20 可见性下才有提升** → A4 改检索接口把候选分数传出

### 本格不做

- 不改 `chat_service`、不改 `assess_context` 的调用方（那是 A4）
- 不训练任何模型（第三步，M5 之后视数据积累而定）

---

## A2 · rerank 可用性（4-5h）

**问题**：rerank 让识别率从 19% 到 59%、MRR@10 从 0.902 到 0.947，
但 CPU P50 9.8 秒，不能默认开启。

### 做什么

1. 测延迟随候选数的变化：rerank 候选 20 / 10 / 5，各自的 P50/P95、
   MRR@10、可答性识别率。cross-encoder 延迟与候选数近似线性，候选从 20 削到 5
   理论上 4 倍提速 —— 测出来才算
2. 把 A1 的评测框架用在 rerank 口径上，得出 rerank 口径的独立阈值
3. 给出启用策略，三选一或组合，由数据决定：
   - 全量开启（若削候选后延迟可接受）
   - **只对灰区开启**：余弦口径判不出时才 rerank（与 A3 竞争同一个灰区）
   - 只在 Agent 路由开启（B3 之后）
4. **检索缓存（M2 T4b）在这里决定**：若 rerank 进入热路径，缓存 rerank 结果
   的收益是秒级的，值得做（连同穿透/击穿/雪崩防御，设计见 M2 计划书）；
   若 rerank 只在少数路径开启，继续不做

### 验收

- 报告 `docs/eval/rerank.md`：候选数 × {延迟, MRR, 识别率} 的表
- 启用策略写进 ADR-006，含"为什么不全量开启/为什么开启"的数字
- 缓存做或不做的结论写进 STATUS，做则另开一格

### 失败条件

- 削到 5 个候选质量崩塌且延迟仍 > 2s → rerank 只用于 A3 的离线对比与
  M5 的 GPU 评估，生产不启用

---

## A3 · 可答性 v2：灰区 LLM 校验（3-4h）

A1 的规则把样本分成三类：明确可答、明确不可答、**拿不准**。只对拿不准
的那部分调一次 LLM："给定这些片段，能否回答这个问题？只回答 是/否/部分"。

### 做什么

- `rag/answerability.py` 增加灰区判定（规则输出三值而非二值）
- LLM 校验 prompt + 解析（结构化输出，解析失败按"不确定"处理，**不默认为可答**）
- 评测：灰区比例（= 额外 LLM 调用比例）、灰区内 LLM 判定的准确率、
  整体识别率与误拒率。抽样人工核验 30 条报一致率（与 M1 生成层评测同一纪律）

### 验收

- 报告给出"每多花 x% 的 LLM 调用，识别率提升 y 个百分点"
- 灰区宽度可配置，报告给出 2-3 个宽度的权衡点

### 失败条件

- LLM 判定与人工一致率 < 75% → 不进生产，结论写进报告

---

## A4 · 生产接入与阈值分口径（2-3h）

- `assess_context` 改为调用 `rag.answerability.decide`，返回值带
  `reason` 与 `signals`
- 配置：`ANSWERABLE_MIN_RELEVANCE` 拆为按口径的两个值（`*_COSINE`、
  `*_RERANK`），旧名保留为 cosine 的别名一个版本；`/config/thresholds`
  同步下发，前端按 `relevance_basis` 取对应阈值（T0a 的单一真相源不变）
- SSE 新增 `answerability` 事件：`{answerable, reason, signals}` —— 拒答时
  前端能说出为什么（对齐"禁止无声降级"）
- `runs` 表记录可答性结论（`route` 之外加一列或写进 `run_steps` 快照，
  B2 前先写 `runs`，需一次迁移）
- 若 A1 结论是 top-20 才有效：`retrieve_advanced` 把候选池分数随
  `stats` 传出，`assess_context` 用候选池而非最终 3 条

### 验收

- 架构测试规则 4 继续通过（service 不 import 阈值）
- 改 `.env` 里任一口径的阈值，前端对应口径的配色随之变
- 拒答时 SSE 带 `reason`，`runs` 表可按结论聚合

---

## A5 · 历史补丁与 BM25 的离线评估（2-3h，可穿插）

两个都是"当初为弥补失效 encoder 而加、换模型后是否还需要"的问题，
以及一个"BM25 到底有没有用"的问题。都能用已有数据离线回答，成本低。

1. **BM25 分层**：T2Ranking 无 query 类型标注，用启发式切出
   "关键词型"子集（含数字、型号、英文 token、专名，如"华为p10和p10plus"），
   在已有 run 记录上分别算 `vector_bge` 与 `rrf` 的 Recall@5。
   子集小时报 CI。**启发式本身是假设**，报告里列出切分规则与样例
2. **jieba 实体注入**：构造多轮追问样本（取同主题 query 对，第二句截短成
   追问形式），比较注入与不注入的 Recall@5
3. **Multi-Query 扩展**：抽 100 条 query，比较开与关的 Recall@5 与延迟
   （要调 LLM，成本有限）

### 验收

- 每项给出"保留 / 删除 / 改为可配置默认关"的结论与数字
- 结论为删除的，在同一格里删代码（删代码比留着"以防万一"更需要证据，
  所以证据就在这一格产出）

---

## B1 · AgentState、节点 runner 与流式桥接（3-4h）

- `AgentState` dataclass；节点签名 `async def node(state, emit) -> str`
  返回下一个节点名，对齐 LangGraph StateGraph 形状（不引入依赖，见 ROADMAP）
- runner：按返回值跳转、全局步数上限、每步计时
- **重做 `_sync_generator_to_async`**：当前每个流式请求一个专属线程 +
  一个阻塞在 `queue.get` 上的线程池 worker。Agent 一次 run 多个节点各自
  流式调用 LLM，会成倍放大。改为生产者线程通过
  `loop.call_soon_threadsafe` 往 `asyncio.Queue` 推数据，消费端纯协程等待，
  不占线程池
- `chat_service.answer_stream` 的现有流程先原样包成一个节点（`direct_rag`），
  行为不变 —— 先换骨架再加能力

### 验收

- 现有 SSE 事件序列逐字节不变（快照测试）
- 并发 20 个流式请求时线程池占用不随请求数增长（测试断言）

## B2 · 轨迹落库、节点事件与取消（2-3h）

- 每个节点结束写一行 `run_steps`（seq、node、ms、state 快照的摘要而非全量）
- SSE 新增 `node_start` / `node_end`（旧事件全部保留）
- 取消：`POST /runs/{trace_id}/cancel` 写 Redis `cancel:{trace_id}`（TTL 10min），
  runner 在节点边界检查；Redis 不可用时取消不可用但 run 正常完成
- `runs.status` 增加 `cancelled` 终态的写入方

### 验收

- 一次 run 的 `run_steps` 按 seq 连续、无重复（唯一约束已在 M2）
- 取消后 run 在当前节点结束时停止，状态为 cancelled

## B3 · 路由（3h）

- 规则优先：问候、超短查询、明确单跳 → `direct_rag`；规则拿不准才调小模型
  做结构化分类（`direct_rag` / `clarify` / `agent`）
- **路由评测集**：从 T2Ranking 抽样 + 手写多跳/计算/需澄清样本，标注期望路由
- 目标：`direct_rag` 占比 ≥ 60%，路由准确率单独报告

## B4 · Tool Protocol + dispatcher + 幂等（3-4h）

- ADR-004 阶段 2 的 `Tool` Protocol：`name`、`args_model`（Pydantic）、
  `side_effects: bool`（默认 True）、`timeout_s`、`async run(args)`
- dispatcher 一处做完横切关注点：入参校验、超时、重试（只对声明幂等的）、
  取消检查、落库、SSE `tool_start`/`tool_end`
- **ADR-003 落地**：迁移给 `tool_calls.idempotency_key` 加 UNIQUE；
  执行模式"先插占位行（ok=NULL）再执行"，冲突读回已有结果；只读工具的跨 run
  结果缓存 `ret:{tool}:{sha1(args)}:{kb_version}`
- 死循环兜底：每 run 工具调用次数上限 + 同一 `(tool, args)` 重复上限

### 验收

- 加一个新工具只需新增一个类并注册，dispatcher 零改动
- 并发两次相同步骤，只执行一次（测试断言）

## B5 · 四个工具（3h）

| 工具 | 要点 |
|---|---|
| `search_knowledge_base` | 走 A2 定下的检索策略；返回带 `relevance_basis` |
| `get_document_context` | 按 `doc_key + seq` 取相邻 chunk（M0 已备好字段） |
| `calculate` | `ast.parse` + 节点白名单，**绝不 eval**；恶意输入单测（`__import__`、超大幂、深嵌套） |
| `web_search` | 默认关闭需授权；SSRF：仅 https、拒私网/回环/链路本地 IP（解析后校验，防 DNS rebinding）、禁跨主机重定向、响应 ≤512KB |

## B6 · Agent 循环与降级链（3-4h）

- `plan → act → evidence_check → (rewrite_query ≤2 次) → generate_with_citations`
- `evidence_check` 节点**直接调用 `rag.answerability`**，不在编排层写判断
- 降级链：主模型 → 重试 1 次（指数退避 + jitter）→ 降级小模型 →
  仍失败则返回已检索证据 + 明确失败说明。**绝不静默返回空**

---

## C1 · citation_verify（3h）

- 答案按标点切句；每个带引用标记的句子，用 cross-encoder 算
  (句子, 被引 chunk) 支持度，低于阈值 `verified=false`
- 写 `citations` 表（M2 已建，`verified` NULL = 未校验，False = 未通过）
- 与 A2 共享 reranker 实例与加载逻辑，不加载第二份模型

## C2 · 引用指标评测（2h）

- **引用准确率**：带引用的句子中，被引 chunk 确实支持该句的比例
- **引用覆盖率**：需要引用的事实句中，带了引用的比例
- 复用 M1 生成层的 50 条样本，人工核验 20 条报一致率

---

## 验收标准（M3 整体）

| # | 标准 | 验证方式 |
|---|---|---|
| M3-C1 | 可答性规则在交叉验证下的识别率与 95% CI，对比生产口径基线 19.1%/3.0% | `docs/eval/answerability.md` |
| M3-C2 | rerank 启用策略由延迟/质量数据决定并写入 ADR | `docs/eval/rerank.md` + ADR-006 |
| M3-C3 | 阈值按口径分开，前端按口径取值 | 改 `.env` 实测 |
| M3-C4 | 拒答带 reason，SSE 与 runs 表可见 | SSE 抓包 + SQL |
| M3-C5 | 一次 Agent run 的 run_steps / tool_calls 完整落库 | SQL |
| M3-C6 | 加新工具 dispatcher 零改动；同一步骤并发只执行一次 | 测试 |
| M3-C7 | `calculate` 拒绝全部恶意输入；`web_search` SSRF 判定单测 | 测试 |
| M3-C8 | 路由准确率与 direct_rag 占比报告 | 路由评测集 |
| M3-C9 | 引用准确率与覆盖率 + 人工一致率 | `docs/eval/citation.md` |
| M3-C10 | 全部测试通过、前端 `vue-tsc` 通过 | 命令输出 |

## 评测纪律（本阶段全部评测共用）

1. **参数不在测试集上调**：选参与计分分离（交叉验证或留出集）
2. **报区间，不只报点估计**：300 条样本上几个百分点的差异需要 CI 才能判断
3. **口径与生产一致**：评测里的判定逻辑调用生产同一个函数，不另写一份
4. **负结果照实写**：规则没赢就写没赢，不换个指标凑出"提升"
5. **LLM 当裁判必须报人工一致率**

## 明确不做

- 训练可答性分类器（需要线上标注数据积累，M5 之后）
- LLMProvider 端口与 fake provider（M5）
- 面板改造（M4）—— 本阶段只保证事件与数据齐全
- 引入 LangGraph / LangChain（ROADMAP 已定）

## 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| 无答案集是**合成**的（剔除 gold + 随机干扰） | 真实知识库里的无答案情形，残留文档往往与问题**同主题**，比随机干扰更难；本阶段的识别率可能高估线上表现 | 报告显式标注；A3 的人工核验抽样包含同主题难例 |
| 每条 query 平均 ~5 条 gold | 真实知识库常常只有 1 个相关 chunk，分数形态不同 | 报告按"top-20 中 gold 数"分层给结果 |
| A1 规则全部不显著 | 第一格没有生产收益 | 已在失败条件里定义回退：负结果 + 进 A3 |
| rerank 削候选后仍太慢 | A2 无法进生产 | 已在 A2 失败条件定义 |
