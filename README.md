# DeepBlue Brain

**可追溯的 RAG 知识库研究平台** —— 全链路白盒化，且每个技术决策都有评测数据支撑。

在线体验: [http://deepblue-brain.cn/](http://deepblue-brain.cn/)

## 解决什么问题

做 RAG 的人最大的困难不是搭起来，而是**出了问题不知道该改哪里**：
答案不对，是文档没切好、embedding 选错、检索召不回、阈值太严，还是 prompt 写坏了？

多数知识库产品把这条链路做成黑盒 —— 你只看到答案，看不到它怎么来的。
本项目把整条链路摊开，并且做了两件产品化工具通常不做的事：

1. **展示的每个数值都有明确口径**。相关性分数只从
   [`rag/scoring.py`](rag/scoring.py) 取，标注了它是余弦相似度还是
   cross-encoder 的相关概率。不展示混用量纲的"综合得分"。
2. **技术选型用公开人工精标数据集验证，包括结论为负的**。
   见 [评测报告](docs/eval/report.md)。

## 与知识库产品（ima、Notion AI 等）的区别

它们的目标是**帮你用知识库**，用户要答案，过程是黑盒。
本项目的目标是**让 RAG 系统本身可被观察和验证**，用户是开发者与研究者。

| | 知识库产品 | 本项目 |
|---|---|---|
| 检索路径 | 不可见 | 展示走了向量/BM25/融合，各路排名与分数 |
| 相关性分数 | 不展示，或只给"高/中/低" | 数值 + 口径标注 + 计算依据说明 |
| 方案选择 | 厂商定好 | 5 种方案的 Recall@5 / MRR / nDCG / 延迟对比 |
| 阈值 | 不可见 | 校准曲线，并说明单一阈值为何不足 |
| 引用 | 可点击查看原文 | 可点击 + 相关性 + 引用校验（M3） |
| 负面结果 | 不会公开 | 报告中如实记录 |

## 已验证的结论

这些不是设计意图，是实测数据（T2Ranking，300 query / 13,536 段语料）：

**换中文 embedding 模型使 Recall@5 从 0.023 提升到 0.707（约 30 倍）。**
`all-MiniLM-L6-v2` 是英文模型，在中文语料上 MRR@10 仅 0.058 —— 等同随机排序。
唯一变量是模型：同一份语料、同一套分块、同样的召回数。

**RRF 混合检索在该数据集上没有增益**（0.708 vs 0.707），延迟反而涨 3.5 倍。
原因是 T2Ranking 的 query 以语义匹配为主，BM25 单独只有 0.586，
对已达 0.707 的向量路提供不了补充。这个负面结果限定了 RRF 的适用条件：
它的收益来自两路互补性，需要 BM25 的是编号、型号、专有名词类查询。

**判断"知识库里没有答案"，相对分数比绝对分数有效。**
有答案与无答案的 top1 分数分布严重重叠 —— 无答案样本的最小值（0.714）
甚至高于有答案样本的最小值（0.697）：bge 余弦的绝对值挤在一个窄区间里。
单一阈值在误拒率 10% 时能识别约 59% 的无答案查询，但阈值每动 0.01，
识别率变化约 12 个百分点，非常脆弱。改看"最相关片段比其余候选高出多少"，
识别率升到 70.6%（5 折交叉验证，相对单阈值 +11.7 [+5.7, +18.3]），
与开启 rerank（CPU 上慢约 190 倍）统计上无差别。
详见 [可答性评测](docs/eval/answerability.md)。

## 适合做什么

- 排查 RAG 项目的问题出在哪一环
- 对比检索方案，用自己的数据集复现这套评测
- 理解向量检索、BM25、RRF、rerank、citation 之间的关系
- 作品集中展示可解释性与工程严谨度

## 你能看到什么

### 实时检索日志流

右侧运行面板通过 SSE 持续展示后台事件。你不需要等整段回答结束再猜发生了什么，页面会实时告诉你：

- 问题如何进入会话
- 是否触发了指代消解
- 检索走的是向量、BM25，还是混合模式
- 命中了哪些文档，分数如何
- LLM 什么时候开始生成
- 上传文档时，解析、分块、Embedding、入库各走到哪一步

这一条“实时检索日志流”是项目最核心的白盒化入口。

### 提示词原文溯源

每次回答结束后，你都可以展开 `Prompt Inspector`，直接看到后端实际送入模型的完整 Prompt，包括：

- 用户原始问题
- 指代消解后的问题
- 被拼入上下文的原始文档片段
- 引用要求
- 对话历史上下文

这会让“模型为什么这么答”从猜测变成可验证的事实。

### 原文证据卡片

回答中的来源标记可以点击。点击后会弹出对应的原文片段，展示：

- 来源文件名
- 引用顺序
- 原始 chunk 内容
- chunk ID

这样一来，答案、引用、检索命中、原始证据就连成了一条完整链路。

### 流式文档摄入

上传文档不是一个静默等待的黑盒。你会看到文档从接收、解析、分块、Embedding 到入库的全过程。支持的文件格式包括：

- `.md`
- `.txt`
- `.pdf`
- `.docx`

## 功能一览

- 文档上传、删除、列表查看和 chunk 展示
- RAG 模式与通用知识模式切换
- SSE 流式问答
- SSE 流式上传进度
- 多轮对话与简单指代消解
- 引用追踪与原文弹窗
- Prompt Inspector
- 实时检索质量与相似度展示
- FastAPI Swagger 接口文档
- Docker Compose 启动

## 技术栈

前端

- Vue 3
- TypeScript
- Vite
- Tailwind CSS
- lucide-vue-next

后端

- FastAPI
- Pydantic v2
- StreamingResponse

RAG 核心

- ChromaDB
- sentence-transformers
- rank-bm25
- jieba

文档解析

- Markdown / TXT
- pypdf
- python-docx

LLM

- OpenAI
- Anthropic Claude
- 智谱 GLM
- 通义千问

部署

- Docker
- Docker Compose
- Nginx

## 架构速览

```text
Browser
  |
  |  Vue 3 SPA
  v
frontend-vue/
  - LibraryPanel: 上传、删除、查看文档 chunk
  - ChatPanel: 流式对话、引用标记、原文证据卡片
  - BrainPanel: 实时日志流、相似度仪表、Prompt Inspector
  |
  |  /api/*
  v
backend/
  - main.py: FastAPI 应用入口
  - api/sse.py: 对话 SSE 接口
  - api/upload.py: 文档上传接口
  - api/routes.py: 文档列表、删除、统计、非流式对话
  - services/chat_service.py: 会话、检索、生成编排
  - adapters/streaming_ingestion.py: 文档摄入事件流
  |
  v
rag/
  - ingestion.py: 文档读取、分块、入库
  - chunker.py: Markdown/段落感知分块
  - embedder.py: SentenceTransformer 封装
  - vectordb.py: ChromaDB 访问层
  - retriever.py: 向量检索、BM25、Hybrid、Rerank
  - citation.py: 引用 Prompt 与解析
  - llm.py: 多 LLM 提供商封装
  - conversation.py: 多轮对话和指代消解
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/xiyue188/RAG.git
cd RAG
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

推荐先用智谱作为默认演示模型：

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your-zhipu-api-key
ZHIPU_MODEL=glm-4-flash-250414
```

如果你想切换到 OpenAI，可以这样写：

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4.1-mini
```

### 3. 本地开发启动

后端：

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

前端：

```bash
cd frontend-vue
npm install
npm run dev
```

访问地址：

- 前端: `http://localhost:3000`
- 后端接口文档: `http://localhost:8000/docs`
- 健康检查: `http://localhost:8000/health`

### 4. Docker 启动

```bash
docker compose up -d --build
```

启动后访问：

- 前端: `http://localhost`
- 后端接口文档: `http://localhost:8000/docs`
- 健康检查: `http://localhost:8000/health`

## 推荐体验路径

1. 打开页面。
2. 在左侧知识库上传文档。
3. 等待右侧日志流显示解析、分块、Embedding、入库完成。
4. 在中间对话框发问。
5. 观察右侧实时检索日志流，确认命中文档和生成状态。
6. 点击回答中的来源标记，查看原文证据卡片。
7. 展开 Prompt Inspector，查看真实发送给模型的提示词。

## 配置说明

常用配置都集中在 `.env` 中，由 `config.py` 统一读取。

```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your-key
ZHIPU_MODEL=glm-4-flash-250414

OPENAI_MODEL=gpt-4.1-mini
ANTHROPIC_MODEL=claude-sonnet-4-20250514
QWEN_MODEL=qwen3.6-plus

TOP_K_RESULTS=3
SIMILARITY_THRESHOLD=0.7
ENABLE_HYBRID=true
BM25_WEIGHT=0.3
VECTOR_WEIGHT=0.7
HYBRID_TOP_K=20

CHUNK_SIZE=500
CHUNK_OVERLAP=100

LOG_LEVEL=INFO
ALLOWED_ORIGINS=http://localhost,http://localhost:3000
```

### 支持的文档格式

| 格式 | 解析方式 | 限制 |
|---|---|---|
| `.md` | 直接读取 | 按标题层级切分，效果最好 |
| `.txt` | 直接读取 | 无结构信息，按段落切分 |
| `.pdf` | pypdf 文本抽取 | **扫描件（图片型 PDF）抽不出内容**，需要 OCR |
| `.docx` | python-docx | 只取段落文本，表格与图片中的文字会丢失 |

单文件上限 10 MB。

`.md` 效果最好的原因是分块器能利用标题层级：每个 chunk 携带
`h1/h2/h3` 上下文，检索命中后能知道它在文档结构中的位置。
纯文本只能按段落切，语义边界更模糊。

### 常见补充项

- `HF_ENDPOINT=https://hf-mirror.com`，适合国内环境下载模型较慢时使用。
- `.env` 不要提交到 Git。
- `chroma_db/`、`logs/`、`frontend-vue/dist/`、`node_modules/`、`venv/` 都是本地运行或构建产物。

## 主要接口

- `POST /api/v1/chat/stream`: SSE 流式问答。
- `GET /api/v1/chat/stream-get`: 兼容 EventSource 的流式问答。
- `POST /api/v1/chat/message`: 非流式问答。
- `POST /api/v1/documents/upload/stream`: SSE 流式文档上传。
- `POST /api/v1/documents/upload`: 同步文档上传。
- `GET /api/v1/documents?include_chunks=true`: 查看文档与 chunk 列表。
- `DELETE /api/v1/documents/{file_name}`: 删除文档。
- `GET /api/v1/stats`: 查看系统统计。
- `GET /health`: 健康检查。


## 目录说明

```text
rag-project/
  backend/              FastAPI Web 层
  rag/                  RAG 核心逻辑
  frontend-vue/         Vue 前端
  data/documents/       示例/本地文档目录
  chroma_db/            本地 ChromaDB 数据目录
  scripts/              调试和验证脚本
  config.py             配置入口
  requirements.txt      Python 依赖
  docker-compose.yml    Docker 编排
  Dockerfile.backend    后端镜像
```


## License

MIT
