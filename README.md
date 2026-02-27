# DeepBlue Intelligence — 生产级 RAG 系统

> 基于 **检索增强生成（RAG）** 的智能问答系统。上传你的文档，系统自动建立知识库，用自然语言提问即可得到精准的、带来源引用的回答。

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Vue 3](https://img.shields.io/badge/Vue-3-brightgreen)](https://vuejs.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速部署（Docker）](#快速部署docker)
- [本地开发](#本地开发)
- [配置参考](#配置参考)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 功能特性

### 核心 RAG 能力

| 功能 | 说明 |
|------|------|
| **混合检索** | 向量语义检索（70%）+ BM25 关键词匹配（30%）融合，中英文均优 |
| **中文分词** | 集成 jieba，BM25 检索准确识别中文词语边界 |
| **引用追踪** | 每条答案自动标注来源文档和段落，支持点击跳转 |
| **多轮对话** | 上下文感知，代词自动消解（"它"→"上一个文档"） |
| **流式输出** | SSE 逐字推送，无需等待完整响应 |

### 系统工程

| 功能 | 说明 |
|------|------|
| **多 LLM 支持** | 智谱 GLM-4 / OpenAI GPT / 通义千问 / Claude，配置一行切换 |
| **实时日志** | 前端 BrainPanel 实时显示 RAG 每一步：`[USER]→[EMBED]→[SEARCH]→[HIT]→[GEN]` |
| **速率限制** | 30 次/分钟/IP，防止滥用 |
| **健康检查** | Docker 容器自动健康检查，前端等待后端就绪后再启动 |
| **持久化存储** | ChromaDB + bind mount，容器重启数据不丢失 |

### 前端界面（三栏布局）

```
┌─────────────────┬───────────────────┬──────────────────┐
│  知识库管理      │    对话界面        │   RAG 日志        │
│                 │                   │                  │
│ 📄 上传文档     │ 💬 流式问答        │ ▶ 实时过程日志   │
│ 📋 文档列表     │ 🔗 引用来源展示    │ 📊 检索命中情况  │
│ 🗑️ 删除文档    │ 🔄 RAG/直答切换   │ 🔍 Prompt 预览   │
└─────────────────┴───────────────────┴──────────────────┘
```

---

## 系统架构

```
用户浏览器
    │  HTTP (port 80)
    ▼
┌─────────────────────────────────────┐
│  Nginx（前端容器内）                 │
│  ├─ / → Vue 3 SPA                   │
│  └─ /api/ → 反向代理到后端           │
└───────────────┬─────────────────────┘
                │ 内网 Docker rag-network
                ▼
┌─────────────────────────────────────┐
│  FastAPI 后端（port 8000）           │
│                                     │
│  POST /api/v1/chat/stream  (SSE)    │
│  POST /api/v1/documents/upload      │
│  GET  /api/v1/documents             │
│  GET  /health                       │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│  RAG 核心                           │
│  ┌──────────┐  ┌──────────────────┐ │
│  │ 向量检索  │  │   BM25 检索      │ │
│  │ChromaDB  │  │   jieba 分词     │ │
│  └──────┬───┘  └────────┬─────────┘ │
│         └──────┬─────────┘          │
│                ▼                    │
│         混合评分融合                  │
│                ▼                    │
│         LLM 生成（带引用）            │
└─────────────────────────────────────┘
```

---

## 快速部署（Docker）

### 前置条件

- 一台服务器（或本地机器），已安装 Docker 和 Docker Compose
- 至少 2GB 内存（Embedding 模型需要）
- LLM API Key（推荐智谱AI，申请地址：https://open.bigmodel.cn/）

### 第一步：获取代码

```bash
git clone <your-repo-url>
cd rag-project
```

### 第二步：配置环境变量

```bash
# 创建配置文件（从示例复制）
cp .env.example .env

# 用任意编辑器打开，填入你的 API Key
nano .env   # Linux/Mac
notepad .env  # Windows
```

**最少只需配置这几行：**

```env
# 选择 LLM 提供商
LLM_PROVIDER=zhipu

# 填入对应的 API Key
ZHIPU_API_KEY=你的智谱API密钥
ZHIPU_MODEL=glm-4
```

**如果使用 OpenAI：**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-你的OpenAI密钥
OPENAI_MODEL=gpt-3.5-turbo
```

**生产部署额外建议配置：**
```env
# 限制允许访问的域名（留空则允许所有来源）
ALLOWED_ORIGINS=http://你的服务器IP,https://你的域名.com
```

### 第三步：启动服务

```bash
docker compose up -d --build
```

首次启动需要下载 Embedding 模型（约 90MB），请耐心等待 1-2 分钟。

**查看启动日志：**
```bash
docker compose logs -f
```

当你看到以下输出时，系统已就绪：
```
deepblue-backend  | ============================================================
deepblue-backend  | DeepBlue Intelligence API starting...
deepblue-backend  | ============================================================
```

### 第四步：访问系统

| 地址 | 说明 |
|------|------|
| `http://服务器IP` | 主界面（给同学发这个地址） |
| `http://服务器IP:8000/docs` | 后端 API 文档（Swagger UI） |

### 常用运维命令

```bash
# 查看运行状态
docker compose ps

# 查看后端日志
docker compose logs backend -f

# 重启服务
docker compose restart

# 停止服务（数据不丢失）
docker compose down

# 彻底清除（包括数据！谨慎使用）
docker compose down -v
rm -rf chroma_db/
```

---

## 本地开发

### 后端

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# 2. 安装依赖
pip install -r requirements.txt
# 国内用户：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 4. 启动后端（支持热重载）
python -m uvicorn backend.main:app --reload --port 8000

# 访问 API 文档：http://localhost:8000/docs
```

### 前端

```bash
cd frontend-vue

# 安装依赖
npm install

# 启动开发服务器（自动代理 /api 到 localhost:8000）
npm run dev

# 访问 http://localhost:3000
```

### 向量数据库初始化

第一次运行前，可以批量导入文档：

```bash
# 将文档放入目录（支持 .md .txt）
mkdir -p data/documents/my-docs
cp your-docs/*.md data/documents/my-docs/

# 批量导入（Python 脚本）
python -c "
from rag import DocumentIngestion
ing = DocumentIngestion()
stats = ing.ingest_directory()
print(f'导入完成：{stats[\"total_chunks\"]} 个文本块')
"
```

---

## 配置参考

所有配置通过 `.env` 文件控制，无需修改代码。

### LLM 提供商

```env
LLM_PROVIDER=zhipu      # 可选：openai / anthropic / zhipu / qwen

# 智谱AI（推荐，国内访问快）
ZHIPU_API_KEY=your-key
ZHIPU_MODEL=glm-4

# OpenAI
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_API_BASE=https://api.openai.com/v1   # 可改为代理地址

# 通义千问
QWEN_API_KEY=your-key
QWEN_MODEL=qwen-turbo
```

### 检索参数

```env
TOP_K_RESULTS=3           # 最终返回给 LLM 的文档块数（推荐 3，噪声最低）
SIMILARITY_THRESHOLD=0.7  # 相似度阈值（0-1，越高越严格）
ENABLE_HYBRID=true        # 混合检索开关（推荐开启）
BM25_WEIGHT=0.3           # BM25 关键词权重
VECTOR_WEIGHT=0.7         # 向量语义权重
HYBRID_TOP_K=20           # 混合检索候选数量
```

### 生成参数

```env
LLM_TEMPERATURE=0.5       # 温度（RAG 场景推荐 0.3-0.5，减少幻觉）
LLM_MAX_TOKENS=800        # 最大输出 Token 数
LLM_TIMEOUT=30            # API 超时（秒）
```

### 文本分块

```env
CHUNK_SIZE=500            # 每块字符数（中文文档建议 300-500）
CHUNK_OVERLAP=100         # 块间重叠字符数（推荐 CHUNK_SIZE 的 20%）
```

### 部署安全

```env
# 限制访问来源（留空则允许所有，不限来源）
ALLOWED_ORIGINS=http://your-server.com,https://your-domain.com

# 日志级别
LOG_LEVEL=INFO            # DEBUG / INFO / WARNING / ERROR
```

---

## API 文档

启动后访问 `http://localhost:8000/docs` 查看完整的 Swagger UI 文档。

### 核心接口

#### 流式问答（SSE）

```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "question": "文档里关于退款政策是怎么说的？",
  "session_id": "user-abc123",
  "use_retrieval": true,
  "enable_citation": true
}
```

返回 SSE 事件流，事件类型：

| 事件 | 含义 | 数据 |
|------|------|------|
| `connected` | 连接建立 | `session_id` |
| `resolved` | 指代消解完成 | 解析后的问题 |
| `retrieval_status` | 开始检索 | 状态描述 |
| `retrieval_done` | 检索完成 | 命中文档数 |
| `answer_chunk` | 答案片段 | 文字内容 |
| `citations` | 引用来源 | 文档列表+评分 |
| `done` | 完成 | — |
| `error` | 错误 | 错误信息 |

#### 上传文档（流式进度）

```http
POST /api/v1/documents/upload/stream
Content-Type: multipart/form-data

files: [file1.md, file2.txt]
```

返回 SSE 事件流，包含解析→分块→向量化→存储的全过程进度。

#### 文档管理

```http
GET    /api/v1/documents               # 获取文档列表
DELETE /api/v1/documents/{filename}    # 删除文档
GET    /api/v1/stats                   # 系统统计
GET    /health                         # 健康检查
```

---

## 项目结构

```
rag-project/
│
├── 📂 backend/                   # FastAPI Web 层（六边形架构）
│   ├── main.py                   # 应用入口、中间件注册
│   ├── settings.py               # 后端配置（CORS、端口等）
│   ├── rate_limit.py             # 速率限制中间件（滑动窗口）
│   ├── env_validation.py         # 启动时验证环境变量
│   ├── 📂 api/
│   │   ├── routes.py             # REST 接口（历史记录、统计）
│   │   ├── sse.py                # SSE 流式接口（对话主入口）
│   │   └── upload.py             # 文档上传接口
│   ├── 📂 schemas/
│   │   └── __init__.py           # Pydantic v2 请求/响应模型
│   ├── 📂 services/
│   │   └── chat_service.py       # RAG 编排（会话管理 + 流式生成）
│   └── 📂 adapters/
│       └── streaming_ingestion.py # 上传进度流式适配器
│
├── 📂 rag/                       # RAG 核心业务逻辑
│   ├── __init__.py               # 公开 API 导出
│   ├── chunker.py                # 文本分块（结构感知 + 中文友好）
│   ├── embedder.py               # 向量化封装（all-MiniLM-L6-v2）
│   ├── vectordb.py               # ChromaDB CRUD 抽象
│   ├── retriever.py              # 高级检索（向量 + BM25 混合 + Rerank）
│   ├── ingestion.py              # 文档摄入流水线
│   ├── llm.py                    # 多 LLM 客户端（4个提供商）
│   ├── citation.py               # 引用提取与追踪
│   ├── conversation.py           # 多轮对话历史管理
│   └── logger.py                 # 统一日志配置
│
├── 📂 frontend-vue/              # Vue 3 前端
│   ├── src/
│   │   ├── App.vue               # 根组件（三栏布局）
│   │   ├── 📂 components/
│   │   │   ├── LibraryPanel.vue  # 知识库管理面板
│   │   │   ├── ChatPanel.vue     # 对话面板（流式 + 引用）
│   │   │   └── BrainPanel.vue    # RAG 过程可视化面板
│   │   ├── 📂 composables/
│   │   │   ├── useChat.ts        # 对话状态与 SSE 事件处理
│   │   │   ├── useDocuments.ts   # 文档 CRUD 状态
│   │   │   └── useSSEStream.ts   # SSE 流状态跟踪
│   │   ├── 📂 services/
│   │   │   └── api.ts            # 后端 API 客户端（SSE + REST）
│   │   └── 📂 types/
│   │       └── index.ts          # TypeScript 类型定义
│   ├── nginx.conf                # Nginx 反向代理（/api → backend）
│   └── Dockerfile.frontend       # 前端 Docker 镜像（多阶段构建）
│
├── config.py                     # 统一配置中心（从 .env 读取所有参数）
├── requirements.txt              # Python 依赖
├── Dockerfile.backend            # 后端 Docker 镜像（多阶段构建）
├── docker-compose.yml            # 服务编排（backend + frontend + 网络 + 卷）
├── .dockerignore                 # Docker 构建排除文件
└── .env.example                  # 环境变量模板（复制为 .env 后编辑）
```

---

## 常见问题

**Q: 第一次启动很慢？**
A: 首次启动会从 HuggingFace 下载 Embedding 模型（all-MiniLM-L6-v2，约 90MB）。下载完成后缓存到 `~/.cache/`，后续启动秒级完成。如果下载失败，可以手动设置镜像：
```env
HF_ENDPOINT=https://hf-mirror.com
```

**Q: 上传文档后问答结果没变化？**
A: 检查两点：① 文件格式是否为 `.md` 或 `.txt`；② 文件大小是否超过 10MB 上限。查看后端日志确认是否成功摄入。

**Q: 回答"我不知道"或"找不到相关信息"？**
A: 说明检索到的文档相似度低于阈值（0.7）。可以降低阈值（`SIMILARITY_THRESHOLD=0.5`），或检查文档内容与问题是否确实相关。

**Q: 如何切换 LLM 不重新构建镜像？**
A: 修改 `.env` 文件中的 `LLM_PROVIDER` 和对应 API Key，然后 `docker compose restart backend` 即可（不需要重新 build）。

**Q: 多人同时使用会混用会话吗？**
A: 不会。前端会为每个用户生成唯一的 `session_id`（浏览器本地存储），会话隔离。

**Q: 数据持久化了吗？容器重启会丢失数据吗？**
A: 不会丢失。向量数据库存在 `./chroma_db/`，文档存在 `./data/documents/`，都是宿主机目录的 bind mount，容器重启不影响。

**Q: 如何备份数据？**
A:
```bash
# 备份向量数据库
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz chroma_db/

# 备份文档
tar -czf docs_backup_$(date +%Y%m%d).tar.gz data/documents/
```

---

## 开发者说明

### 添加新的 LLM 提供商

在 `rag/llm.py` 中的 `LLMClient` 类添加新分支，参考现有 zhipu/openai 实现即可。

### 修改检索策略

`rag/retriever.py` 中的 `retrieve_advanced()` 是检索入口，各功能通过参数控制：
- `enable_hybrid=True` → 混合检索（当前默认）
- `enable_rerank=True` → Cross-Encoder 精排（需下载 bge-reranker-base 模型）
- `enable_multi_query=True` → 多查询扩展（与 hybrid 互斥）

### 自定义系统 Prompt

在 `config.py` 中修改 `SYSTEM_PROMPT_HYBRID`（当前默认使用），支持切换为 `STRICT`（纯文档）、`FRIENDLY`（亲切风格）等模式。
