# RAG 项目 - 检索增强生成系统

一个生产级的 RAG (Retrieval-Augmented Generation) 系统，支持文档摄入、语义检索和 LLM 问答。

## 项目特点

✨ **模块化设计** - 清晰的职责分离，易于维护和扩展
🎯 **配置集中** - 所有配置在 `config.py` 统一管理
🔌 **多 LLM 支持** - 支持 OpenAI、Anthropic、智谱AI、通义千问
📦 **向量数据库** - 使用 ChromaDB 进行持久化存储
🚀 **简单易用** - 提供命令行界面和交互式问答

---

## 快速开始

### 1. 安装依赖

```bash
# 基础安装
pip install -r requirements.txt

# 国内用户可以使用镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
# Windows: notepad .env
# Linux/Mac: nano .env
```

**.env 示例：**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

### 3. 准备文档

将文档放入 `data/documents/` 目录，按类别组织：

```
data/documents/
├── policies/           # 政策类文档
│   ├── pet_policy.md
│   └── remote_work.md
└── benefits/           # 福利类文档
    └── health_insurance.md
```

### 4. 运行系统

```bash
# 初始化数据库
python main.py init

# 摄入文档
python main.py ingest

# 测试检索
python main.py query

# 运行完整 RAG 系统
python main.py run
```

---

## 项目结构

```
rag-project/
├── config.py              # 配置文件（唯一真相源）
├── requirements.txt       # 依赖管理
├── .env.example          # 环境变量模板
├── README.md             # 本文档
├── main.py               # 主程序入口
│
├── data/                 # 数据目录
│   └── documents/        # 待处理文档
│
├── chroma_db/            # 向量数据库存储
│
├── rag/                  # 核心模块（所有业务逻辑）
│   ├── __init__.py
│   ├── chunker.py        # 文本分块
│   ├── embedder.py       # 向量化
│   ├── vectordb.py       # 向量数据库操作（只做连接&CRUD）
│   ├── retriever.py      # 检索逻辑
│   ├── ingestion.py      # 文档摄入逻辑
│   └── llm.py           # LLM 调用
│
└── scripts/              # 脚本（调用 rag 模块，不写逻辑）
    ├── 1_init_db.py      # 初始化数据库
    ├── 2_ingest_docs.py  # 摄入文档
    ├── 3_test_query.py   # 测试检索
    └── 4_run_rag.py      # 完整 RAG 流程
```

---

## 设计原则

### 1. ✅ `vectordb.py` 只做连接 & 基础操作
- 封装 ChromaDB 的 CRUD 接口
- 不包含业务逻辑
- 单一职责：数据库交互

### 2. ✅ `scripts` 调用模块，不写逻辑
- 脚本只是入口，调用 `rag` 模块的功能
- 所有业务逻辑在 `rag/` 模块中
- 保持脚本简洁易读

### 3. ✅ `config.py` 是唯一真相源
- 所有配置集中管理
- 避免硬编码
- 提供配置验证功能

---

## 使用指南

### 命令行界面

```bash
# 查看帮助
python main.py --help

# 检查配置
python main.py config

# 初始化数据库（可选重置）
python main.py init

# 摄入文档（批量处理）
python main.py ingest

# 测试检索功能
python main.py query

# 运行 RAG 问答系统
python main.py run
```

### 直接运行脚本

```bash
# 或者直接运行具体脚本
python scripts/1_init_db.py
python scripts/2_ingest_docs.py
python scripts/3_test_query.py
python scripts/4_run_rag.py
```

### 交互式问答

运行 `python main.py run` 后进入交互模式：

```
你的问题 > 可以带宠物来公司吗？

正在检索相关文档...
✓ 找到 3 个相关文档:
  1. policies/pet_policy.md
  2. policies/remote_work.md
  3. benefits/health_insurance.md

正在生成答案...

======================================================================
回答:
======================================================================
根据 TechCorp 的宠物政策，员工可以在每周五带宠物来办公室。
但需要注意：
1. 宠物必须性格温顺
2. 宠物需要已接种疫苗
3. 在公共区域需要牵引

CEO 的金毛寻回犬是公司吉祥物，经常在办公室出没。
======================================================================
```

---

## 核心功能详解

### 1. 文本分块 (Chunking)

```python
from rag import chunk_text

text = "长文本内容..."
chunks = chunk_text(text, size=500, overlap=100)
# 生成多个有重叠的文本块，保持语义完整性
```

**特点**：
- 滑动窗口策略
- 可配置块大小和重叠
- 支持按句子边界分块

### 2. 向量化 (Embedding)

```python
from rag import Embedder

embedder = Embedder()
embedding = embedder.encode("文本内容")
# 转换为 384 维向量
```

**特点**：
- 使用 Sentence-BERT 模型
- 支持批量处理
- 捕捉语义信息

### 3. 向量数据库

```python
from rag import VectorDB

db = VectorDB()
db.add(ids=[...], embeddings=[...], documents=[...])
results = db.query(query_embeddings=[...], n_results=3)
```

**特点**：
- 持久化存储
- 余弦相似度检索
- 支持元数据过滤

### 4. 语义检索

```python
from rag import Retriever

retriever = Retriever()
results = retriever.retrieve("用户问题", top_k=3)
# 返回最相关的文档块
```

**特点**：
- 理解语义而非关键词
- 返回最相关的 K 个结果
- 支持类别过滤

### 5. LLM 调用

```python
from rag import LLMClient

llm = LLMClient()
answer = llm.answer_with_context(question, context)
# 基于检索到的上下文生成答案
```

**支持的提供商**：
- OpenAI (GPT-3.5, GPT-4)
- Anthropic (Claude)
- 智谱AI (GLM-4)
- 通义千问 (Qwen)

---

## 配置说明

所有配置在 `config.py` 中管理，主要包括：

### 分块配置
```python
CHUNK_SIZE = 500        # 每块字符数
CHUNK_OVERLAP = 100     # 重叠字符数
```

### Embedding 配置
```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
```

### 检索配置
```python
TOP_K_RESULTS = 3           # 返回结果数量
MIN_RELEVANCE_SCORE = 0.3   # 相关性阈值
```

### LLM 配置
```python
LLM_PROVIDER = "openai"      # 提供商
LLM_TEMPERATURE = 0.7        # 温度参数
LLM_MAX_TOKENS = 500         # 最大 token 数
```

---

## 常见问题

### 1. 如何切换 LLM 提供商？

修改 `.env` 文件：
```env
LLM_PROVIDER=anthropic  # 改为 anthropic
ANTHROPIC_API_KEY=sk-ant-xxx
```

### 2. 如何调整分块策略？

修改 `config.py`：
```python
CHUNK_SIZE = 1000      # 增大块大小
CHUNK_OVERLAP = 200    # 增大重叠
```

### 3. 如何添加自定义文档类型？

修改 `config.py`：
```python
SUPPORTED_FILE_TYPES = [".md", ".txt", ".pdf", ".docx"]
```

然后在 `rag/ingestion.py` 中添加对应的解析逻辑。

### 4. 数据库在哪里？

向量数据库存储在 `chroma_db/` 目录，可以安全删除重建。

### 5. 如何重置数据库？

```bash
# 方法1：删除数据库目录
rm -rf chroma_db/

# 方法2：运行初始化脚本时选择重置
python main.py init
# 然后输入 'y' 确认重置
```

---

## 性能优化

### 1. 使用 GPU 加速

```bash
# 安装 GPU 版本的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 在 config.py 中启用
USE_GPU = True
```

### 2. 批量处理

```python
BATCH_SIZE = 64  # 增大批处理大小
```

### 3. 使用更快的向量库

```bash
pip install faiss-cpu
```

---

## 扩展开发

### 添加新的 LLM 提供商

1. 在 `config.py` 添加配置
2. 在 `rag/llm.py` 的 `LLMClient` 类中添加对应方法
3. 实现 `_init_client()` 和 `_generate_xxx()` 方法

### 添加新的检索策略

1. 在 `rag/retriever.py` 中添加新方法
2. 实现自定义的相似度计算或排序逻辑

### 添加文档预处理

1. 在 `rag/ingestion.py` 中扩展 `DocumentIngestion` 类
2. 添加自定义的文本清洗或格式化逻辑

---

## 许可证

MIT License

---

## 致谢

本项目使用了以下开源项目：
- [ChromaDB](https://www.trychroma.com/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 文本 Embedding
- [OpenAI](https://openai.com/) - LLM API

---

## 联系方式

如有问题或建议，欢迎提 Issue 或 Pull Request。

**快乐编码！🚀**
