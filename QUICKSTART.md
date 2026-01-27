# 🚀 RAG 项目快速开始指南

## 一、环境准备（5分钟）

### 1. 检查 Python 版本
```bash
python --version
# 需要 Python 3.8+
```

### 2. 安装依赖
```bash
cd rag-project
pip install -r requirements.txt
```

国内用户建议使用镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 二、配置 API（3分钟）

### 1. 复制环境变量模板
```bash
cp .env.example .env
```

### 2. 编辑 .env 文件
```bash
# Windows
notepad .env

# Linux/Mac
nano .env
```

### 3. 填入 API Key

**选项A: 使用 OpenAI（推荐）**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

**选项B: 使用智谱AI（国内推荐）**
```env
LLM_PROVIDER=zhipu
ZHIPU_API_KEY=your-zhipu-key-here
ZHIPU_MODEL=glm-4
```

**API Key 获取地址**：
- OpenAI: https://platform.openai.com/api-keys
- 智谱AI: https://open.bigmodel.cn/
- 通义千问: https://dashscope.console.aliyun.com/

---

## 三、运行系统（4步搞定）

### 步骤1: 检查配置
```bash
python main.py config
```

应该看到：
```
✓ 数据目录: 存在
✓ 数据库目录: 存在
✓ 配置验证通过
```

### 步骤2: 初始化数据库
```bash
python main.py init
```

第一次运行选择 `N`（不重置）

### 步骤3: 摄入文档
```bash
python main.py ingest
```

应该看到：
```
处理类别: policies
  pet_policy.md (3 chunks) ✓
  remote_work.md (4 chunks) ✓

处理类别: benefits
  health_insurance.md (6 chunks) ✓

摄入完成！
  • 处理文件数: 3
  • 生成块数: 13
```

### 步骤4: 运行 RAG 系统
```bash
python main.py run
```

现在可以开始提问了！

---

## 四、测试问题

尝试问以下问题：

```
你的问题 > 可以带宠物来公司吗？

你的问题 > 远程办公有什么限制？

你的问题 > 公司的健康保险包括哪些？

你的问题 > 401k 匹配比例是多少？

你的问题 > 入职满一年有多少天年假？
```

---

## 五、常见问题

### Q1: 提示 "数据库为空"
**A**: 运行 `python main.py ingest` 摄入文档

### Q2: API 调用失败
**A**: 检查：
1. .env 文件是否存在
2. API Key 是否正确
3. 网络是否能访问 API（可能需要代理）

### Q3: 模块导入错误
**A**: 确保在 `rag-project/` 目录下运行命令

### Q4: 编码错误（Windows）
**A**: 确保文件编码为 UTF-8

### Q5: 下载模型很慢
**A**: 第一次运行会下载 Embedding 模型（约90MB），请耐心等待。国内用户可以：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 六、项目结构一览

```
rag-project/
├── config.py          ← 所有配置在这里
├── .env              ← API Key 在这里
├── main.py           ← 主程序入口
├── requirements.txt  ← 依赖列表
│
├── data/documents/   ← 放你的文档到这里
│   ├── policies/
│   └── benefits/
│
├── rag/              ← 核心模块（不用改）
│   ├── chunker.py    - 文本分块
│   ├── embedder.py   - 向量化
│   ├── vectordb.py   - 数据库
│   ├── retriever.py  - 检索
│   ├── ingestion.py  - 文档摄入
│   └── llm.py       - LLM 调用
│
└── scripts/          ← 脚本（可以直接运行）
    ├── 1_init_db.py
    ├── 2_ingest_docs.py
    ├── 3_test_query.py
    └── 4_run_rag.py
```

---

## 七、下一步

### 添加自己的文档
1. 将文档放入 `data/documents/` 目录
2. 按类别组织（如 policies、benefits、handbook）
3. 支持 .md 和 .txt 文件
4. 运行 `python main.py ingest` 重新摄入

### 调整配置
在 `config.py` 中可以调整：
- `CHUNK_SIZE` - 块大小（默认500）
- `CHUNK_OVERLAP` - 重叠大小（默认100）
- `TOP_K_RESULTS` - 检索结果数（默认3）

### 切换 LLM
修改 `.env` 中的 `LLM_PROVIDER`：
- `openai` - OpenAI GPT
- `anthropic` - Claude
- `zhipu` - 智谱AI
- `qwen` - 通义千问

---

## 八、获取帮助

```bash
# 查看帮助
python main.py --help

# 测试配置
python main.py config

# 查看 README
cat README.md
```

---

## 🎉 开始使用吧！

完成以上步骤后，你就有了一个完整的 RAG 系统！

**核心流程回顾**：
1. 文档 → 分块 → 向量化 → 存入数据库
2. 用户提问 → 向量化 → 检索相关文档
3. 文档 + 问题 → LLM → 生成答案

**祝你使用愉快！🚀**
