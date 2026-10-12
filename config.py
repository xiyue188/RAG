"""
配置文件 - 唯一真相源（Single Source of Truth）
从 .env 读取所有配置参数，提供默认值和验证
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================
# 辅助函数
# ============================================================

def get_bool(key: str, default: bool = False) -> bool:
    """从环境变量读取布尔值"""
    return os.getenv(key, str(default)).lower() in ('true', '1', 'yes')

def get_int(key: str, default: int) -> int:
    """从环境变量读取整数"""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_float(key: str, default: float) -> float:
    """从环境变量读取浮点数"""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

PLACEHOLDER_VALUES = {
    "",
    "your-api-key-here",
    "sk-your-openai-api-key-here",
    "sk-your-openai-key",
    "sk-your-openai-key-here",
    "sk-ant-your-anthropic-api-key-here",
    "your-zhipu-api-key-here",
    "your-zhipu-api-key",
    "your-key",
    "sk-your-qwen-api-key-here",
}

def has_real_secret(value: str | None) -> bool:
    """检查密钥是否不是空值或示例占位符。"""
    if value is None:
        return False
    normalized = value.strip()
    return bool(normalized) and normalized.lower() not in PLACEHOLDER_VALUES

# ============================================================
# 项目路径配置（不可变）
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
DB_DIR = PROJECT_ROOT / "chroma_db"

# ============================================================
# 关系数据库（可追溯轨迹、会话、评测结果）
# ============================================================
# 向量继续留在 ChromaDB，不引入 pgvector：数据量用不到其优势，
# 而 evidence 表只存 chunk_id 这个字符串指针，不碰向量本身。
#
# 默认 SQLite 是为了零运维起步（Windows 本地开发无需 Docker）。
# 生产/多副本必须换 PostgreSQL —— SQLite 单写者模型下，
# 并发写会拿不到锁而报 "database is locked"。
# 两者共用同一套 SQLAlchemy 模型与 Alembic 迁移，只改这一行。
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}",
)

# ============================================================
# Redis（缓存、限流、取消信号、摄入进度）
# ============================================================
# Redis 不作为事实来源：其中全部内容可丢弃后重建。
# 因此所有依赖 Redis 的功能都必须有降级路径 ——
# 限流降级为放行，缓存降级为穿透，锁降级为无锁。
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = get_bool("REDIS_ENABLED", True)

# 缓存 TTL（秒）
CACHE_TTL_RETRIEVAL = get_int("CACHE_TTL_RETRIEVAL", 3600)
CACHE_TTL_EMBEDDING = get_int("CACHE_TTL_EMBEDDING", 86400)

# 限流：每 IP 每窗口允许的请求数
RATE_LIMIT_PER_MINUTE = get_int("RATE_LIMIT_PER_MINUTE", 30)

# 限流窗口长度（秒）。默认 60 秒与 RATE_LIMIT_PER_MINUTE 对应。
# 改短窗口可以得到更平滑的限制（如 10s 内最多 5 次），不需要改 RATE_LIMIT_PER_MINUTE。
RATE_LIMIT_WINDOW_SECONDS = get_int("RATE_LIMIT_WINDOW_SECONDS", 60)

# 可信反向代理 IP 列表。只有来自这些 IP 的请求才会采信 x-forwarded-for。
# 多个 IP 用逗号分隔，例：TRUSTED_PROXY_IPS=10.0.0.1,10.0.0.2
# 默认空列表 = 不信任任何代理 = 直接用 TCP 连接 IP 限流（安全默认）。
# 在 Nginx / k8s Ingress 后运行时，把代理节点 IP 加进来。
TRUSTED_PROXY_IPS: list[str] = [
    ip.strip()
    for ip in os.getenv("TRUSTED_PROXY_IPS", "").split(",")
    if ip.strip()
]

# 异步摄入：pending / running 超过该时长即视为失效并标 error。
# 必须大于"排队 + 单个文件摄入"的最长耗时，否则会清掉还在排队的活任务。
# 10MB 上限的文件在 CPU 上摄入约数分钟，默认 30 分钟留足余量。
INGEST_STALE_SECONDS = get_int("INGEST_STALE_SECONDS", 1800)
# 清扫间隔
INGEST_SWEEP_INTERVAL_SECONDS = get_int("INGEST_SWEEP_INTERVAL_SECONDS", 300)

# ============================================================
# Embedding 模型配置
# ============================================================
# 默认使用中文模型：知识库与查询均为中文，英文模型（all-MiniLM-L6-v2）
# 在中文语料上向量检索接近噪声排序。
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")

# bge 中文系列要求查询侧加指令前缀、文档侧不加，该非对称性必须复现，
# 否则检索收益大幅下降。换回非 bge 模型时自动不加前缀。
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def needs_query_prefix(model_name: str) -> bool:
    """判断该 embedding 模型是否需要查询侧指令前缀。"""
    lowered = model_name.lower()
    return "bge" in lowered and "zh" in lowered


# ============================================================
# 向量数据库配置
# ============================================================
CHROMA_DB_PATH = str(DB_DIR)
COLLECTION_BASE_NAME = os.getenv("COLLECTION_NAME", "techcorp_docs")
SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", "cosine")


def model_slug(model_name: str) -> str:
    """把模型名转成可用作 collection 名的片段。"""
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def collection_name_for(model_name: str) -> str:
    """按 embedding 模型生成隔离的 collection 名。

    不同模型的向量必须物理隔离：维度不同（384 vs 512）会直接报错；
    维度相同则静默返回垃圾结果且极难排查。
    隔离同时是 A/B 对比与回滚的前提 —— 换模型不销毁旧数据。

    例：techcorp_docs__baai_bge_small_zh_v1_5
    """
    return f"{COLLECTION_BASE_NAME}__{model_slug(model_name)}"


COLLECTION_NAME = collection_name_for(EMBEDDING_MODEL_NAME)

# ============================================================
# 文本分块配置
# ============================================================
CHUNK_SIZE = get_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 100)
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP

# ============================================================
# 检索配置
# ============================================================
TOP_K_RESULTS = get_int("TOP_K_RESULTS", 3)

# ── 相关性阈值 ────────────────────────────────────────────────
# 两个阈值都建立在同一个口径上：relevance ∈ [0,1]，越大越相关，
# 由 rag.scoring.compute_relevance() 统一计算。
#
# 此前只有一个 SIMILARITY_THRESHOLD，却在两处被当作不同物理量使用
# （一处是 hybrid_score 下限，越大越好；一处是余弦距离上限，越小越好），
# 导致任何基于它的指标都无法解释。现拆为两个语义明确的阈值。
#
# M1 已用 300 条 T2Ranking 校准（横轴阈值、纵轴无答案识别率与误拒率，
# 取拐点），见 docs/eval/threshold.md。
# **这两个默认值必须与 .env.example 一致**：不一致时，没有 .env 的环境
# （CI、新克隆的机器）与开发机行为不同，而这种差异不会报错。
RETRIEVAL_MIN_RELEVANCE = get_float("RETRIEVAL_MIN_RELEVANCE", 0.35)   # 低于此值不进 prompt 上下文
# 0.75 是 M1 校准结论（docs/eval/threshold.md「建议」节）。
# 此前默认值是拍定的 0.50 而 .env 写 0.75，两者不一致 ——
# 校准做完只改了 .env，忘了同步这里。
ANSWERABLE_MIN_RELEVANCE = get_float("ANSWERABLE_MIN_RELEVANCE", 0.75)  # top1 低于此值判定知识库无答案

# RRF 融合平滑常数，见 rag/scoring.py
RRF_K = get_int("RRF_K", 60)

# 检索模式（universal=全库搜索，metadata_only=类别名过滤，keyword=关键词分类）
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "universal")

# 检索优化开关
ENABLE_THRESHOLD_FILTERING = get_bool("ENABLE_THRESHOLD_FILTERING", True)
ENABLE_AUTO_CLASSIFICATION = get_bool("ENABLE_AUTO_CLASSIFICATION", True)

# ============================================================
# 引用追踪配置（Phase 2）
# ============================================================
ENABLE_CITATION_TRACKING = get_bool("ENABLE_CITATION_TRACKING", True)  # 启用引用追踪
CITATION_MODE = os.getenv("CITATION_MODE", "inline")  # inline(内联标记) 或 json(遗留)
CITATION_STYLE = os.getenv("CITATION_STYLE", "inline")  # inline(句后标注) 或 footnote(脚注)

# ============================================================
# LLM增强检索配置（阶段2）
# ============================================================
# ENABLE_QUERY_REWRITE 已删除（Phase 1 优化：实测导致检索退化）
ENABLE_MULTI_QUERY = get_bool("ENABLE_MULTI_QUERY", False)   # Hybrid 模式下自动跳过
NUM_EXPANDED_QUERIES = get_int("NUM_EXPANDED_QUERIES", 3)

# ============================================================
# Rerank精排序配置（阶段3）
# ============================================================
ENABLE_RERANK = get_bool("ENABLE_RERANK", False)
RERANK_TOP_K = get_int("RERANK_TOP_K", 20)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")  # 中文首选

# ============================================================
# Hybrid 混合检索配置
# ============================================================
# 融合方式为 RRF（Reciprocal Rank Fusion），只用排名不用分数，
# 因此不存在 BM25_WEIGHT / VECTOR_WEIGHT —— 两路分数量纲不同，
# 加权相加没有物理意义，原先的 0.7/0.3 是无效参数。
ENABLE_HYBRID = get_bool("ENABLE_HYBRID", True)   # 默认开启，中文场景首选
HYBRID_TOP_K = get_int("HYBRID_TOP_K", 20)       # 每路召回的候选数

# ============================================================
# 对话管理配置（Phase 1）
# ============================================================
MAX_CONVERSATION_TURNS = get_int("MAX_CONVERSATION_TURNS", 20)  # 最大保留对话轮次
ENABLE_REFERENCE_RESOLUTION = get_bool("ENABLE_REFERENCE_RESOLUTION", True)  # 启用指代消解
# ENABLE_CONTEXT_AWARE_REWRITE 已删除（Query Rewrite 整体已移除）

# Multi-Query 提示词模板
MULTI_QUERY_PROMPT = """请为以下查询生成 {n} 个不同角度的变体查询，用于提高检索召回率。
要求：
1. 每个变体从不同角度描述同一个问题
2. 使用不同的关键词和表达方式
3. 每行一个查询，不要编号
4. 只输出查询，不要解释

原始查询：{query}

变体查询："""

# 关键词分类映射（仅在 RETRIEVAL_MODE="keyword" 时使用）
# 根据您的实际领域修改以下关键词
CATEGORY_KEYWORDS = {
    "benefits": [
        "401k", "保险", "福利", "假期", "休假", "年假", "病假", "健康",
        "退休", "养老", "医疗", "牙科", "视力", "401K", "配对", "匹配"
    ],
    "policies": [
        "宠物", "远程", "办公", "着装", "考勤", "规定", "政策", "制度",
        "出勤", "迟到", "请假", "工作时间", "dress code", "pet", "remote"
    ],
    "general": [
        "公司", "文化", "介绍", "关于", "是什么", "历史", "使命", "愿景"
    ]
}

# ============================================================
# LLM API 配置
# ============================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu")

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Anthropic Claude 配置
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# 智谱AI 配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash-250414")

# 通义千问 配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.6-plus")

# LLM 生成参数
LLM_TEMPERATURE = get_float("LLM_TEMPERATURE", 0.5)   # RAG场景偏低更好
LLM_MAX_TOKENS = get_int("LLM_MAX_TOKENS", 800)
LLM_TIMEOUT = get_int("LLM_TIMEOUT", 30)

# ============================================================
# RAG 系统提示词模板
# ============================================================

# 严格模式：只基于文档回答
SYSTEM_PROMPT_STRICT = """你是一个智能文档助手。
你的任务是根据提供的文档内容回答用户问题。

重要规则：
1. 只基于提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，明确告知用户
3. 回答要简洁、准确、有帮助
4. 可以引用具体的政策或文档内容
"""

# 混合模式：文档优先 + LLM 补充
SYSTEM_PROMPT_HYBRID = """你是一个智能文档助手。
你的任务是根据提供的文档内容回答用户问题。

重要规则：
1. 优先使用提供的文档内容回答问题
2. 如果文档内容不足以完整回答问题，可以基于你的通用知识补充
3. 当使用通用知识时，必须明确标注："以下内容来自通用知识，非文档内容"
4. 回答要简洁、准确、有帮助
5. 可以引用具体的政策或文档内容
"""

# ============================================================
# 自定义风格模板（可选）
# ============================================================

# 专业正式风格
SYSTEM_PROMPT_PROFESSIONAL = """你是一个专业的文档知识顾问。

回答风格：
• 使用正式、专业的语言
• 避免口语化表达
• 引用文档时注明具体来源
• 如有不确定的信息，明确说明

回答规则：
1. 优先使用提供的文档内容
2. 回答结构清晰，使用分点列举
3. 当使用通用知识补充时，使用【补充说明】标签
4. 避免使用 emoji 或表情符号
"""

# 友好简洁风格
SYSTEM_PROMPT_FRIENDLY = """你是一个友好的智能文档助手。

回答风格：
• 友好、亲切、易懂
• 语言简洁，避免冗长
• 必要时可以使用 emoji 让回答更生动
• 多用"你"而非"您"

回答规则：
1. 优先使用文档内容回答
2. 用通俗易懂的语言解释复杂概念
3. 如果文档内容不足，可以补充（标注来源）
4. 结尾可以主动询问是否需要更多信息
"""

# 技术详细风格
SYSTEM_PROMPT_TECHNICAL = """你是一个技术文档助手。

回答风格：
• 提供详细、全面的技术信息
• 使用准确的技术术语
• 包含相关的代码示例或配置说明
• 提供多个解决方案供用户选择

回答规则：
1. 基于文档内容提供技术细节
2. 补充相关的最佳实践建议（标注来源）
3. 使用代码块、列表等格式化输出
4. 提供参考链接或相关文档建议
"""

# ============================================================
# 选择使用的风格（修改这里切换风格）
# ============================================================

# 默认使用混合模式（文档优先 + 允许补充）
SYSTEM_PROMPT = SYSTEM_PROMPT_HYBRID

# 其他可选风格：
# SYSTEM_PROMPT = SYSTEM_PROMPT_STRICT        # 严格模式（只用文档）
# SYSTEM_PROMPT = SYSTEM_PROMPT_PROFESSIONAL  # 专业正式风格
# SYSTEM_PROMPT = SYSTEM_PROMPT_FRIENDLY      # 友好简洁风格
# SYSTEM_PROMPT = SYSTEM_PROMPT_TECHNICAL     # 技术详细风格

# ============================================================
# 查询模板（控制如何向 LLM 提问）
# ============================================================

# 标准模板
QUERY_TEMPLATE = """基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答："""

# 结构化模板（要求分点回答）
QUERY_TEMPLATE_STRUCTURED = """请基于以下上下文信息，以清晰的结构回答用户问题。

【上下文信息】
{context}

【用户问题】
{question}

【回答要求】
1. 直接回答核心问题
2. 使用分点列举（如适用）
3. 引用文档来源
4. 保持简洁专业

【你的回答】
"""

# 简短模板（要求简洁回答）
QUERY_TEMPLATE_CONCISE = """根据以下信息，用1-2句话简洁回答问题。

信息：{context}
问题：{question}
回答："""

# 详细模板（要求详细解释）
QUERY_TEMPLATE_DETAILED = """请基于以下上下文信息，详细回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答要求：
• 提供完整详细的解释
• 包含所有相关细节
• 如有多个方面，分点说明
• 可以补充相关建议（标注来源）

详细回答："""

# 当前使用的模板（修改这里切换）
# QUERY_TEMPLATE = QUERY_TEMPLATE_STRUCTURED  # 结构化
# QUERY_TEMPLATE = QUERY_TEMPLATE_CONCISE     # 简短
# QUERY_TEMPLATE = QUERY_TEMPLATE_DETAILED    # 详细

# 无上下文时的提示模板
NO_CONTEXT_TEMPLATE = """用户问题：{question}

注意：知识库中没有找到相关文档。请基于你的通用知识回答，并明确说明这不是来自知识库的文档。

回答："""

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "rag.log"

# ============================================================
# 性能配置（固定值，一般不需调整）
# ============================================================
BATCH_SIZE = 32
USE_GPU = False

# ============================================================
# 文档处理配置（固定值）
# ============================================================
SUPPORTED_FILE_TYPES = [".md", ".txt", ".pdf", ".docx"]
ENCODING = "utf-8"

# ============================================================
# 辅助函数
# ============================================================
def ensure_directories():
    """确保必要的目录存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

def validate_config():
    """验证配置的有效性"""
    errors = []

    # 检查 LLM API Key
    if LLM_PROVIDER == "openai" and not has_real_secret(OPENAI_API_KEY):
        errors.append("OPENAI_API_KEY 未设置")
    elif LLM_PROVIDER == "anthropic" and not has_real_secret(ANTHROPIC_API_KEY):
        errors.append("ANTHROPIC_API_KEY 未设置")
    elif LLM_PROVIDER == "zhipu" and not has_real_secret(ZHIPU_API_KEY):
        errors.append("ZHIPU_API_KEY 未设置")
    elif LLM_PROVIDER == "qwen" and not has_real_secret(QWEN_API_KEY):
        errors.append("QWEN_API_KEY 未设置")

    # 检查参数合理性
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) 必须小于 CHUNK_SIZE ({CHUNK_SIZE})")

    if errors:
        raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))

    return True

def get_llm_config():
    """获取当前 LLM 提供商的配置"""
    configs = {
        "openai": {
            "api_key": OPENAI_API_KEY,
            "api_base": OPENAI_API_BASE,
            "model": OPENAI_MODEL,
        },
        "anthropic": {
            "api_key": ANTHROPIC_API_KEY,
            "model": ANTHROPIC_MODEL,
        },
        "zhipu": {
            "api_key": ZHIPU_API_KEY,
            "model": ZHIPU_MODEL,
        },
        "qwen": {
            "api_key": QWEN_API_KEY,
            "model": QWEN_MODEL,
        },
    }
    return configs.get(LLM_PROVIDER, {})

if __name__ == "__main__":
    # 测试配置
    print("=" * 70)
    print("配置检查")
    print("=" * 70)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"数据库目录: {DB_DIR}")
    print(f"Embedding 模型: {EMBEDDING_MODEL_NAME}")
    print(f"分块大小: {CHUNK_SIZE}, 重叠: {CHUNK_OVERLAP}")
    print(f"LLM 提供商: {LLM_PROVIDER}")
    print(f"检索 Top-K: {TOP_K_RESULTS}")

    ensure_directories()
    print("\n✓ 目录已创建")

    try:
        validate_config()
        print("✓ 配置验证通过")
    except ValueError as e:
        print(f"✗ {e}")
