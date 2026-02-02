"""
配置文件 - 唯一真相源（Single Source of Truth）
从 .env 读取所有配置参数，提供默认值和验证
"""

import os
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

# ============================================================
# 项目路径配置（不可变）
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
DB_DIR = PROJECT_ROOT / "chroma_db"

# ============================================================
# 向量数据库配置
# ============================================================
CHROMA_DB_PATH = str(DB_DIR)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "techcorp_docs")
SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", "cosine")

# ============================================================
# Embedding 模型配置
# ============================================================
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 固定维度

# ============================================================
# 文本分块配置
# ============================================================
CHUNK_SIZE = get_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 100)
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP

# ============================================================
# 检索配置（阶段1）
# ============================================================
TOP_K_RESULTS = get_int("TOP_K_RESULTS", 3)
SIMILARITY_THRESHOLD = get_float("SIMILARITY_THRESHOLD", 0.7)
RETRIEVAL_DISTANCE_THRESHOLD = get_float("RETRIEVAL_DISTANCE_THRESHOLD", 0.7)
MIN_RELEVANCE_SCORE = 0.3  # 保留用于未来扩展

# 检索模式
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "metadata_only")

# 检索优化开关
ENABLE_THRESHOLD_FILTERING = get_bool("ENABLE_THRESHOLD_FILTERING", True)
ENABLE_AUTO_CLASSIFICATION = get_bool("ENABLE_AUTO_CLASSIFICATION", True)

# ============================================================
# 阶段2: LLM增强检索配置
# ============================================================
ENABLE_QUERY_REWRITE = get_bool("ENABLE_QUERY_REWRITE", False)
ENABLE_MULTI_QUERY = get_bool("ENABLE_MULTI_QUERY", False)
NUM_EXPANDED_QUERIES = get_int("NUM_EXPANDED_QUERIES", 3)

# Query Rewrite 提示词模板
QUERY_REWRITE_PROMPT = """请将以下用户查询重写为更适合向量检索的形式。
要求：
1. 保持原意，但使用更精确的关键词
2. 去除口语化表达
3. 补充可能的同义词或相关术语
4. 只输出重写后的查询，不要解释

原始查询：{query}

重写后的查询："""

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Anthropic Claude 配置
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

# 智谱AI 配置
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4")

# 通义千问 配置
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")

# LLM 生成参数
LLM_TEMPERATURE = get_float("LLM_TEMPERATURE", 0.7)
LLM_MAX_TOKENS = get_int("LLM_MAX_TOKENS", 500)
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
SUPPORTED_FILE_TYPES = [".md", ".txt", ".pdf"]
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
    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY 未设置")
    elif LLM_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY 未设置")
    elif LLM_PROVIDER == "zhipu" and not ZHIPU_API_KEY:
        errors.append("ZHIPU_API_KEY 未设置")
    elif LLM_PROVIDER == "qwen" and not QWEN_API_KEY:
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
