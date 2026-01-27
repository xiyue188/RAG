"""
配置文件 - 唯一真相源（Single Source of Truth）
所有配置参数集中管理在这里
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件 - 指定完整路径
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# ============================================================
# 项目路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
DB_DIR = PROJECT_ROOT / "chroma_db"

# ============================================================
# 向量数据库配置
# ============================================================
CHROMA_DB_PATH = str(DB_DIR)
COLLECTION_NAME = "techcorp_docs"
SIMILARITY_METRIC = "cosine"  # 可选: cosine, l2, ip

# ============================================================
# Embedding 模型配置
# ============================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 的输出维度

# ============================================================
# 文本分块配置
# ============================================================
CHUNK_SIZE = 500        # 每个块的字符数
CHUNK_OVERLAP = 100     # 重叠字符数
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP  # 滑动步长 = 400

# ============================================================
# 检索配置
# ============================================================
TOP_K_RESULTS = 3       # 检索返回的 top-k 结果数量
MIN_RELEVANCE_SCORE = 0.3  # 最小相关性阈值（可选）

# ============================================================
# LLM API 配置
# ============================================================
# 支持多种 LLM 提供商
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, zhipu, qwen

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
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
LLM_TIMEOUT = 30  # 秒

# ============================================================
# RAG 系统提示词模板
# ============================================================
SYSTEM_PROMPT = """你是 TechCorp 公司的智能助手。
你的任务是根据提供的文档内容回答用户问题。

重要规则：
1. 只基于提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，明确告知用户
3. 回答要简洁、准确、有帮助
4. 可以引用具体的政策或文档内容
"""

QUERY_TEMPLATE = """基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{question}

回答："""

# ============================================================
# 日志配置
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = PROJECT_ROOT / "rag.log"

# ============================================================
# 性能配置
# ============================================================
BATCH_SIZE = 32  # 批量处理大小
USE_GPU = False  # 是否使用 GPU（需要安装 torch-gpu）

# ============================================================
# 文档处理配置
# ============================================================
SUPPORTED_FILE_TYPES = [".md", ".txt", ".pdf"]  # 支持的文件类型
ENCODING = "utf-8"  # 文件编码

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
