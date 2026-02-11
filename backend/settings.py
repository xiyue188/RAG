"""
后端配置
从项目根目录的config.py显式导入需要的配置
"""

from config import (
    PROJECT_ROOT, DATA_DIR, DB_DIR,
    CHROMA_DB_PATH, COLLECTION_NAME,
    TOP_K_RESULTS,
)

# ==================== 服务器配置 ====================

BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000

# CORS配置
CORS_ORIGINS = [
    "http://localhost:5173",  # Vite开发服务器
    "http://localhost:3000",  # 前端容器
    "http://localhost:8080",  # 备用端口
    "*",                      # 开发阶段允许所有来源
]

# Session配置
SESSION_TIMEOUT = 3600  # Session超时（1小时）
MAX_SESSIONS = 1000

# API配置
API_PREFIX = "/api/v1"

# 流式输出配置
STREAM_BUFFER_SIZE = 10  # 缓冲区大小（字符数），用于[doc_X]标记检测
