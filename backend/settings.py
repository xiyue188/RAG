"""
后端配置
从项目根目录的config.py显式导入需要的配置
"""

import os
from config import (
    PROJECT_ROOT, DATA_DIR, DB_DIR,
    CHROMA_DB_PATH, COLLECTION_NAME,
    TOP_K_RESULTS,
)

# ==================== 服务器配置 ====================

BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000

# CORS配置
# 生产部署：在 .env 中设置 ALLOWED_ORIGINS，用逗号分隔多个域名
# 例：ALLOWED_ORIGINS=http://your-server-ip,https://your-domain.com
# 开发环境：不设置则默认允许常用本地地址
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")

if _allowed_origins_env:
    # 生产模式：从环境变量读取，精确控制
    CORS_ORIGINS = [origin.strip() for origin in _allowed_origins_env.split(",") if origin.strip()]
else:
    # 开发模式：允许本地常用端口
    CORS_ORIGINS = [
        "http://localhost:5173",   # Vite开发服务器
        "http://localhost:3000",   # 前端容器（开发）
        "http://localhost:80",     # 前端容器（生产端口）
        "http://localhost",        # 标准HTTP
    ]

# Session配置
SESSION_TIMEOUT = 3600  # Session超时（1小时）
MAX_SESSIONS = 1000

# API配置
API_PREFIX = "/api/v1"

# 流式输出配置
STREAM_BUFFER_SIZE = 10  # 缓冲区大小（字符数），用于[doc_X]标记检测
