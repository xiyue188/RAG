"""
FastAPI主应用
六边形架构 - Web适配层入口点
"""

import sys
import io
from pathlib import Path
from contextlib import asynccontextmanager

# 解决Windows GBK编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径（仅在入口点设置一次）
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.settings import CORS_ORIGINS, BACKEND_HOST, BACKEND_PORT
from backend.rate_limit import RateLimitMiddleware
from backend.env_validation import validate_env, get_env_info

# 验证环境变量（启动时立即执行）
validate_env()
env_info = get_env_info()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 60)
    print("DeepBlue Intelligence API starting...")
    print(f"  Docs:   http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print(f"  SSE:    http://{BACKEND_HOST}:{BACKEND_PORT}/api/v1/chat/stream")
    print(f"  REST:   http://{BACKEND_HOST}:{BACKEND_PORT}/api/v1/chat/message")
    print(f"  LLM:    {env_info['llm_provider']}")
    print(f"  Rate Limit: 10 requests/minute per IP")
    print("=" * 60)
    yield
    print("DeepBlue Intelligence API stopped.")


app = FastAPI(
    title="DeepBlue Intelligence API",
    description="RAG交互系统后端API - 六边形架构适配层\n\n速率限制：每IP每分钟10次请求",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS中间件（必须在速率限制之前）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 速率限制中间件
app.add_middleware(RateLimitMiddleware, requests_per_minute=10)

# 注册路由
from backend.api import routes, sse
from backend.schemas import ErrorResponse, ErrorDetail
from fastapi.responses import JSONResponse
from fastapi import status

app.include_router(routes.router, prefix="/api/v1", tags=["REST API"])
app.include_router(sse.router, prefix="/api/v1", tags=["SSE Stream"])


# ==================== 全局异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """全局异常处理器 - 统一错误响应格式"""
    error_response = ErrorResponse(
        error=ErrorDetail(
            message=str(exc),
            type=type(exc).__name__,
            detail=None
        ),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        path=str(request.url.path)
    )

    return JSONResponse(
        status_code=error_response.status_code,
        content=error_response.model_dump()
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 错误处理"""
    error_response = ErrorResponse(
        error=ErrorDetail(
            message="Resource not found",
            type="NotFoundError",
            detail=f"The requested path '{request.url.path}' does not exist"
        ),
        status_code=404,
        path=str(request.url.path)
    )

    return JSONResponse(
        status_code=404,
        content=error_response.model_dump()
    )


# ==================== 系统端点 ====================


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "DeepBlue Intelligence RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
        log_level="info"
    )
