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
from backend.settings import (
    CORS_ORIGINS, BACKEND_HOST, BACKEND_PORT,
    DATABASE_URL, RATE_LIMIT_PER_MINUTE,
)
from backend.rate_limit import RateLimitMiddleware
from backend.env_validation import validate_env, get_env_info

# 验证环境变量（启动时立即执行）
validate_env()
env_info = get_env_info()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理

    启动时只**探测**依赖，不因它们不可用而拒绝启动：

    - 数据库不通 → 记日志继续。此刻还没有任何请求路径依赖它（M2 的
      Repository 层尚未接入），启动即退出只会让人以为是别的问题。
    - Redis 不通 → 同上，且这是设计要求：限流降级放行、缓存降级穿透。

    **不在这里建表。** 建表只有 `alembic upgrade head` 一条路径 ——
    启动时跑 `create_all` 会与迁移形成两个真相源，模型改了却只跑
    create_all 就会得到旧表（它不更新已存在的表），而且多副本同时启动
    时并发建表会互相撞上。
    """
    from backend.cache import redis_client
    from backend.db import session as db_session
    from backend.services import ingest_service

    db_ok = await db_session.healthcheck()
    redis_ok = await redis_client.ping()

    # T6：启动后台摄入 worker，回收进程重启时遗留的孤儿任务
    await ingest_service.start_worker()
    orphans = await ingest_service.reclaim_pending()
    if orphans:
        print(f"  ⚠ reclaimed {orphans} orphaned ingest task(s) as error")

    print("=" * 60)
    print("DeepBlue Intelligence API starting...")
    print(f"  Docs:   http://localhost:{BACKEND_PORT}/docs")
    print(f"  SSE:    http://localhost:{BACKEND_PORT}/api/v1/chat/stream")
    print(f"  REST:   http://localhost:{BACKEND_PORT}/api/v1/chat/message")
    print(f"  LLM:    {env_info['llm_provider']}")
    print(f"  Rate Limit: {RATE_LIMIT_PER_MINUTE} requests/minute per IP")
    # 只打方言，不打完整连接串 —— 后者含密码
    print(f"  DB:     {DATABASE_URL.split('://')[0]} {'ok' if db_ok else 'unavailable'}")
    print(f"  Redis:  {'ok' if redis_ok else 'unavailable (降级运行)'}")
    print("=" * 60)

    yield

    # 反序关闭。不关连接池会在进程退出时报 "Event loop is closed" ——
    # 连接的析构逻辑跑在已关闭的事件循环上。
    await ingest_service.stop_worker()
    await redis_client.close()
    await db_session.dispose_engine()
    print("DeepBlue Intelligence API stopped.")


app = FastAPI(
    title="DeepBlue Intelligence API",
    description=(
        "RAG交互系统后端API - 六边形架构适配层\n\n"
        f"速率限制：每IP每分钟 {RATE_LIMIT_PER_MINUTE} 次请求"
    ),
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
# 取配置值而非写死 30：写死会让 .env 里改了 RATE_LIMIT_PER_MINUTE 不生效，
# 而启动日志又照着配置打印，两者对不上时排查会绕远路
app.add_middleware(RateLimitMiddleware, requests_per_minute=RATE_LIMIT_PER_MINUTE)

# 注册路由
from backend.api import routes, sse, upload
from backend.schemas import ErrorResponse, ErrorDetail
from fastapi.responses import JSONResponse
from fastapi import status

app.include_router(routes.router, prefix="/api/v1", tags=["REST API"])
app.include_router(sse.router, prefix="/api/v1", tags=["SSE Stream"])
app.include_router(upload.router, prefix="/api/v1", tags=["Documents"])


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
    """健康检查。

    **依赖不可用时仍返回 200。** 这个端点是容器编排的存活探针 ——
    返回非 200 会让 Docker 重启容器，而重启一个"数据库连不上"的进程
    解决不了任何问题，只会让服务在重启循环里彻底不可用。
    组件状态放在响应体里，由监控去判断该告警哪一个。
    """
    from backend.cache import redis_client
    from backend.db import session as db_session

    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "database": "ok" if await db_session.healthcheck() else "unavailable",
            # Redis 不可用不影响正确性（限流放行、缓存穿透），
            # 因此这里是 degraded 而非 unavailable
            "redis": "ok" if await redis_client.ping() else "degraded",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
        log_level="info"
    )
