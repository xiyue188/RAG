"""
速率限制中间件
基于IP的简单内存限流，适合单机部署
"""

import time
from collections import defaultdict, deque
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    简单的内存速率限制中间件

    策略：滑动窗口
    - 每个IP存储最近N秒的请求时间戳
    - 超过阈值则拒绝请求
    """

    def __init__(self, app, requests_per_minute: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        # 存储每个IP的请求时间戳队列
        self.request_history = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # 系统端点不限流
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # 获取客户端IP
        client_ip = request.client.host

        # 当前时间
        now = time.time()

        # 清理过期的请求记录
        request_times = self.request_history[client_ip]
        while request_times and request_times[0] < now - self.window_seconds:
            request_times.popleft()

        # 检查是否超过限制
        if len(request_times) >= self.requests_per_minute:
            # 计算需要等待的时间
            oldest_request = request_times[0]
            wait_time = int(self.window_seconds - (now - oldest_request))

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_minute} requests per minute.",
                    "retry_after": wait_time,
                },
                headers={"Retry-After": str(wait_time)}
            )

        # 记录此次请求
        request_times.append(now)

        # 执行请求
        response = await call_next(request)

        # 添加速率限制相关的响应头
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - len(request_times))
        )

        return response
