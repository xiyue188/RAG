"""
速率限制中间件

**三个已知问题的修复（T4）：**

1. **内存态 → Redis 优先，降级为内存。**
   原先的 `defaultdict(deque)` 重启归零，多副本各自计数（N 副本等于配额翻 N 倍）。
   现在：Redis 可用时用 INCR+EXPIRE 固定窗口；Redis 不可用时降级为本地内存，
   此时降级为单机限流，行为与原实现一致，不拒绝流量。

2. **无界内存增长修复。**
   原先每个见过的 IP 都建一个 deque 且永不清理：被换 IP 的扫描器会把
   `request_history` 撑到进程 OOM。
   修法：内存表加上限 `_MAX_TRACKED_IPS`（默认 10000），按 LRU 淘汰：
   每次访问 `move_to_end`，满了 `popitem(last=False)` 弹出最久未访问的 IP。
   选 LRU 而非 FIFO：扫描器灌入大量新 IP 时，FIFO 会把正在活跃的正常
   用户也按插入顺序挤出去（其计数随之清零，等于给他重置配额）；LRU 只淘汰
   最久没来的那些。

3. **XFF 按可信代理校验。**
   原先无条件信任 `x-forwarded-for`，攻击者每个请求伪造一个 IP 即可绕过限流。
   修法：只有请求来自 `TRUSTED_PROXY_IPS` 列表里的 IP 时才采信 XFF，
   否则直接用 `request.client.host`。默认空列表 = 不信任任何代理 = 取直连 IP。
   在 Nginx/k8s Ingress 前使用时，把代理 IP 加进 `TRUSTED_PROXY_IPS`。

**降级策略**：Redis 不可用时放行而非拒绝，限流是保护手段，
不该因为 Redis 挂了就把全部流量挡住。
"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from typing import Iterable, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import config
from backend.cache import redis_client

# 内存降级表的 IP 数量上限。超出时淘汰最久未访问的 IP（LRU）。
# 10000 个 IP × 每条记录约 200 字节 ≈ 2MB，不影响进程内存。
_MAX_TRACKED_IPS = 10_000


def _get_client_ip(request: Request, trusted_proxies: frozenset[str]) -> str:
    """从请求里取客户端 IP。

    只有请求来自可信代理时才采信 XFF，否则直接用 TCP 连接的对端地址。
    XFF 的第一个值是原始客户端（从左往右，最近的代理追加在右边），
    多层代理时只取第一段。
    """
    direct_ip = request.client.host if request.client else "unknown"

    if direct_ip not in trusted_proxies:
        # 连接不是来自已知代理，XFF 可能是攻击者伪造的，忽略它
        return direct_ip

    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if not forwarded_for:
        return direct_ip

    # 取最左边一段（原始客户端 IP）
    return forwarded_for.split(",")[0].strip() or direct_ip


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    速率限制：Redis 固定窗口为主，内存滑动窗口为降级。

    主路径：Redis INCR+EXPIRE 固定窗口（多副本共享计数，重启不归零）。
    降级路径：进程内 OrderedDict + deque 滑动窗口（单机，有界内存，LRU 淘汰）。

    `window_seconds` / `trusted_proxies` 可注入，默认取 config。
    中间件在第一次请求时才由 Starlette 实例化，测试若靠临时改全局 config
    传参，改回之后才实例化，参数实际不生效。
    """

    # 跳过限流的路径
    _EXEMPT_PATHS = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})

    def __init__(
        self,
        app,
        requests_per_minute: int = 10,
        window_seconds: Optional[int] = None,
        trusted_proxies: Optional[Iterable[str]] = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = (
            window_seconds if window_seconds is not None
            else config.RATE_LIMIT_WINDOW_SECONDS
        )

        # 可信代理 IP 集合（frozenset 查找 O(1)）
        proxies = trusted_proxies if trusted_proxies is not None else config.TRUSTED_PROXY_IPS
        self.trusted_proxies: frozenset[str] = frozenset(
            ip.strip() for ip in proxies if ip.strip()
        )

        # 降级用的内存表：OrderedDict 保序，便于 LRU 淘汰
        self._mem_table: OrderedDict[str, deque] = OrderedDict()

    # ------------------------------------------------------------------
    # Redis 路径（优先）
    # ------------------------------------------------------------------

    async def _check_redis(self, ip: str) -> Optional[tuple[bool, int]]:
        """用 Redis INCR+EXPIRE 实现固定窗口计数。

        返回 (is_limited, count)，或 None 表示 Redis 不可用（让调用方降级）。

        为什么用固定窗口（INCR+EXPIRE）而不是 sorted-set 精确滑动窗口：
        - 固定窗口：2 条命令，边界抖动在窗口末尾，对限流场景可接受
        - sorted-set：每次请求 2 条命令 + ZREMRANGEBYSCORE + ZCARD，
          写放大更大，且 TTL 管理复杂
        本场景不需要精确到毫秒，固定窗口够用。
        """
        key = f"rl:{ip}:{int(time.time()) // self.window_seconds}"
        count = await redis_client.incr(key, ttl=self.window_seconds)
        if count is None:
            return None  # Redis 不可用
        return count > self.requests_per_minute, count

    # ------------------------------------------------------------------
    # 内存降级路径
    # ------------------------------------------------------------------

    def _check_memory(self, ip: str) -> tuple[bool, int]:
        """本地内存滑动窗口（单机、有界）。

        OrderedDict 按访问顺序排列，满额时 popitem(last=False) 淘汰最久未访问的 IP。
        """
        now = time.time()
        cutoff = now - self.window_seconds

        # 若 IP 不存在且表已满，先腾出一格
        if ip not in self._mem_table and len(self._mem_table) >= _MAX_TRACKED_IPS:
            self._mem_table.popitem(last=False)

        if ip not in self._mem_table:
            self._mem_table[ip] = deque()
        else:
            # 把它移到末尾，保持"最近使用"在右边，"最久未用"留在左边
            self._mem_table.move_to_end(ip)

        times = self._mem_table[ip]

        # 清理超出窗口的旧记录
        while times and times[0] < cutoff:
            times.popleft()

        times.append(now)
        count = len(times)
        return count > self.requests_per_minute, count

    # ------------------------------------------------------------------
    # 请求处理
    # ------------------------------------------------------------------

    async def dispatch(self, request: Request, call_next):
        # OPTIONS 与系统端点不限流
        if request.method == "OPTIONS" or request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)

        ip = _get_client_ip(request, self.trusted_proxies)

        # 优先走 Redis；不可用时降级为内存
        result = await self._check_redis(ip)
        if result is None:
            is_limited, count = self._check_memory(ip)
        else:
            is_limited, count = result

        if is_limited:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": (
                        f"Too many requests. "
                        f"Limit: {self.requests_per_minute} requests "
                        f"per {self.window_seconds}s."
                    ),
                    "retry_after": self.window_seconds,
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - count)
        )
        return response
