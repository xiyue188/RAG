"""T4 限流测试

覆盖三个问题的修复：

1. **内存有界**：降级表不超过 `_MAX_TRACKED_IPS`，按 LRU 淘汰
2. **XFF 校验**：只有对端是可信代理时才采信 x-forwarded-for
3. **Redis 降级**：Redis 不可用时退回内存计数，仍然限流，不拒绝全部流量

## 两条此前写错的测试（M2 收口时重写）

- `test_memory_table_bounded` 原先用一个不存在的请求头"伪造 IP"，
  实际所有请求同一个 IP，表里永远 1 条，断言必然通过 —— 什么都没测。
- `_make_app` 原先临时改全局 config、`add_middleware` 后立刻改回。
  但 Starlette 在**第一次请求时**才实例化中间件，参数从未生效；
  现有用例恰好都用默认值所以是绿的。改为构造参数注入。
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.rate_limit as rate_limit
from backend.rate_limit import RateLimitMiddleware, _get_client_ip

REDIS_DOWN = "backend.rate_limit.redis_client.incr"


def _make_app(
    requests_per_minute: int = 3,
    window_seconds: int = 60,
    trusted_proxies: list[str] | None = None,
) -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=requests_per_minute,
        window_seconds=window_seconds,
        trusted_proxies=trusted_proxies or [],
    )
    return app


def _middleware(requests_per_minute: int = 5, window_seconds: int = 60) -> RateLimitMiddleware:
    return RateLimitMiddleware(
        app=None,
        requests_per_minute=requests_per_minute,
        window_seconds=window_seconds,
        trusted_proxies=[],
    )


class _Req:
    """`_get_client_ip` 只读 client.host 与 headers。"""

    def __init__(self, host: str, xff: str | None = None):
        self.client = type("C", (), {"host": host})()
        self.headers = {"x-forwarded-for": xff} if xff else {}


# ============================================================
# HTTP 行为
# ============================================================

def test_health_path_exempt():
    """系统路径不限流：容器健康检查不能被 429 打成"不健康"。"""
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=None):
        client = TestClient(_make_app(requests_per_minute=1))
        assert all(client.get("/health").status_code == 200 for _ in range(5))


def test_over_quota_returns_429_with_injected_window():
    """超额返回 429，Retry-After 等于注入的窗口长度（证明参数确实生效）。"""
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=None):
        client = TestClient(_make_app(requests_per_minute=2, window_seconds=30))
        assert client.get("/ping").status_code == 200
        assert client.get("/ping").status_code == 200
        r = client.get("/ping")
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "30"
    assert r.json()["retry_after"] == 30


def test_remaining_header():
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=1):
        r = TestClient(_make_app(requests_per_minute=5)).get("/ping")
    assert r.headers["X-RateLimit-Remaining"] == "4"


def test_spoofed_xff_ignored_when_peer_not_trusted():
    """对端不是可信代理：每个请求伪造不同 XFF 也绕不过限流。"""
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=None):
        client = TestClient(_make_app(requests_per_minute=2))
        codes = [
            client.get("/ping", headers={"x-forwarded-for": f"9.9.9.{i}"}).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 429]


def test_xff_honored_when_peer_is_trusted_proxy():
    """对端是可信代理：按 XFF 里的原始客户端分别计数。

    TestClient 的对端地址是 "testclient"，把它配成可信代理即可端到端验证。
    """
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=None):
        client = TestClient(_make_app(requests_per_minute=2, trusted_proxies=["testclient"]))
        codes = [
            client.get("/ping", headers={"x-forwarded-for": f"9.9.9.{i}"}).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 200]


# ============================================================
# 内存降级表：有界 + LRU
# ============================================================

def test_memory_table_bounded_at_cap():
    """灌入超过上限的不同 IP，表大小停在上限，最早的那批被淘汰。"""
    mw = _middleware(requests_per_minute=1000)
    cap = rate_limit._MAX_TRACKED_IPS
    for i in range(cap + 10):
        mw._check_memory(f"ip-{i}")

    assert len(mw._mem_table) == cap
    assert "ip-0" not in mw._mem_table
    assert "ip-9" not in mw._mem_table
    assert "ip-10" in mw._mem_table
    assert f"ip-{cap + 9}" in mw._mem_table


def test_memory_table_evicts_least_recently_used(monkeypatch):
    """淘汰的是最久未访问的，不是最早插入的。

    正常用户 a 先来、一直活跃，扫描器的新 IP 不能把它挤掉 ——
    挤掉意味着它的计数清零，等于给它重置了配额。
    """
    monkeypatch.setattr(rate_limit, "_MAX_TRACKED_IPS", 3)
    mw = _middleware(requests_per_minute=1000)
    for ip in ("a", "b", "c"):
        mw._check_memory(ip)
    mw._check_memory("a")          # a 变成最近使用
    mw._check_memory("scanner-1")  # 满了，应淘汰 b 而不是 a

    assert list(mw._mem_table) == ["c", "a", "scanner-1"]


def test_memory_window_expires():
    """超出窗口的记录被清理，计数重置。"""
    mw = _middleware(requests_per_minute=2, window_seconds=1)
    assert [mw._check_memory("x")[0] for _ in range(3)] == [False, False, True]
    time.sleep(1.1)
    is_limited, count = mw._check_memory("x")
    assert not is_limited and count == 1


# ============================================================
# Redis 路径
# ============================================================

@pytest.mark.asyncio
@pytest.mark.parametrize("count, limited", [(3, False), (5, False), (6, True)])
async def test_redis_path(count, limited):
    """count ≤ 配额放行，> 配额拦截（边界值 5 放行）。"""
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=count):
        assert await _middleware(requests_per_minute=5)._check_redis("1.2.3.4") == (limited, count)


@pytest.mark.asyncio
async def test_redis_unavailable_returns_none():
    """incr 返回 None 即 Redis 不可用，交给调用方降级。"""
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=None):
        assert await _middleware()._check_redis("1.2.3.4") is None


@pytest.mark.asyncio
async def test_redis_key_is_per_window():
    """key 含窗口序号：同一窗口内同一 key，跨窗口换 key（固定窗口语义）。"""
    with patch(REDIS_DOWN, new_callable=AsyncMock, return_value=1) as incr:
        await _middleware(window_seconds=60)._check_redis("1.2.3.4")
    key = incr.await_args.args[0]
    assert key == f"rl:1.2.3.4:{int(time.time()) // 60}"
    assert incr.await_args.kwargs["ttl"] == 60


# ============================================================
# _get_client_ip
# ============================================================

def test_xff_ignored_without_trusted_proxy():
    assert _get_client_ip(_Req("192.168.1.1", "1.2.3.4"), frozenset()) == "192.168.1.1"


def test_xff_takes_leftmost_when_trusted():
    req = _Req("10.0.0.2", "5.6.7.8, 10.0.0.1, 10.0.0.2")
    assert _get_client_ip(req, frozenset({"10.0.0.2"})) == "5.6.7.8"


def test_trusted_proxy_without_xff_uses_peer():
    assert _get_client_ip(_Req("10.0.0.1"), frozenset({"10.0.0.1"})) == "10.0.0.1"
