"""
Redis 客户端 —— 带降级的缓存与协调层

**核心约定：Redis 不是事实来源。** 其中全部内容可丢弃后重建，
因此每个调用点都必须能在 Redis 不可用时继续工作：

    限流  → 降级为进程内有界计数（单机配额，不拒绝全部流量；见 rate_limit.py）
    缓存  → 降级为穿透（变慢，但结果正确）
    锁    → 降级为无锁（幂等性由业务逻辑保证，不依赖锁）

这不是"容错"这种事后补丁，而是设计前提。把 Redis 当强依赖的系统，
其可用性上限等于 Redis 的可用性。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.client import Redis
from redis.exceptions import RedisError

import config

logger = logging.getLogger(__name__)

_client: Optional[Redis] = None
# 连不上时打一次警告就够了，否则每个请求刷一行日志
_warned_unavailable = False


def get_client() -> Optional[Redis]:
    """获取 Redis 客户端。未启用或创建失败时返回 None。

    返回 Optional 而非抛异常：调用方需要的是"能用就用"，
    把判断放在这里，避免每个调用点都写 try/except。
    """
    global _client

    if not config.REDIS_ENABLED:
        return None

    if _client is None:
        try:
            _client = aioredis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                # 连不上时快速失败：默认无超时会让请求挂住，
                # 比"没有缓存"糟糕得多
                socket_connect_timeout=2,
                socket_timeout=2,
                # 探测空闲连接，避免拿到已被服务端关闭的连接
                health_check_interval=30,
                max_connections=20,
            )
        except Exception as exc:  # noqa: BLE001 - 创建失败也要降级
            logger.warning("Redis 客户端创建失败，降级运行: %s", exc)
            return None

    return _client


async def ping() -> bool:
    """探测可用性。供 /health 端点与启动日志使用。"""
    global _warned_unavailable

    client = get_client()
    if client is None:
        return False

    try:
        await client.ping()
        _warned_unavailable = False
        return True
    except RedisError as exc:
        if not _warned_unavailable:
            logger.warning("Redis 不可用，相关功能降级: %s", exc)
            _warned_unavailable = True
        return False


async def close() -> None:
    """关闭连接池。在应用 lifespan 结束时调用。"""
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:  # noqa: BLE001 - 关闭失败不影响退出
            logger.debug("Redis 关闭时出错，忽略", exc_info=True)
        _client = None
        logger.info("Redis 连接已关闭")


# ============================================================
# 基础读写（全部静默降级）
# ============================================================

async def get_json(key: str) -> Optional[Any]:
    """读 JSON 值。不可用或未命中都返回 None。

    两种情况都返回 None 是故意的：调用方的处理方式相同
    （去算一遍），区分它们只会让调用点变复杂。
    需要区分时用 ping()。
    """
    client = get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        return json.loads(raw) if raw else None
    except (RedisError, json.JSONDecodeError) as exc:
        # JSONDecodeError 说明缓存里是脏数据（格式变更后的旧值），
        # 当作未命中处理，新值写入时会覆盖它
        logger.debug("读缓存失败 key=%s: %s", key, exc)
        return None


async def set_json(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """写 JSON 值。返回是否成功。

    **必须带 TTL**：缓存无过期会持续增长直到打满内存，
    而 Redis 默认策略 noeviction 下写入会开始报错。
    ttl=None 时用配置的默认值，不会真的无过期。
    """
    client = get_client()
    if client is None:
        return False
    try:
        payload = json.dumps(value, ensure_ascii=False)
        await client.set(key, payload, ex=ttl or config.CACHE_TTL_RETRIEVAL)
        return True
    except (RedisError, TypeError) as exc:
        # TypeError 说明 value 含不可序列化对象，是调用方的 bug，
        # 但不该因此让请求失败 —— 记日志后继续
        logger.debug("写缓存失败 key=%s: %s", key, exc)
        return False


async def incr(key: str, ttl: Optional[int] = None) -> Optional[int]:
    """自增并返回新值。不可用时返回 None。

    首次自增时设置 TTL。用 pipeline 把 INCR 与 EXPIRE 打包，
    减少一次往返；两条命令不需要原子性 —— 即使 EXPIRE 丢失，
    下一个窗口的 INCR 会再设一次。
    """
    client = get_client()
    if client is None:
        return None
    try:
        if ttl is None:
            return await client.incr(key)
        async with client.pipeline(transaction=False) as pipe:
            pipe.incr(key)
            pipe.expire(key, ttl)
            results = await pipe.execute()
        return results[0]
    except RedisError as exc:
        logger.debug("INCR 失败 key=%s: %s", key, exc)
        return None


async def get_int(key: str, default: int = 0) -> int:
    """读整数值。不可用或未设置时返回 default。"""
    client = get_client()
    if client is None:
        return default
    try:
        raw = await client.get(key)
        return int(raw) if raw is not None else default
    except (RedisError, ValueError):
        return default


async def delete(*keys: str) -> int:
    """删除 key，返回删除数量。"""
    client = get_client()
    if client is None or not keys:
        return 0
    try:
        return await client.delete(*keys)
    except RedisError as exc:
        logger.debug("DELETE 失败: %s", exc)
        return 0


async def hset_mapping(key: str, mapping: dict, ttl: Optional[int] = None) -> bool:
    """写 hash（用于摄入进度）。值会被转成字符串。"""
    client = get_client()
    if client is None:
        return False
    try:
        flat = {k: (json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list)) else str(v))
                for k, v in mapping.items()}
        async with client.pipeline(transaction=False) as pipe:
            pipe.hset(key, mapping=flat)
            if ttl:
                pipe.expire(key, ttl)
            await pipe.execute()
        return True
    except RedisError as exc:
        logger.debug("HSET 失败 key=%s: %s", key, exc)
        return False


async def hgetall(key: str) -> dict:
    """读 hash。不可用或不存在时返回空字典。"""
    client = get_client()
    if client is None:
        return {}
    try:
        return await client.hgetall(key) or {}
    except RedisError:
        return {}


# ============================================================
# 知识库版本（缓存失效机制）
# ============================================================

KB_VERSION_KEY = "kb_version"


async def get_kb_version() -> int:
    """当前知识库版本。

    这个值嵌进检索缓存的 key，文档一变就自增，
    旧 key 因前缀不匹配自然失效 —— 不必遍历删除。

    为什么不遍历删：KEYS 会阻塞整个 Redis（单线程模型），
    在生产是禁用操作；SCAN 不阻塞但仍是 O(n)，
    且删除期间新写入的缓存可能又用了旧版本号。
    版本号方案把失效变成 O(1) 且无竞态。
    """
    return await get_int(KB_VERSION_KEY, default=0)


async def bump_kb_version() -> Optional[int]:
    """文档增删改后调用，使全部检索缓存失效。"""
    version = await incr(KB_VERSION_KEY)
    if version is not None:
        logger.info("知识库版本自增至 %s，检索缓存已失效", version)
    return version
