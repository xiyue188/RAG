"""
可选 API Key 认证
- 无 Key：基础访问（每分钟10次）
- 有 Key：高级访问（每分钟30次）
"""

import os
from fastapi import Security, Request
from fastapi.security import APIKeyHeader
from typing import Optional

# API Key Header（可选）
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# 从环境变量读取 API Key
VALID_API_KEYS = set()
if api_key := os.getenv("API_KEY"):
    VALID_API_KEYS.add(api_key)

# 支持多个 API Key（逗号分隔）
if api_keys_str := os.getenv("API_KEYS"):
    VALID_API_KEYS.update(api_keys_str.split(","))


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    获取 API Key（可选）

    返回:
        - None: 普通用户（基础限流）
        - str: 有效的 API Key（高级限流）
    """
    if api_key and api_key in VALID_API_KEYS:
        return api_key
    return None


def is_authenticated(request: Request) -> bool:
    """检查请求是否已认证（用于速率限制分级）"""
    return hasattr(request.state, "api_key") and request.state.api_key is not None


def get_rate_limit_key(request: Request) -> str:
    """
    获取速率限制的 key

    策略:
        - 有API Key: 使用 api_key 作为标识
        - 无API Key: 使用 IP 地址作为标识
    """
    if is_authenticated(request):
        return f"user:{request.state.api_key}"
    return f"ip:{request.client.host}"
