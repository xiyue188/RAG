"""
环境变量验证
启动时检查必需的配置
"""

import os
import sys


PLACEHOLDER_VALUES = {
    "",
    "your-api-key-here",
    "sk-your-openai-api-key-here",
    "sk-your-openai-key",
    "sk-your-openai-key-here",
    "sk-ant-your-anthropic-api-key-here",
    "your-zhipu-api-key-here",
    "your-zhipu-api-key",
    "your-key",
    "your-key-here",
    "sk-your-qwen-api-key-here",
}


def _has_real_value(value: str | None) -> bool:
    """判断环境变量是否填了真实值，而不是模板占位符。"""
    if value is None:
        return False
    normalized = value.strip()
    return bool(normalized) and normalized.lower() not in PLACEHOLDER_VALUES


def validate_env():
    """
    验证必需的环境变量

    如果缺少必需配置，打印错误并退出
    """
    errors = []

    # 检查 LLM 提供商
    llm_provider = os.getenv("LLM_PROVIDER", "zhipu")
    if not llm_provider:
        errors.append("LLM_PROVIDER 未设置")
    else:
        # 根据提供商检查对应的 API Key
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "zhipu": "ZHIPU_API_KEY",
            "qwen": "QWEN_API_KEY",
        }

        if llm_provider not in provider_keys:
            errors.append(f"不支持的 LLM_PROVIDER: {llm_provider}")
            errors.append(f"   支持的提供商: {', '.join(provider_keys.keys())}")
        else:
            key_name = provider_keys[llm_provider]
            if not _has_real_value(os.getenv(key_name)):
                errors.append(f"{key_name} 未设置")
                errors.append(f"   LLM_PROVIDER 设置为 {llm_provider}，但缺少对应的 API Key")

    # 如果有错误，打印并退出
    if errors:
        print("\n" + "=" * 60)
        print("🚫 环境变量验证失败")
        print("=" * 60)
        for error in errors:
            print(error)
        print("\n解决方法:")
        print("1. 复制 .env.example 为 .env")
        print("   cp .env.example .env")
        print("\n2. 编辑 .env 文件，填入您的配置")
        print("\n3. 确保设置了 LLM_PROVIDER 和对应的 API Key")
        print("=" * 60 + "\n")
        sys.exit(1)

    # 成功
    print("环境变量验证通过")


def get_env_info() -> dict:
    """
    获取环境配置信息（用于显示）

    返回:
        配置信息字典
    """
    llm_provider = os.getenv("LLM_PROVIDER", "unknown")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # 检查是否配置了 API Key（但不显示值）
    provider_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "zhipu": "ZHIPU_API_KEY",
        "qwen": "QWEN_API_KEY",
    }
    api_key_configured = _has_real_value(os.getenv(provider_keys.get(llm_provider, "")))

    return {
        "llm_provider": llm_provider,
        "log_level": log_level,
        "api_key_configured": api_key_configured,
    }
