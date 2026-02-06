"""
统一日志系统
提供全局logger配置和敏感信息过滤
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import re

# 敏感信息过滤规则
SENSITIVE_PATTERNS = [
    (r'(api[_-]?key["\s:=]+)([a-zA-Z0-9\-_]{20,})', r'\1****'),  # API Key
    (r'(sk-[a-zA-Z0-9]{20,})', r'sk-****'),                      # OpenAI Key
    (r'(Bearer\s+)([a-zA-Z0-9\-_\.]{20,})', r'\1****'),         # Bearer Token
    (r'([a-f0-9]{32,})', r'****'),                              # 长hash值（可能是key）
]


class SensitiveDataFilter(logging.Filter):
    """过滤日志中的敏感信息"""

    def filter(self, record):
        # 过滤message中的敏感信息
        for pattern, replacement in SENSITIVE_PATTERNS:
            record.msg = re.sub(pattern, replacement, str(record.msg))
        return True


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True
) -> logging.Logger:
    """
    设置模块logger

    参数:
        name: logger名称（通常是模块名）
        level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_file: 日志文件路径（可选）
        enable_console: 是否输出到控制台

    返回:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 统一日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 添加敏感信息过滤器
    sensitive_filter = SensitiveDataFilter()

    # 控制台输出
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(sensitive_filter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sensitive_filter)
        logger.addHandler(file_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    获取模块logger（简化调用）

    使用方式：
        from rag.logger import get_logger
        logger = get_logger(__name__)
        logger.info("消息")
    """
    # 从环境变量读取配置（可选）
    import os
    level = os.getenv("LOG_LEVEL", "INFO")
    log_dir = os.getenv("LOG_DIR", "")

    # 生成日志文件路径
    log_file = None
    if log_dir:
        log_file = f"{log_dir}/rag.log"

    return setup_logger(module_name, level, log_file)


def mask_secret(text: str, show_chars: int = 4) -> str:
    """
    掩码敏感信息（用于显示）

    示例：
        mask_secret("sk-1234567890abcdef")
        → "sk-1****cdef"
    """
    if not text or len(text) <= show_chars * 2:
        return "****"
    return f"{text[:show_chars]}****{text[-show_chars:]}"


# 模块级别的logger（用于本模块）
_module_logger = get_logger(__name__)


if __name__ == "__main__":
    # 测试日志系统
    print("=" * 70)
    print("日志系统测试")
    print("=" * 70)

    # 创建测试logger
    logger = get_logger("test")

    # 测试不同级别
    logger.debug("这是DEBUG消息")
    logger.info("这是INFO消息")
    logger.warning("这是WARNING消息")
    logger.error("这是ERROR消息")

    # 测试敏感信息过滤
    print("\n敏感信息过滤测试:")
    logger.info("API Key: sk-1234567890abcdefghijklmnopqrstuvwxyz")
    logger.info("Bearer Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
    logger.info("配置: api_key=d511db972b5d4c2e9c9672e3c7120109")

    # 测试掩码函数
    print("\n掩码测试:")
    print(f"原始: sk-1234567890abcdef")
    print(f"掩码: {mask_secret('sk-1234567890abcdef')}")

    print("\n测试完成")
