"""
测试敏感信息过滤
"""

import sys
import io
from pathlib import Path

# 解决编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent))

from rag.logger import get_logger

def test_sensitive_filter():
    """测试敏感信息过滤"""
    print("=" * 70)
    print("敏感信息过滤测试")
    print("=" * 70)

    logger = get_logger("test_security")

    # 测试各类敏感信息
    test_cases = [
        ("API Key OpenAI", "配置: OPENAI_API_KEY=sk-1234567890abcdefghijklmnopqrstuvwxyz"),
        ("API Key Zhipu", "智谱AI: api_key=d511db972b5d4c2e9c9672e3c7120109"),
        ("Bearer Token", "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"),
        ("长Hash", "Session ID: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"),
    ]

    print("\n测试用例：")
    for name, message in test_cases:
        print(f"\n{name}:")
        print(f"  原始: {message}")
        print(f"  日志: ", end="")
        logger.info(message)

    print("\n" + "=" * 70)
    print("测试完成！如果上面的日志中敏感信息被替换为****，说明过滤成功")
    print("=" * 70)

if __name__ == "__main__":
    test_sensitive_filter()
