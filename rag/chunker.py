"""
文本分块模块
提供文本分块功能，支持滑动窗口和重叠
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, size=None, overlap=None):
    """
    将文本分块，使用滑动窗口策略

    参数:
        text: str - 要分块的原始文本
        size: int - 每个块的大小（字符数），默认从 config 读取
        overlap: int - 重叠区域大小（字符数），默认从 config 读取

    返回:
        list[str] - 分块后的文本列表

    示例:
        >>> text = "人工智能是计算机科学的一个分支..." * 10
        >>> chunks = chunk_text(text, size=500, overlap=100)
        >>> print(f"生成了 {len(chunks)} 个块")
    """
    # 使用 config 中的默认值
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    chunks = []
    start = 0

    while start < len(text):
        # 计算结束位置，确保不超出文本长度
        end = min(start + size, len(text))

        # 提取当前块
        chunk = text[start:end]
        chunks.append(chunk)

        # 如果已到达文本末尾，退出循环
        if end >= len(text):
            break

        # 计算下一个块的起始位置（滑动窗口）
        start += size - overlap

    return chunks


def chunk_text_by_sentences(text, size=None, overlap=None, respect_sentences=True):
    """
    智能分块：尊重句子边界aaaaaa

    参数:
        text: str - 要分块的文本
        size: int - 目标块大小
        overlap: int - 重叠大小
        respect_sentences: bool - 是否尊重句子边界

    返回:
        list[str] - 分块后的文本列表
    """
    import re

    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    if not respect_sentences:
        return chunk_text(text, size, overlap)

    # 简单的句子分割（可以使用 nltk 或 spacy 获得更好的效果）
    sentences = re.split(r'[。！？.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 如果加入当前句子不会超过 size，就添加
        if len(current_chunk) + len(sentence) <= size:
            current_chunk += sentence + "。"
        else:
            # 保存当前块
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 开始新块，可能包含重叠
            if overlap > 0 and chunks:
                # 从上一块的末尾取 overlap 个字符作为重叠
                overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) >= overlap else chunks[-1]
                current_chunk = overlap_text + sentence + "。"
            else:
                current_chunk = sentence + "。"

    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


if __name__ == "__main__":
    # 测试代码
    test_text = """
    TechCorp 宠物政策：员工可以在每周五带宠物来办公室。
    宠物必须性格温顺且已接种疫苗。CEO 的金毛寻回犬是公司吉祥物。

    远程办公政策：员工每周最多可远程办公3天。
    核心工作时间为上午10点至下午3点。
    所有会议需要录制以便异步协作。
    """

    print("=" * 70)
    print("文本分块测试")
    print("=" * 70)

    print(f"\n原始文本长度: {len(test_text)} 字符\n")

    # 测试基本分块
    chunks = chunk_text(test_text, size=100, overlap=20)
    print(f"基本分块（size=100, overlap=20）:")
    print(f"  生成 {len(chunks)} 个块\n")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:40].replace('\n', ' ').strip()
        print(f"  块 {i} [{len(chunk):3d} 字符]: {preview}...")

    # 测试智能分块
    print(f"\n智能分块（尊重句子边界）:")
    smart_chunks = chunk_text_by_sentences(test_text, size=100, overlap=20)
    print(f"  生成 {len(smart_chunks)} 个块\n")

    for i, chunk in enumerate(smart_chunks, 1):
        preview = chunk[:40].replace('\n', ' ').strip()
        print(f"  块 {i} [{len(chunk):3d} 字符]: {preview}...")

    print("\n✓ 测试完成")
