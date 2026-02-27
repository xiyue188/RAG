"""
文本分块模块
提供文本分块功能，支持滑动窗口、语义感知切分
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text, size=None, overlap=None):
    """
    将文本分块，使用滑动窗口策略（基础方法）

    参数:
        text: str - 要分块的原始文本
        size: int - 每个块的大小（字符数），默认从 config 读取
        overlap: int - 重叠区域大小（字符数），默认从 config 读取

    返回:
        list[str] - 分块后的文本列表
    """
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += size - overlap

    return chunks


def chunk_markdown_by_headers(text: str, size: int = None, overlap: int = None) -> List[str]:
    """
    🎯 最佳实践：基于Markdown标题层级的语义切分

    策略：
    1. 按 ## 二级标题切分为 section
    2. 对每个 section，始终按 ### 三级标题切分（不论大小）
    3. 如果某个 subsection 仍太大，按段落切分
    4. 每个 chunk 都包含祖先标题作为上下文

    参数:
        text: str - Markdown文本
        size: int - 目标块大小
        overlap: int - 重叠大小

    返回:
        List[str] - 语义连贯的文本块列表
    """
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    chunks = []

    # 提取文档标题（一级标题）
    doc_title_match = re.search(r'^#\s+(.+?)$', text, re.MULTILINE)
    doc_title = doc_title_match.group(0) if doc_title_match else ""

    # 按二级标题分割
    sections = re.split(r'\n(?=##\s+)', text)

    for section in sections:
        if not section.strip():
            continue

        # 提取 section 的二级标题
        section_title_match = re.search(r'^##\s+(.+?)$', section, re.MULTILINE)
        section_title = section_title_match.group(0) if section_title_match else ""

        # 始终尝试按 ### 三级标题切分（不论 section 大小）
        subsections = re.split(r'\n(?=###\s+)', section)

        # 如果只有一个 subsection（没有 ### 标题），直接处理
        if len(subsections) == 1:
            subsection = subsections[0].strip()
            if not subsection:
                continue

            # 足够小，直接添加
            if len(subsection) <= size:
                chunks.append(subsection)
                continue

            # 太大，按段落切分
            _split_by_paragraphs(subsection, doc_title, section_title, size, chunks)
            continue

        # 有 ### 子标题：每个 subsection 单独成 chunk
        for subsection in subsections:
            if not subsection.strip():
                continue

            # 判断是否是 ### 标题行开头的 subsection
            has_h3 = bool(re.match(r'^###\s+', subsection.strip()))

            if has_h3:
                # 带 ### 标题的 subsection：添加祖先标题上下文
                context_prefix = ""
                if doc_title and doc_title not in subsection:
                    context_prefix += doc_title + "\n\n"
                if section_title and section_title not in subsection:
                    context_prefix += section_title + "\n\n"
                chunk = (context_prefix + subsection).strip()
            else:
                # ## 标题本身的内容（### 之前的部分）
                chunk = subsection.strip()

            if not chunk:
                continue

            # 如果 chunk 太大，按段落切分
            if len(chunk) > size:
                _split_by_paragraphs(chunk, doc_title, section_title, size, chunks)
            else:
                chunks.append(chunk)

    return chunks


def _split_by_paragraphs(text: str, doc_title: str, section_title: str,
                          size: int, chunks: List[str]) -> None:
    """辅助函数：将大块文本按段落切分，追加到 chunks 列表"""
    paragraphs = text.split('\n\n')
    header_prefix = ""
    if doc_title and doc_title not in text[:len(doc_title) + 5]:
        header_prefix += doc_title + "\n\n"
    if section_title and section_title not in text[:len(section_title) + 50]:
        header_prefix += section_title + "\n\n"

    current_chunk = header_prefix

    for para in paragraphs:
        if not para.strip():
            continue
        if len(current_chunk) + len(para) + 2 <= size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = header_prefix + para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())


def chunk_text_by_paragraphs(text: str, size: int = None, overlap: int = None) -> List[str]:
    """
    按段落边界切分纯文本

    参数:
        text: str - 纯文本
        size: int - 目标块大小
        overlap: int - 重叠大小

    返回:
        List[str] - 文本块列表
    """
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    # 按双换行符分割段落
    paragraphs = re.split(r'\n\n+', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # 如果单个段落就超过size，按句子切分
        if len(para) > size:
            # 保存当前chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 切分大段落
            sentences = re.split(r'([。！？.!?]+)', para)
            sentence_with_punct = []
            for i in range(0, len(sentences) - 1, 2):
                sentence_with_punct.append(sentences[i] + (sentences[i+1] if i+1 < len(sentences) else ""))

            for sent in sentence_with_punct:
                if len(current_chunk) + len(sent) <= size:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent

            continue

        # 正常段落处理
        if len(current_chunk) + len(para) + 2 <= size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    # 添加最后一个chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_text_semantic(text: str, size: int = None, overlap: int = None) -> List[str]:
    """
    🎯 智能语义切分（推荐使用）：自动检测Markdown或纯文本

    参数:
        text: str - 输入文本
        size: int - 目标块大小
        overlap: int - 重叠大小

    返回:
        List[str] - 语义连贯的文本块
    """
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    # 检测是否为Markdown
    has_markdown_headers = bool(re.search(r'^#+\s+', text, re.MULTILINE))

    if has_markdown_headers:
        return chunk_markdown_by_headers(text, size, overlap)
    else:
        return chunk_text_by_paragraphs(text, size, overlap)


def chunk_with_metadata(text: str, size: int = None, overlap: int = None) -> List[Tuple[str, Dict]]:
    """
    🎯 SOTA 方案：切块同时返回元数据（标题层级上下文）

    每个返回项为 (chunk_text, metadata_dict)，其中 metadata 包含：
      - header: 当前 chunk 所属标题（如 "### 允许时间" 或 "## 第78条"）
      - h1: 一级标题
      - h2: 二级标题（若有）
      - h3: 三级标题（若有）

    参数:
        text: str - 输入文本
        size: int - 目标块大小
        overlap: int - 重叠大小

    返回:
        List[Tuple[str, Dict]] - [(chunk文本, 元数据字典), ...]
    """
    if size is None:
        size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP

    has_markdown_headers = bool(re.search(r'^#+\s+', text, re.MULTILINE))

    if not has_markdown_headers:
        # 纯文本：按段落切，元数据为空
        chunks = chunk_text_by_paragraphs(text, size, overlap)
        return [(c, {}) for c in chunks]

    results: List[Tuple[str, Dict]] = []

    # 提取一级标题
    doc_title_match = re.search(r'^#\s+(.+?)$', text, re.MULTILINE)
    h1 = doc_title_match.group(1).strip() if doc_title_match else ""
    doc_title_line = doc_title_match.group(0) if doc_title_match else ""

    # 按二级标题分割
    sections = re.split(r'\n(?=##\s+)', text)

    for section in sections:
        if not section.strip():
            continue

        section_title_match = re.search(r'^##\s+(.+?)$', section, re.MULTILINE)
        h2 = section_title_match.group(1).strip() if section_title_match else ""
        section_title_line = section_title_match.group(0) if section_title_match else ""

        subsections = re.split(r'\n(?=###\s+)', section)

        if len(subsections) == 1:
            # 无 ### 子标题，整个 section 为一个 chunk
            chunk_text = subsections[0].strip()
            if not chunk_text:
                continue
            meta = {"h1": h1, "h2": h2, "h3": "", "header": h2 or h1}
            if len(chunk_text) <= size:
                results.append((chunk_text, meta))
            else:
                for sub in _split_to_list(chunk_text, doc_title_line, section_title_line, size):
                    results.append((sub, meta))
            continue

        # 有 ### 子标题：每个 subsection 独立
        for subsection in subsections:
            if not subsection.strip():
                continue

            h3_match = re.match(r'^###\s+(.+?)$', subsection.strip(), re.MULTILINE)
            h3 = h3_match.group(1).strip() if h3_match else ""
            has_h3 = bool(h3_match)

            # 决定 header 显示最精确的标题
            header = h3 or h2 or h1
            meta = {"h1": h1, "h2": h2, "h3": h3, "header": header}

            if has_h3:
                context_prefix = ""
                if doc_title_line and doc_title_line not in subsection:
                    context_prefix += doc_title_line + "\n\n"
                if section_title_line and section_title_line not in subsection:
                    context_prefix += section_title_line + "\n\n"
                chunk_text = (context_prefix + subsection).strip()
            else:
                chunk_text = subsection.strip()

            if not chunk_text:
                continue

            if len(chunk_text) <= size:
                results.append((chunk_text, meta))
            else:
                for sub in _split_to_list(chunk_text, doc_title_line, section_title_line, size):
                    results.append((sub, meta))

    return results


def _split_to_list(text: str, doc_title: str, section_title: str, size: int) -> List[str]:
    """辅助：大块按段落切分，返回列表"""
    out: List[str] = []
    _split_by_paragraphs(text, doc_title, section_title, size, out)
    return out
