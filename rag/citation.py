"""
引用追踪模块
实现句级引用和来源标注
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import re

try:
    from .logger import get_logger
except ImportError:
    # 测试时使用绝对导入
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from rag.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Citation:
    """单条引用"""
    sentence: str      # 答案句子
    source_doc: str    # 来源文档编号（如doc_1）
    source_file: str   # 来源文件名
    confidence: float = 1.0  # 置信度


class CitationManager:
    """引用管理器"""

    def build_citation_prompt(
        self,
        question: str,
        documents: List[Dict],
        conversation_context: Optional[str] = None,
        mode: str = "inline"
    ) -> str:
        """
        构建带引用的prompt

        参数:
            question: 用户问题
            documents: 检索到的文档列表
            conversation_context: 对话历史（可选）
            mode: 引用模式 - "inline"内联标记（推荐，支持流式）或 "json"（遗留）

        返回:
            构建好的prompt字符串
        """
        # 输入验证
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        if not documents:
            raise ValueError("文档列表不能为空")

        # 格式化文档，添加编号
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            meta = doc['metadata']
            doc_texts.append(
                f"[doc_{i}] 来源: {meta.get('file', 'unknown')}\n"
                f"内容: {doc['document']}"
            )

        docs_str = "\n\n".join(doc_texts)

        # 根据模式构建不同的prompt
        if mode == "inline":
            # 内联标记模式（推荐）- 支持流式输出
            prompt = f"""请根据以下文档回答问题。在引用文档内容时，在句子末尾添加来源标记 [doc_X]。

文档：
{docs_str}

问题：{question}

回答要求：
1. 直接用自然语言回答，不要使用JSON格式
2. 在每句话或每个观点后面，用 [doc_1]、[doc_2] 等标记来源
3. 标记要紧跟在引用内容之后，放在句号之前
4. 如果一句话引用多个文档，可以写 [doc_1][doc_2]
5. 只标记有明确来源支持的内容

示例格式：
TechCorp允许员工每周五携带宠物来办公室[doc_1]。宠物必须性格温顺且已接种疫苗[doc_2]。

请回答："""

        else:
            # JSON模式（遗留，不支持流式）
            prompt = f"""请根据以下文档回答问题，并为每个答案片段标注来源。

文档：
{docs_str}

问题：{question}

请以JSON格式返回，将答案分成多个片段，每个片段标注来源：
{{
  "segments": [
    {{"text": "答案的第一句或第一段。", "source": "doc_1"}},
    {{"text": "答案的第二句或第二段。", "source": "doc_2"}}
  ]
}}

要求：
1. segments数组包含所有答案片段（按顺序）
2. text是完整流畅的句子或段落
3. source是文档编号（doc_1, doc_2等）
4. 每个片段必须有明确的来源支持
5. 拼接所有text应构成完整答案
6. 确保返回有效的JSON格式"""

        # 如果有对话历史，添加到prompt开头
        if conversation_context:
            prompt = f"{conversation_context}\n\n{prompt}"

        return prompt

    def parse_inline_citations(
        self,
        response: str,
        documents: List[Dict]
    ) -> Dict:
        """
        解析内联标记格式的引用（新方法，支持流式）

        参数:
            response: LLM返回的带 [doc_X] 标记的文本
            documents: 原始文档列表

        返回:
            包含answer、citations、formatted_answer的字典
        """
        # 输入验证
        if not response:
            logger.warning("响应为空")
            return {
                'answer': '',
                'citations': [],
                'formatted_answer': '',
                'parse_success': False,
                'error': '响应为空'
            }

        if not documents:
            logger.warning("文档列表为空")
            return {
                'answer': response,
                'citations': [],
                'formatted_answer': response,
                'parse_success': False,
                'error': '文档列表为空'
            }

        # 映射doc_id到实际文档
        doc_map = {
            f"doc_{i+1}": doc
            for i, doc in enumerate(documents)
        }

        # 提取所有引用标记 [doc_X]
        citation_pattern = r'\[doc_(\d+)\]'
        citations = []
        cited_docs = set()

        # 找到所有引用
        for match in re.finditer(citation_pattern, response):
            doc_id = f"doc_{match.group(1)}"
            if doc_id in doc_map:
                cited_docs.add(doc_id)

        # 创建引用对象（每个文档一个）
        for doc_id in sorted(cited_docs):
            doc = doc_map[doc_id]
            citations.append(Citation(
                sentence="",  # 内联模式下不需要具体句子
                source_doc=doc_id,
                source_file=doc['metadata'].get('file', 'unknown'),
                confidence=1.0
            ))

        # 格式化答案：将 [doc_X] 替换为更友好的显示
        def replace_citation(match):
            doc_id = f"doc_{match.group(1)}"
            if doc_id in doc_map:
                file_name = doc_map[doc_id]['metadata'].get('file', 'unknown')
                return f" [来源: {file_name}]"
            return match.group(0)

        formatted_answer = re.sub(citation_pattern, replace_citation, response)

        result = {
            'answer': response,  # 保留原始答案（带[doc_X]标记）
            'citations': citations,
            'formatted_answer': formatted_answer,  # 格式化后的答案（[来源: xxx.md]）
            'parse_success': True,
            'cited_count': len(cited_docs)
        }

        logger.info(f"内联引用解析成功：引用了 {len(cited_docs)} 个文档")
        return result

    def parse_citation_response(
        self,
        response: str,
        documents: List[Dict],
        mode: str = "inline"
    ) -> Dict:
        """
        解析LLM返回的引用响应（支持多种模式）

        参数:
            response: LLM返回的文本
            documents: 原始文档列表
            mode: 解析模式 - "inline"内联标记（推荐）或 "json"（遗留）

        返回:
            包含answer和citations的字典
        """
        if mode == "inline":
            # 使用内联标记解析
            return self.parse_inline_citations(response, documents)
        else:
            # 使用JSON解析（遗留模式）
            return self._parse_json_citations(response, documents)

    def _parse_json_citations(
        self,
        response: str,
        documents: List[Dict]
    ) -> Dict:
        """
        解析JSON格式的引用（遗留方法，不支持流式）

        参数:
            response: LLM返回的JSON字符串
            documents: 原始文档列表

        返回:
            包含answer和citations的字典
        """
        # 输入验证
        if not response:
            logger.warning("响应为空")
            return {
                'answer': '',
                'citations': [],
                'raw_response': response,
                'parse_success': False,
                'error': '响应为空'
            }

        if not documents:
            logger.warning("文档列表为空")
            return {
                'answer': response,
                'citations': [],
                'raw_response': response,
                'parse_success': False,
                'error': '文档列表为空'
            }

        try:
            # 尝试提取JSON（处理LLM可能返回额外文本的情况）
            # 方法1: 尝试直接解析
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # 方法2: 查找第一个 { 到最后一个 } 之间的内容
                first_brace = response.find('{')
                last_brace = response.rfind('}')

                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str = response[first_brace:last_brace + 1]
                    data = json.loads(json_str)
                else:
                    raise ValueError("未找到有效的JSON结构")

            # 映射doc_id到实际文档
            doc_map = {
                f"doc_{i+1}": doc
                for i, doc in enumerate(documents)
            }

            # 解析 segments 结构（新格式）
            segments = data.get('segments', [])
            citations = []
            answer_parts = []

            for segment in segments:
                text = segment.get('text', '').strip()
                doc_id = segment.get('source', '')

                if text:
                    answer_parts.append(text)

                    if doc_id in doc_map:
                        doc = doc_map[doc_id]
                        citations.append(Citation(
                            sentence=text,
                            source_doc=doc_id,
                            source_file=doc['metadata'].get('file', 'unknown'),
                            confidence=segment.get('confidence', 1.0)
                        ))

            # 拼接完整答案（用换行或空格连接）
            answer = '\n'.join(answer_parts) if answer_parts else ''

            result = {
                'answer': answer,
                'citations': citations,
                'raw_response': response,
                'parse_success': True
            }

            logger.info(f"引用解析成功：{len(citations)} 个片段")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}，返回原始文本")
            # JSON解析失败，返回原始文本
            return {
                'answer': response,
                'citations': [],
                'raw_response': response,
                'parse_success': False,
                'error': f'JSON解析失败: {str(e)}'
            }
        except Exception as e:
            logger.error(f"引用解析异常: {e}", exc_info=True)
            return {
                'answer': response,
                'citations': [],
                'raw_response': response,
                'parse_success': False,
                'error': str(e)
            }

    def format_answer_with_citations(
        self,
        answer: str,
        citations: List[Citation],
        style: str = "inline"
    ) -> str:
        """
        格式化带引用标注的答案

        参数:
            answer: 原始答案
            citations: 引用列表
            style: 格式化风格（inline/footnote）

        返回:
            格式化后的答案字符串
        """
        if not citations:
            return answer

        if style == "inline":
            return self._format_inline_citations(answer, citations)
        elif style == "footnote":
            return self._format_footnote_citations(answer, citations)
        else:
            return answer

    def _format_inline_citations(self, answer: str, citations: List[Citation]) -> str:
        """
        内联引用格式：每个片段后直接标注来源（改进版：无需模糊匹配）

        由于新格式下 citations 直接包含文本片段，直接按顺序添加引用即可
        """
        result = []
        for citation in citations:
            # citation.sentence 就是对应的文本片段
            result.append(f"{citation.sentence} [来源: {citation.source_file}]")

        return '\n'.join(result)

    def _format_footnote_citations(self, answer: str, citations: List[Citation]) -> str:
        """脚注引用格式：答案末尾列出所有来源"""
        # 收集所有唯一的来源文件
        sources = list(set(cite.source_file for cite in citations))

        # 构建脚注
        footnotes = "\n\n参考来源：\n" + "\n".join(
            f"[{i+1}] {source}" for i, source in enumerate(sources)
        )

        return answer + footnotes



if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    print("=" * 70)
    print("引用追踪模块测试")
    print("=" * 70)

    # 创建测试数据
    test_documents = [
        {
            'document': 'TechCorp允许员工每周五携带宠物来办公室。',
            'metadata': {'file': 'pet_policy.md', 'category': 'policies'}
        },
        {
            'document': '宠物必须性格温顺且已接种疫苗。',
            'metadata': {'file': 'pet_policy.md', 'category': 'policies'}
        }
    ]

    manager = CitationManager()

    # 测试1：构建prompt
    print("\n测试1：构建引用prompt")
    print("-" * 70)
    prompt = manager.build_citation_prompt(
        "宠物政策有哪些要求？",
        test_documents
    )
    print(prompt[:200] + "...")

    # 测试2：解析JSON响应（新格式：segments）
    print("\n测试2：解析引用响应")
    print("-" * 70)
    test_response = '''
{
  "segments": [
    {"text": "TechCorp允许员工每周五携带宠物来办公室。", "source": "doc_1"},
    {"text": "宠物必须性格温顺且已接种疫苗。", "source": "doc_2"}
  ]
}
'''

    result = manager.parse_citation_response(test_response, test_documents)
    print(f"解析成功: {result['parse_success']}")
    print(f"答案: {result['answer']}")
    print(f"引用数量: {len(result['citations'])}")

    # 测试3：格式化引用
    print("\n测试3：格式化引用")
    print("-" * 70)
    formatted = manager.format_answer_with_citations(
        result['answer'],
        result['citations'],
        style="inline"
    )
    print(formatted)

    print("\n" + "=" * 70)
    print("测试完成")
