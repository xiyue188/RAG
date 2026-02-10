"""
LLM 调用模块
支持多个 LLM 提供商（OpenAI, Anthropic, 智谱AI, 通义千问）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    LLM_PROVIDER, SYSTEM_PROMPT, QUERY_TEMPLATE, NO_CONTEXT_TEMPLATE,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT, SIMILARITY_THRESHOLD,
    get_llm_config
)
from typing import Optional, List, Dict, Iterator
import time
from .logger import get_logger

# 初始化logger
logger = get_logger(__name__)


class LLMClient:
    """
    LLM 客户端
    封装多个 LLM 提供商的调用接口
    """

    def __init__(self, provider: Optional[str] = None):
        """
        初始化 LLM 客户端

        参数:
            provider: str - LLM 提供商（openai/anthropic/zhipu/qwen）
        """
        self.provider = provider or LLM_PROVIDER
        self.config = get_llm_config()

        # 验证配置和依赖
        self._validate_config()

        # 根据提供商初始化客户端
        self.client = None
        self._init_client()

    def _validate_config(self):
        """验证配置和依赖"""
        # 1. 验证 provider 是否支持
        supported_providers = ["openai", "anthropic", "zhipu", "qwen"]
        if self.provider not in supported_providers:
            raise ValueError(
                f"不支持的 LLM 提供商: {self.provider}\n"
                f"支持的提供商: {', '.join(supported_providers)}"
            )

        # 2. 验证 API Key 是否存在
        api_key = self.config.get("api_key")
        if not api_key or api_key == "your-api-key-here":
            raise ValueError(
                f"未配置 API Key\n"
                f"请在 .env 文件中设置 {self.provider.upper()}_API_KEY"
            )

        # 3. 验证依赖库是否已安装
        dependency_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "zhipu": "zhipuai",
            "qwen": "dashscope"
        }

        required_package = dependency_map[self.provider]
        try:
            __import__(required_package)
        except ImportError:
            raise ImportError(
                f"缺少依赖库: {required_package}\n"
                f"请运行: pip install {required_package}"
            )

    def _init_client(self):
        """初始化对应提供商的客户端"""
        logger.info(f"初始化 LLM 客户端: {self.provider}")

        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.config["api_key"],
                base_url=self.config.get("api_base")
            )

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=self.config["api_key"]
            )

        elif self.provider == "zhipu":
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(
                api_key=self.config["api_key"]
            )

        elif self.provider == "qwen":
            import dashscope
            dashscope.api_key = self.config["api_key"]
            self.client = dashscope

        else:
            raise ValueError(f"不支持的 LLM 提供商: {self.provider}")

        logger.info("LLM 客户端初始化完成")

    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> str:
        """
        生成回复

        参数:
            prompt: str - 用户提示词
            system_prompt: str - 系统提示词（可选）
            temperature: float - 温度参数（可选）
            max_tokens: int - 最大 token 数（可选）

        返回:
            str - LLM 生成的回复
        """
        system_prompt = system_prompt or SYSTEM_PROMPT
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

        try:
            if self.provider == "openai":
                return self._generate_openai(prompt, system_prompt, temperature, max_tokens)

            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, temperature, max_tokens)

            elif self.provider == "zhipu":
                return self._generate_zhipu(prompt, system_prompt, temperature, max_tokens)

            elif self.provider == "qwen":
                return self._generate_qwen(prompt, system_prompt, temperature, max_tokens)

            else:
                raise ValueError(f"不支持的提供商: {self.provider}")

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}", exc_info=True)
            raise  # 重新抛出异常，避免静默失败

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Iterator[str]:
        """
        流式生成回复（实时输出）

        参数:
            prompt: str - 用户提示词
            system_prompt: str - 系统提示词（可选）
            temperature: float - 温度参数（可选）
            max_tokens: int - 最大 token 数（可选）

        返回:
            Iterator[str] - 生成的文本流
        """
        system_prompt = system_prompt or SYSTEM_PROMPT
        temperature = temperature if temperature is not None else LLM_TEMPERATURE
        max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

        try:
            if self.provider == "openai":
                yield from self._stream_openai(prompt, system_prompt, temperature, max_tokens)

            elif self.provider == "anthropic":
                yield from self._stream_anthropic(prompt, system_prompt, temperature, max_tokens)

            elif self.provider == "zhipu":
                yield from self._stream_zhipu(prompt, system_prompt, temperature, max_tokens)

            elif self.provider == "qwen":
                yield from self._stream_qwen(prompt, system_prompt, temperature, max_tokens)

            else:
                raise ValueError(f"不支持的提供商: {self.provider}")

        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}", exc_info=True)
            raise

    def _generate_openai(self, prompt, system_prompt, temperature, max_tokens):
        """OpenAI API 调用"""
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_TIMEOUT
        )
        return response.choices[0].message.content

    def _stream_openai(self, prompt, system_prompt, temperature, max_tokens):
        """OpenAI API 流式调用"""
        stream = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_TIMEOUT,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _generate_anthropic(self, prompt, system_prompt, temperature, max_tokens):
        """Anthropic Claude API 调用"""
        response = self.client.messages.create(
            model=self.config["model"],
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_TIMEOUT
        )
        return response.content[0].text

    def _stream_anthropic(self, prompt, system_prompt, temperature, max_tokens):
        """Anthropic Claude API 流式调用"""
        with self.client.messages.stream(
            model=self.config["model"],
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=LLM_TIMEOUT
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _generate_zhipu(self, prompt, system_prompt, temperature, max_tokens):
        """智谱AI API 调用"""
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    def _stream_zhipu(self, prompt, system_prompt, temperature, max_tokens):
        """智谱AI API 流式调用"""
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _generate_qwen(self, prompt, system_prompt, temperature, max_tokens):
        """通义千问 API 调用"""
        from dashscope import Generation

        response = Generation.call(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message'
        )

        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"API 调用失败: {response.message}")

    def _stream_qwen(self, prompt, system_prompt, temperature, max_tokens):
        """通义千问 API 流式调用"""
        from dashscope import Generation

        responses = Generation.call(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            result_format='message',
            stream=True
        )

        for response in responses:
            if response.status_code == 200:
                # 获取增量内容
                if response.output.choices[0].message.content:
                    yield response.output.choices[0].message.content
            else:
                raise Exception(f"API 调用失败: {response.message}")

    def answer_with_context(self, question: str, context: str,
                            template: Optional[str] = None,
                            conversation_context: Optional[str] = None) -> str:
        """
        基于上下文回答问题（RAG 核心方法）

        参数:
            question: str - 用户问题
            context: str - 检索到的上下文
            template: str - 提示词模板（可选）
            conversation_context: str - 对话历史（可选）

        返回:
            str - LLM 生成的答案
        """
        template = template or QUERY_TEMPLATE

        # 如果有对话历史，在prompt中包含
        if conversation_context:
            # 在context前添加对话历史
            full_prompt = f"""{conversation_context}

当前查询的文档内容：
{context}

用户当前问题：{question}

回答："""
        else:
            # 构建完整提示词
            full_prompt = template.format(context=context, question=question)

        # 调用 LLM
        answer = self.generate(full_prompt)

        return answer

    def answer_with_citations(
        self,
        question: str,
        documents: List[Dict],
        conversation_context: Optional[str] = None,
        format_style: str = "inline",
        mode: str = "inline"
    ) -> Dict:
        """
        带引用的答案生成（Phase 2：精细化溯源）

        参数:
            question: str - 用户问题
            documents: List[Dict] - 检索到的文档列表
            conversation_context: str - 对话历史（可选）
            format_style: str - 引用格式（inline/footnote）
            mode: str - 引用模式（inline内联标记/json）

        返回:
            Dict - 包含answer、citations、formatted_answer等
        """
        from .citation import CitationManager

        citation_mgr = CitationManager()

        # 构建引用prompt
        prompt = citation_mgr.build_citation_prompt(
            question,
            documents,
            conversation_context,
            mode=mode
        )

        # 调用LLM
        logger.info(f"生成带引用的答案（模式：{mode}）...")
        response = self.generate(prompt)

        # 解析引用
        result = citation_mgr.parse_citation_response(response, documents, mode=mode)

        # 格式化引用（内联模式已经在parse时格式化）
        if not result.get('formatted_answer'):
            if result['parse_success'] and result['citations']:
                formatted = citation_mgr.format_answer_with_citations(
                    result['answer'],
                    result['citations'],
                    style=format_style
                )
                result['formatted_answer'] = formatted
            else:
                result['formatted_answer'] = result['answer']

        logger.info(f"引用生成完成：{len(result.get('citations', []))} 条引用")

        return result

    def answer_with_citations_stream(
        self,
        question: str,
        documents: List[Dict],
        conversation_context: Optional[str] = None
    ):
        """
        带引用的答案生成（流式版本，使用内联标记模式）

        参数:
            question: str - 用户问题
            documents: List[Dict] - 检索到的文档列表
            conversation_context: str - 对话历史（可选）

        返回:
            生成器 - 先yield元信息，然后流式yield文本（含内联引用标记）
        """
        from .citation import CitationManager

        citation_mgr = CitationManager()

        # 构建内联标记模式的prompt
        prompt = citation_mgr.build_citation_prompt(
            question,
            documents,
            conversation_context,
            mode="inline"
        )

        # 先yield元信息
        yield {
            'mode': 'with_citations',
            'citation_format': 'inline',
            'documents_count': len(documents)
        }

        # 流式调用LLM并实时输出
        logger.info("流式生成带引用的答案...")

        # 收集完整响应用于后处理
        full_response = ""
        for token in self.stream_generate(prompt):
            full_response += token
            yield token

        # 流式结束后，解析并yield引用信息
        result = citation_mgr.parse_inline_citations(full_response, documents)
        yield {
            'type': 'citation_meta',
            'citations': result.get('citations', []),
            'cited_count': result.get('cited_count', 0),
            'parse_success': result.get('parse_success', False)
        }

    def answer_with_context_stream(self, question: str, context: str,
                                   template: Optional[str] = None,
                                   conversation_context: Optional[str] = None) -> Iterator[str]:
        """
        基于上下文回答问题（RAG 核心方法 - 流式版本）

        参数:
            question: str - 用户问题
            context: str - 检索到的上下文
            template: str - 提示词模板（可选）
            conversation_context: str - 对话历史（可选）

        返回:
            Iterator[str] - 生成的答案文本流
        """
        template = template or QUERY_TEMPLATE

        # 如果有对话历史，在prompt中包含
        if conversation_context:
            full_prompt = f"""{conversation_context}

当前查询的文档内容：
{context}

用户当前问题：{question}

回答："""
        else:
            # 构建完整提示词
            full_prompt = template.format(context=context, question=question)

        # 流式调用 LLM
        yield from self.stream_generate(full_prompt)

    def answer_without_context(self, question: str,
                               template: Optional[str] = None) -> str:
        """
        无上下文回答问题（使用 LLM 通用知识）

        参数:
            question: str - 用户问题
            template: str - 提示词模板（可选）

        返回:
            str - LLM 生成的答案
        """
        template = template or NO_CONTEXT_TEMPLATE

        # 构建提示词
        prompt = template.format(question=question)

        # 调用 LLM
        answer = self.generate(prompt)

        return answer

    def answer_without_context_stream(self, question: str,
                                      template: Optional[str] = None) -> Iterator[str]:
        """
        无上下文回答问题（使用 LLM 通用知识 - 流式版本）

        参数:
            question: str - 用户问题
            template: str - 提示词模板（可选）

        返回:
            Iterator[str] - 生成的答案文本流
        """
        template = template or NO_CONTEXT_TEMPLATE
        prompt = template.format(question=question)
        yield from self.stream_generate(prompt)

    def answer_smart(self, question: str, retrieval_results: List[Dict],
                    threshold: Optional[float] = None,
                    conversation_context: Optional[str] = None) -> Dict:
        """
        智能回答（基于检索置信度自动分流）

        参数:
            question: str - 用户问题
            retrieval_results: List[Dict] - 检索结果列表
            threshold: float - 相似度阈值（可选，默认使用配置）
            conversation_context: str - 对话上下文（可选）

        返回:
            Dict - 包含答案和元信息:
                - answer: str - 回答内容
                - mode: str - 回答模式 ('with_context' / 'without_context')
                - max_similarity: float - 最高相似度
                - relevant_docs_count: int - 相关文档数量
        """
        threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

        # 检查是否有检索结果
        if not retrieval_results:
            # 没有任何结果，使用通用知识
            answer = self.answer_without_context(question)
            return {
                'answer': answer,
                'mode': 'without_context',
                'max_similarity': None,
                'relevant_docs_count': 0,
                'reason': '知识库中未找到相关文档'
            }

        # 获取最相关文档的相似度（距离越小越相似）
        max_similarity = retrieval_results[0].get('distance', float('inf'))

        # 根据阈值判断
        if max_similarity < threshold:
            # 有可靠文档，使用文档回答
            # 组合上下文
            context_parts = []
            relevant_count = 0
            for i, result in enumerate(retrieval_results, 1):
                # 只使用相似度高于阈值的文档
                if result.get('distance', float('inf')) < threshold:
                    meta = result['metadata']
                    doc = result['document']
                    context_parts.append(
                        f"[文档 {i}] 来源: {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}\n"
                        f"内容: {doc}\n"
                    )
                    relevant_count += 1

            context = "\n".join(context_parts)
            answer = self.answer_with_context(question, context, conversation_context=conversation_context)

            return {
                'answer': answer,
                'mode': 'with_context',
                'max_similarity': max_similarity,
                'relevant_docs_count': relevant_count,
                'reason': f'找到 {relevant_count} 个相关文档（相似度 < {threshold}）'
            }
        else:
            # 没有可靠文档，使用通用知识
            answer = self.answer_without_context(question)
            return {
                'answer': answer,
                'mode': 'without_context',
                'max_similarity': max_similarity,
                'relevant_docs_count': 0,
                'reason': f'文档相关度不足（{max_similarity:.3f} >= {threshold}）'
            }

    def answer_smart_stream(self, question: str, retrieval_results: List[Dict],
                           threshold: Optional[float] = None,
                           conversation_context: Optional[str] = None):
        """
        智能回答（基于检索置信度自动分流 - 流式版本）

        参数:
            question: str - 用户问题
            retrieval_results: List[Dict] - 检索结果列表
            threshold: float - 相似度阈值（可选，默认使用配置）
            conversation_context: str - 对话上下文（可选）

        返回:
            生成器 - 首先 yield 元信息字典，然后 yield 答案文本流
        """
        threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD

        # 检查是否有检索结果
        if not retrieval_results:
            # 先返回元信息
            yield {
                'mode': 'without_context',
                'max_similarity': None,
                'relevant_docs_count': 0,
                'reason': '知识库中未找到相关文档'
            }
            # 然后流式返回答案
            yield from self.answer_without_context_stream(question)
            return

        # 获取最相关文档的相似度
        max_similarity = retrieval_results[0].get('distance', float('inf'))

        # 根据阈值判断
        if max_similarity < threshold:
            # 有可靠文档，使用文档回答
            context_parts = []
            relevant_count = 0
            for i, result in enumerate(retrieval_results, 1):
                if result.get('distance', float('inf')) < threshold:
                    meta = result['metadata']
                    doc = result['document']
                    context_parts.append(
                        f"[文档 {i}] 来源: {meta.get('category', 'unknown')}/{meta.get('file', 'unknown')}\n"
                        f"内容: {doc}\n"
                    )
                    relevant_count += 1

            context = "\n".join(context_parts)

            # 先返回元信息
            yield {
                'mode': 'with_context',
                'max_similarity': max_similarity,
                'relevant_docs_count': relevant_count,
                'reason': f'找到 {relevant_count} 个相关文档（相似度 < {threshold}）'
            }
            # 然后流式返回答案
            yield from self.answer_with_context_stream(question, context, conversation_context=conversation_context)
        else:
            # 没有可靠文档，使用通用知识
            yield {
                'mode': 'without_context',
                'max_similarity': max_similarity,
                'relevant_docs_count': 0,
                'reason': f'文档相关度不足（{max_similarity:.3f} >= {threshold}）'
            }
            yield from self.answer_without_context_stream(question)




if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("LLM 模块测试")
    print("=" * 70)

    # 检查环境变量
    import os
    from dotenv import load_dotenv

    load_dotenv()  # 加载 .env 文件

    print(f"\n当前 LLM 提供商: {LLM_PROVIDER}")

    # 初始化客户端
    try:
        llm = LLMClient()
        print(f"{llm}\n")

        # 测试基础生成
        print("测试1: 基础生成")
        print("-" * 70)

        prompt = "用一句话解释什么是 RAG 系统。"
        print(f"提示词: {prompt}\n")

        answer = llm.generate(prompt)
        print(f"回答: {answer}\n")

        # 测试 RAG 场景
        print("\n测试2: RAG 场景（带上下文）")
        print("-" * 70)

        print("注意：以下使用示例数据进行测试\n")

        context = """
        [文档 1] 来源: test/sample_policy.md
        内容: 这是一个示例政策文档。用于测试系统功能。
        包含规定和制度相关内容，实际使用时请替换为真实文档。
        """

        question = "文档中包含什么内容？"

        print(f"问题: {question}\n")
        print(f"上下文:\n{context}\n")

        answer = llm.answer_with_context(question, context)
        print(f"回答: {answer}\n")

        # 测试流式生成（可选）
        if LLM_PROVIDER == "openai":
            print("\n测试3: 流式生成")
            print("-" * 70)

            prompt = "用三句话介绍向量数据库。"
            print(f"提示词: {prompt}\n")
            print("回答: ", end="", flush=True)

            for token in llm.stream_generate(prompt):
                print(token, end="", flush=True)
                time.sleep(0.05)  # 模拟打字效果

            print("\n")

        print("\n✓ 测试完成")

    except ValueError as e:
        print(f"\n✗ 初始化失败: {e}")
        print("\n请检查：")
        print("  1. 是否设置了正确的 API Key（在 .env 文件中）")
        print("  2. LLM_PROVIDER 是否正确（openai/anthropic/zhipu/qwen）")
        print("  3. 是否安装了对应的依赖包")
