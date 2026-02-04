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
from typing import Optional, List, Dict
import time


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

        # 根据提供商初始化客户端
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化对应提供商的客户端"""
        print(f"初始化 LLM 客户端: {self.provider}")

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

        print(f"[OK] LLM 客户端初始化完成")

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
            return f"[LLM 调用失败: {e}]"

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

    def answer_with_context(self, question: str, context: str,
                            template: Optional[str] = None) -> str:
        """
        基于上下文回答问题（RAG 核心方法）

        参数:
            question: str - 用户问题
            context: str - 检索到的上下文
            template: str - 提示词模板（可选）

        返回:
            str - LLM 生成的答案
        """
        template = template or QUERY_TEMPLATE

        # 构建完整提示词
        prompt = template.format(context=context, question=question)

        # 调用 LLM
        answer = self.generate(prompt)

        return answer

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

    def answer_smart(self, question: str, retrieval_results: List[Dict],
                    threshold: Optional[float] = None) -> Dict:
        """
        智能回答（基于检索置信度自动分流）

        参数:
            question: str - 用户问题
            retrieval_results: List[Dict] - 检索结果列表
            threshold: float - 相似度阈值（可选，默认使用配置）

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
            answer = self.answer_with_context(question, context)

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
