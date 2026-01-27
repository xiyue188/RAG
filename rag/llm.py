"""
LLM 调用模块
支持多个 LLM 提供商（OpenAI, Anthropic, 智谱AI, 通义千问）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    LLM_PROVIDER, SYSTEM_PROMPT, QUERY_TEMPLATE,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
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

        print(f"✓ LLM 客户端初始化完成")

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

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None):
        """
        流式生成（可选功能）

        参数:
            prompt: str - 用户提示词
            system_prompt: str - 系统提示词

        生成器:
            每次返回一个 token
        """
        # 简化实现：只支持 OpenAI
        if self.provider != "openai":
            # 非流式返回
            yield self.generate(prompt, system_prompt)
            return

        system_prompt = system_prompt or SYSTEM_PROMPT

        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def __repr__(self):
        return f"LLMClient(provider={self.provider}, model={self.config.get('model', 'unknown')})"


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

        context = """
        [文档 1] 来源: policies/pet_policy.md
        内容: TechCorp 宠物政策：员工可以在每周五带宠物来办公室。
        宠物必须性格温顺且已接种疫苗。CEO 的金毛寻回犬是公司吉祥物。
        """

        question = "可以带狗来公司吗？"

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
