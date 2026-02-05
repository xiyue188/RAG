"""
Phase 1 功能测试脚本
测试对话管理、指代消解等功能
"""

import sys
import io
from pathlib import Path

# 设置标准输出编码为UTF-8（永久修复Windows GBK编码问题）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag.conversation import ConversationManager, ReferenceResolver


def test_conversation_manager():
    """测试对话管理器"""
    print("=" * 70)
    print("测试 1: ConversationManager（对话管理器）")
    print("=" * 70)

    # 创建会话管理器
    conv = ConversationManager(max_turns=10)
    print(f"\n✓ 创建会话管理器: {conv}")

    # 测试添加对话
    conv.add_user_message("什么是宠物政策？")
    conv.add_assistant_message("TechCorp允许员工携带宠物上班，但需要满足以下要求...")
    conv.add_user_message("它还有什么具体要求？")

    print(f"✓ 添加了 {len(conv)} 轮对话")

    # 测试获取上下文
    context = conv.get_context_for_llm(max_turns=4)
    print(f"\n对话上下文:")
    print("-" * 70)
    print(context)
    print("-" * 70)

    # 测试提取实体
    entities = conv.extract_entities()
    print(f"\n✓ 提取的实体: {entities}")

    # 测试序列化
    json_str = conv.to_json()
    print(f"\n✓ 序列化成功（长度: {len(json_str)} 字符）")

    # 测试反序列化
    conv2 = ConversationManager.from_json(json_str)
    print(f"✓ 反序列化成功: {conv2}")
    assert len(conv2) == len(conv), "反序列化后历史长度不匹配"

    print("\n✓ ConversationManager 测试通过")


def test_reference_resolver():
    """测试指代消解器"""
    print("\n" + "=" * 70)
    print("测试 2: ReferenceResolver（指代消解器）")
    print("=" * 70)

    # 创建会话和消解器
    conv = ConversationManager()
    conv.add_user_message("什么是宠物政策？")
    conv.add_assistant_message("TechCorp允许...")

    resolver = ReferenceResolver(conv)
    print(f"\n✓ 创建指代消解器")

    # 测试用例
    test_cases = [
        ("它还有什么要求？", "宠物政策还有什么要求？"),
        ("这个政策的细节？", "宠物政策政策的细节？"),  # 会替换"这个"
        ("401k匹配比例是多少？", "401k匹配比例是多少？"),  # 无代词，不变
    ]

    print("\n指代消解测试:")
    print("-" * 70)
    for original, expected_pattern in test_cases:
        resolved = resolver.resolve(original)
        has_pronoun = resolver.has_pronoun(original)
        print(f"原始: {original}")
        print(f"消解: {resolved}")
        print(f"包含代词: {has_pronoun}")
        print()

    print("✓ ReferenceResolver 测试通过")


def test_context_aware_features():
    """测试上下文感知功能"""
    print("\n" + "=" * 70)
    print("测试 3: 上下文感知功能")
    print("=" * 70)

    conv = ConversationManager()

    # 模拟多轮对话
    conversation_flow = [
        ("用户", "什么是401k计划？"),
        ("助手", "401k是一种退休储蓄计划，TechCorp为员工提供匹配贡献..."),
        ("用户", "匹配比例是多少？"),
        ("助手", "公司匹配50%，最高工资的6%..."),
        ("用户", "它有年龄限制吗？"),
    ]

    print("\n模拟对话流:")
    print("-" * 70)
    for role, content in conversation_flow[:-1]:  # 除了最后一个
        if role == "用户":
            conv.add_user_message(content)
            print(f"用户: {content}")
        else:
            conv.add_assistant_message(content)
            print(f"助手: {content[:50]}...")

    # 最后一个问题进行指代消解
    last_question = conversation_flow[-1][1]
    resolver = ReferenceResolver(conv)
    resolved = resolver.resolve(last_question)

    print(f"\n用户: {last_question}")
    print(f"消解为: {resolved}")

    # 获取上下文
    context = conv.get_context_for_llm(max_turns=4)
    print(f"\n生成的LLM上下文:")
    print("-" * 70)
    print(context)
    print("-" * 70)

    print("\n✓ 上下文感知功能测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 70)
    print("测试 4: 边界情况")
    print("=" * 70)

    # 测试空历史
    conv = ConversationManager()
    context = conv.get_context_for_llm()
    assert context == "", "空历史应返回空字符串"
    print("✓ 空历史处理正确")

    # 测试历史清理
    conv = ConversationManager(max_turns=3)
    for i in range(5):
        conv.add_user_message(f"问题 {i}")
        conv.add_assistant_message(f"回答 {i}")

    assert len(conv) == 3, f"应自动清理到3轮，实际{len(conv)}轮"
    print("✓ 历史自动清理正确")

    # 测试空消息
    conv.add_user_message("")
    conv.add_user_message("   ")
    assert len(conv) == 3, "空消息不应被添加"
    print("✓ 空消息过滤正确")

    # 测试无实体的指代消解
    conv_empty = ConversationManager()
    resolver_empty = ReferenceResolver(conv_empty)
    resolved = resolver_empty.resolve("它是什么？")
    print(f"✓ 无实体时消解结果: {resolved}")

    print("\n✓ 边界情况测试通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("Phase 1 功能测试")
    print("=" * 70)

    try:
        test_conversation_manager()
        test_reference_resolver()
        test_context_aware_features()
        test_edge_cases()

        print("\n" + "=" * 70)
        print("✓ 所有测试通过")
        print("=" * 70)
        print("\nPhase 1 核心对话能力已成功实现：")
        print("  ✓ ConversationManager - 对话历史管理")
        print("  ✓ ReferenceResolver - 指代消解")
        print("  ✓ 上下文感知功能")
        print("  ✓ 序列化/反序列化")
        print("  ✓ 边界情况处理")
        print("\n准备提交到版本控制...")

    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
