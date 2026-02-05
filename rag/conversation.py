"""
对话管理模块
负责会话状态、历史记录、上下文维护和指代消解

核心功能：
1. ConversationManager - 管理对话历史，提供上下文
2. ReferenceResolver - 处理代词和指代消解
3. ConversationTurn - 单轮对话数据结构
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ConversationTurn:
    """
    单轮对话数据结构

    属性:
        role: str - 角色 ('user' | 'assistant')
        content: str - 消息内容
        metadata: Dict - 元数据（如来源文档、分数等）
        timestamp: datetime - 时间戳
    """

    def __init__(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        if role not in ('user', 'assistant'):
            raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")

        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            'role': self.role,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """从字典反序列化"""
        turn = cls(
            role=data['role'],
            content=data['content'],
            metadata=data.get('metadata', {})
        )
        if 'timestamp' in data:
            try:
                turn.timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                pass
        return turn


class ConversationManager:
    """
    会话管理器

    功能：
    - 维护对话历史
    - 提供上下文字符串（用于LLM）
    - 提取关键实体（用于指代消解）
    - 自动清理老旧对话
    - 支持序列化/反序列化
    """

    def __init__(self, max_turns: int = 20):
        """
        初始化会话管理器

        参数:
            max_turns: 最大保留轮次（自动清理老旧对话）
        """
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")

        self.history: List[ConversationTurn] = []
        self.max_turns = max_turns
        self.metadata: Dict[str, Any] = {}

        logger.info(f"ConversationManager initialized with max_turns={max_turns}")

    def add_user_message(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """添加用户消息到历史"""
        if not text or not text.strip():
            logger.warning("Attempted to add empty user message")
            return

        turn = ConversationTurn('user', text.strip(), metadata)
        self.history.append(turn)
        self._trim_history()

        logger.debug(f"Added user message: {text[:50]}...")

    def add_assistant_message(
        self,
        text: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """添加助手回复到历史"""
        if not text or not text.strip():
            logger.warning("Attempted to add empty assistant message")
            return

        metadata = {'sources': sources or []}
        turn = ConversationTurn('assistant', text.strip(), metadata)
        self.history.append(turn)
        self._trim_history()

        logger.debug(f"Added assistant message: {text[:50]}...")

    def get_recent_turns(self, n: int = 4) -> List[ConversationTurn]:
        """获取最近n轮对话"""
        if n < 1:
            return []
        return self.history[-n:]

    def get_context_for_llm(
        self,
        max_turns: int = 4,
        max_chars: int = 1000
    ) -> str:
        """
        为LLM构建上下文字符串

        格式：
        之前的对话:
        用户: 什么是宠物政策？
        助手: TechCorp允许...
        """
        recent = self.get_recent_turns(max_turns)
        if not recent:
            return ""

        context_lines = ["之前的对话:"]
        for turn in recent:
            prefix = "用户" if turn.role == 'user' else "助手"
            content = turn.content
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            context_lines.append(f"{prefix}: {content}")

        return "\n".join(context_lines)

    def get_last_user_message(self) -> Optional[str]:
        """获取最后一条用户消息"""
        for turn in reversed(self.history):
            if turn.role == 'user':
                return turn.content
        return None

    def extract_entities(self) -> List[str]:
        """
        从历史对话中提取关键实体
        用于指代消解
        """
        entities = []

        for turn in reversed(self.history):
            if turn.role == 'user':
                words = turn.content.split()
                for word in words:
                    if len(word) > 2 and word.isalnum():
                        if word not in entities:
                            entities.append(word)

        return entities[:10]

    def _trim_history(self) -> None:
        """自动清理老旧对话"""
        if len(self.history) > self.max_turns:
            removed = len(self.history) - self.max_turns
            self.history = self.history[-self.max_turns:]
            logger.info(f"Trimmed {removed} old conversation turns")

    def clear(self) -> None:
        """清空历史和元数据"""
        old_count = len(self.history)
        self.history = []
        self.metadata = {}
        logger.info(f"Cleared conversation history ({old_count} turns removed)")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（用于保存）"""
        return {
            'history': [turn.to_dict() for turn in self.history],
            'metadata': self.metadata,
            'max_turns': self.max_turns
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载（用于恢复）"""
        self.max_turns = data.get('max_turns', 20)
        self.metadata = data.get('metadata', {})

        self.history = []
        for turn_data in data.get('history', []):
            try:
                turn = ConversationTurn.from_dict(turn_data)
                self.history.append(turn)
            except (KeyError, ValueError) as e:
                logger.error(f"Failed to load conversation turn: {e}")

        logger.info(f"Loaded conversation with {len(self.history)} turns")

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationManager':
        """从JSON字符串加载"""
        data = json.loads(json_str)
        manager = cls()
        manager.from_dict(data)
        return manager

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"ConversationManager(turns={len(self.history)}, max={self.max_turns})"


class ReferenceResolver:
    """
    指代消解器

    功能：
    - 识别代词（它、这个、那个等）
    - 从对话历史中提取实体
    - 替换代词为具体实体
    """

    PRONOUNS = {
        '它': 1, '这个': 1, '那个': 1, '此': 1,
        '他': 1, '她': 1, '其': 1,
        '它们': 2, '这些': 2, '那些': 2,
        '他们': 2, '她们': 2,
    }

    def __init__(self, conversation: ConversationManager):
        self.conversation = conversation
        logger.debug("ReferenceResolver initialized")

    def has_pronoun(self, query: str) -> bool:
        """检查查询中是否包含代词"""
        return any(pronoun in query for pronoun in self.PRONOUNS)

    def resolve(self, query: str) -> str:
        """
        消解查询中的代词

        示例:
            输入: "它还有什么要求？"
            历史: ["什么是宠物政策？"]
            输出: "宠物政策还有什么要求？"
        """
        if not self.has_pronoun(query):
            return query

        entities = self.conversation.extract_entities()
        if not entities:
            logger.debug("No entities found for reference resolution")
            return query

        resolved_query = query
        replaced = []

        for pronoun in self.PRONOUNS:
            if pronoun in resolved_query and entities:
                entity = entities[0]
                resolved_query = resolved_query.replace(pronoun, entity, 1)
                replaced.append((pronoun, entity))

        if replaced:
            logger.info(f"Reference resolution: {replaced}")

        return resolved_query
