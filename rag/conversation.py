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
        从用户的历史问题中提取主题实体（使用词性标注 + 复合名词识别）

        改进（Phase 1 优化 v3）：
        1. 使用 jieba 词性标注，只提取名词
        2. 合并连续的名词为复合名词（如"宠物"+"政策" → "宠物政策"）
        3. 过滤动词、形容词、疑问词等
        4. 优先保留复合名词
        5. 添加详细日志便于调试

        返回:
            List[str] - 提取的实体列表（最多5个）
        """
        try:
            import jieba.posseg as pseg
        except ImportError:
            logger.warning("jieba.posseg not available, falling back to simple extraction")
            return self._extract_entities_simple()

        entities = []

        # 只看最近的3轮用户问题（6条消息）
        user_turns = [turn for turn in reversed(self.history[-6:]) if turn.role == 'user']

        for turn in user_turns:
            content = turn.content

            # 使用词性标注分词
            words = list(pseg.cut(content))

            # 合并连续的词为复合名词（支持形容词+名词、动词+名词）
            i = 0
            while i < len(words):
                word, flag = words[i]

                # 检查是否可以作为复合名词的起始
                # 支持：形容词(a)、动词(v)、名词(n/nr/ns/nt/nz/vn/an)
                if flag in ['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'an', 'a', 'v']:
                    # 开始构建复合名词
                    compound = word
                    compound_flags = [flag]
                    j = i + 1

                    # 尝试合并后续的名词
                    while j < len(words):
                        next_word, next_flag = words[j]
                        # 后续词必须是名词
                        if next_flag in ['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'an']:
                            compound += next_word
                            compound_flags.append(next_flag)
                            j += 1
                        else:
                            break

                    # 验证复合名词的合法性
                    # 1. 如果只有一个词，必须是名词（过滤单独的形容词/动词）
                    # 2. 如果多个词，最后一个必须是名词
                    is_valid = False
                    if j == i + 1:  # 单个词
                        if flag in ['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'an']:
                            is_valid = True
                    else:  # 复合词
                        # 确保最后一个词是名词
                        if compound_flags[-1] in ['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'an']:
                            is_valid = True

                    # 长度要求：2-8字
                    if is_valid and 2 <= len(compound) <= 8:
                        # 过滤掉疑问词、代词、通用动词/形容词
                        stop_words = {
                            '什么', '哪些', '如何', '怎么', '怎样', '为什么', '为何',
                            '它', '他', '她', '这个', '那个', '这些', '那些',
                            '评价', '改进', '优点', '缺点', '问题', '办法', '方法',
                            '东西', '事情', '地方', '时候', '方面', '情况'
                        }

                        if compound not in stop_words and compound not in entities:
                            entities.append(compound)
                            if j > i + 1:
                                logger.debug(f"[实体提取] 复合名词: '{compound}' (合并了 {j-i} 个词, 词性: {'+'.join(compound_flags)})")
                            else:
                                logger.debug(f"[实体提取] 单一名词: '{compound}' (词性: {flag})")

                    # 跳过已经处理的词
                    i = j
                else:
                    i += 1

        # 返回最多5个实体（最近提到的）
        result = entities[:5]
        if result:
            logger.info(f"[实体提取] 从 {len(user_turns)} 轮对话中提取: {result}")
        else:
            logger.debug("[实体提取] 未提取到任何实体")

        return result

    def _extract_entities_simple(self) -> List[str]:
        """简单实体提取（不使用词性标注）- 回退方法"""
        entities = []

        for turn in reversed(self.history):
            if turn.role == 'user':
                content = turn.content

                # 移除疑问词和标点
                for word in ['什么是', '请问', '如何', '怎么', '怎样', '吗', '呢', '啊', '吧', '？', '?', '！', '!', '。', '，', ',']:
                    content = content.replace(word, ' ')

                content = content.strip()
                if not content:
                    continue

                import re
                words = re.split(r'\s+', content)

                for word in words:
                    word = word.strip()
                    if 2 <= len(word) <= 6 and word not in ['的', '了', '着', '过', '得', '地']:
                        if word not in entities:
                            entities.append(word)

        return entities[:5]

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

        改进（Phase 1 优化 v2）：
        - 使用词性标注优先选择名词实体
        - 只替换第一个代词（避免过度替换）
        - 优先选择复合名词和专有名词
        - 添加详细日志便于调试

        示例:
            输入: "它还有什么要求？"
            历史: ["什么是宠物政策？"]
            输出: "宠物政策还有什么要求？"
        """
        if not self.has_pronoun(query):
            logger.debug(f"[指代消解] 查询无代词，保持原样: '{query}'")
            return query

        entities = self.conversation.extract_entities()
        if not entities:
            logger.warning(f"[指代消解] 失败：无法从历史中提取实体。查询: '{query}'")
            return query

        # 选择最合适的实体（优先选择复合名词，即长度较长的）
        # 例如：['宠物政策', '政策'] → 选择 '宠物政策'
        best_entity = max(entities[:3], key=len)  # 从前3个中选最长的

        resolved_query = query
        replaced = []

        # 替换代词（只替换第一个，避免过度替换）
        for pronoun in self.PRONOUNS:
            if pronoun in resolved_query:
                resolved_query = resolved_query.replace(pronoun, best_entity, 1)
                replaced.append((pronoun, best_entity))
                logger.info(f"[指代消解] ✓ 成功: '{query}' -> '{resolved_query}' (替换: {pronoun} → {best_entity})")
                break  # 只替换第一个代词

        if not replaced:
            logger.debug(f"[指代消解] 查询包含代词但未能消解: '{query}'")

        return resolved_query
