import { ref, computed, type Ref } from 'vue';
import { apiService } from '@/services/api';
import type { Message, LogEntry, SSEEvent, CitationDetail } from '@/types';
import { useSSEStream } from './useSSEStream';

/**
 * 聊天功能 Composable
 * @param logs - 日志列表
 * @param onCitationsReceived - 接收到引用时的回调（用于高亮chunks）
 */
export function useChat(logs: Ref<LogEntry[]>, onCitationsReceived?: (chunkIds: string[]) => void) {
  const messages: Ref<Message[]> = ref([]);
  const isThinking = ref(false);
  const isRagEnabled = ref(true);
  const sessionId = ref(`session-${Date.now()}`);
  const currentSimilarity = ref(0);
  const lastPrompt = ref<string | null>(null);
  const highlightedChunkId = ref<string | null>(null);

  const { eventToLog, updateStatusFromEvent } = useSSEStream();

  // 当前 AI 消息的累积内容
  let currentAiMessage = '';
  let currentMessageId = '';

  /**
   * 发送消息
   */
  const sendMessage = async (text: string) => {
    if (!text.trim() || isThinking.value) return;

    // 添加用户消息
    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now()
    };
    messages.value.push(userMessage);

    // 设置思考状态
    isThinking.value = true;
    currentAiMessage = '';
    currentMessageId = `msg-${Date.now()}-ai`;
    currentSimilarity.value = 0;  // 每次新消息重置相似度

    // 添加空 AI 消息占位
    messages.value.push({
      id: currentMessageId,
      role: 'ai',
      content: '',
      timestamp: Date.now(),
      isThinking: true,
      ragEnabled: isRagEnabled.value  // 快照发送时的 RAG 状态
    });

    try {
      // 调用 API
      await apiService.sendChatStream(
        text,
        sessionId.value,
        isRagEnabled.value,
        (event: SSEEvent) => {
          handleChatEvent(event);
        }
      );
    } catch (error) {
      console.error('Chat error:', error);
      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'error',
        stage: 'SYS',
        message: `Chat error: ${error}`
      });
    } finally {
      isThinking.value = false;
      // 移除 thinking 标记
      const aiMsg = messages.value.find(m => m.id === currentMessageId);
      if (aiMsg) {
        aiMsg.isThinking = false;
      }
    }
  };

  /**
   * 处理聊天 SSE 事件
   */
  const handleChatEvent = (event: SSEEvent) => {
    // 调试日志
    console.log('[SSE Event]', event.type, event.data);

    // 添加日志
    logs.value.push(eventToLog(event));

    // 更新状态
    updateStatusFromEvent(event.type);

    // 特殊处理
    switch (event.type) {
      case 'answer_chunk':
      case 'generation_chunk':
        // 累积 AI 回复内容（后端使用 answer_chunk）
        const chunk = event.data.chunk || event.data.content || '';
        currentAiMessage += chunk;
        const aiMsg = messages.value.find(m => m.id === currentMessageId);
        if (aiMsg) {
          aiMsg.content = currentAiMessage;
        }
        break;

      case 'retrieval_results':
        // 从检索结果的 top-1 中提取相似度
        if (event.data.results && event.data.results.length > 0) {
          const topResult = event.data.results[0];

          // 🎯 最佳实践：轻度非线性映射，突出高质量匹配
          // 后端已使用绝对距离转换，分数更合理，前端只需轻微调整
          let rawSimilarity = 0;

          // 优先使用 score（hybrid_score 或 rerank_score）
          if (topResult.score !== undefined && topResult.score > 0) {
            rawSimilarity = topResult.score;
          } else if (topResult.distance !== undefined) {
            // 回退到 distance 计算（余弦距离 0-2 → 相似度 0-1）
            rawSimilarity = Math.max(0, 1 - topResult.distance / 2);
          }

          // 轻度非线性映射：强调高质量匹配，压缩低质量匹配
          let displaySimilarity: number;
          if (rawSimilarity >= 0.7) {
            // 高质量匹配（70%+）：保持或略微提升
            displaySimilarity = Math.min(1.0, rawSimilarity * 1.1);
          } else if (rawSimilarity >= 0.4) {
            // 中等匹配（40-70%）：保持原值
            displaySimilarity = rawSimilarity;
          } else {
            // 低质量匹配（<40%）：压缩到更低
            displaySimilarity = rawSimilarity * 0.6;  // <40% → <24%
          }

          currentSimilarity.value = Math.max(0, Math.min(1, displaySimilarity));
          console.log('[相似度更新]', 'raw:', rawSimilarity.toFixed(3), '→ display:', currentSimilarity.value.toFixed(3));
        }
        break;

      case 'retrieval_done':
        // 更新相似度分数（从检索结果中提取距离）
        if (event.data.distances && event.data.distances.length > 0) {
          const maxDist = Math.min(...event.data.distances);
          currentSimilarity.value = Math.max(0, 1 - maxDist);
        }
        // 也可能在 data 中直接有分数
        if (event.data.similarity !== undefined) {
          currentSimilarity.value = event.data.similarity;
        }
        break;

      case 'citations':
      case 'citation_generated':
        // 添加引用信息
        const msgWithCitation = messages.value.find(m => m.id === currentMessageId);
        if (msgWithCitation && event.data.citations) {
          const citationsArray: any[] = Array.isArray(event.data.citations)
            ? event.data.citations
            : [event.data.citations];

          // 按出现顺序存储 chunk_id（不去重，与文本中标记一一对应）
          msgWithCitation.citations = citationsArray.map((c: any) => c.chunk_id || c.doc_id || c);

          // 同时保存完整引用信息（content、file 直接来自后端，无需 chunkMap 路径匹配）
          msgWithCitation.citationDetails = citationsArray.map((c: any): CitationDetail => ({
            chunkId: c.chunk_id || c.doc_id || '',
            file: c.file || 'unknown',
            category: c.category || '',
            content: c.content || ''
          }));

          // 触发 chunk 高亮（去重，只高亮唯一的 chunk）
          const uniqueIds = [...new Set(msgWithCitation.citations.filter(Boolean))];
          if (onCitationsReceived && uniqueIds.length > 0) {
            onCitationsReceived(uniqueIds);
          }
        }
        break;

      case 'done':
      case 'answer_complete':
        // 保存完整 prompt
        if (event.data.full_prompt) {
          lastPrompt.value = event.data.full_prompt;
        }
        break;

      case 'error':
        // 处理错误事件
        console.error('[聊天错误]', event.data);
        const errorMsg = messages.value.find(m => m.id === currentMessageId);
        if (errorMsg) {
          errorMsg.content = `❌ 错误: ${event.data.message || event.data.error || '未知错误'}`;
          errorMsg.isThinking = false;
        }
        break;
    }
  };

  /**
   * 切换 RAG 模式
   */
  const toggleRag = () => {
    isRagEnabled.value = !isRagEnabled.value;
  };

  /**
   * 处理引用悬停
   */
  const handleCitationHover = (chunkId: string | null) => {
    highlightedChunkId.value = chunkId;
  };

  return {
    messages,
    isThinking,
    isRagEnabled,
    currentSimilarity,
    lastPrompt,
    highlightedChunkId,
    sendMessage,
    toggleRag,
    handleCitationHover
  };
}
