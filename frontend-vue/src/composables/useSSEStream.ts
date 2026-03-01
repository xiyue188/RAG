import { ref, type Ref } from 'vue';
import type { SSEEvent, LogEntry, RagStatus } from '@/types';
import { SSE_EVENT_TO_STAGE, SSE_EVENT_TO_TYPE } from '@/constants';

/**
 * SSE 流处理 Composable
 */
export function useSSEStream() {
  const isStreaming = ref(false);
  const currentStatus: Ref<RagStatus> = ref('IDLE' as RagStatus);

  /**
   * 将 SSE 事件转换为日志条目
   */
  const eventToLog = (event: SSEEvent): LogEntry => {
    const stage = SSE_EVENT_TO_STAGE[event.type] || 'SYS';
    const type = SSE_EVENT_TO_TYPE[event.type] || 'info';

    let message = '';
    let details = '';

    switch (event.type) {
      // 后端实际事件类型
      case 'connected':
        message = `Connected to session: ${event.data.session_id || ''}`;
        break;
      case 'retrieval_status':
        message = `Retrieval status: ${event.data.status || ''}`;
        break;
      case 'answer_chunk':
        message = event.data.content || '';
        break;
      case 'citations':
        message = `Citations added: ${event.data.citations?.length || 0} sources`;
        break;
      case 'done':
        message = 'Response completed';
        break;

      // 通用事件类型
      case 'query_received':
        message = `Query: ${event.data.question || ''}`;
        break;
      case 'embedding_start':
        message = 'Embedding query...';
        break;
      case 'embedding_done':
        message = `Embedding completed (${event.data.chunk_count || 0} chunks)`;
        break;
      case 'retrieval_start':
        message = `Searching knowledge base (top ${event.data.top_k || 5})...`;
        break;
      case 'retrieval_done':
        message = `Found ${event.data.num_documents || event.data.result_count || 0} candidates`;
        details = event.data.distances ? `Distances: ${JSON.stringify(event.data.distances)}` : '';
        break;
      case 'retrieval_results':
        const results = event.data.results || [];
        message = `Retrieved ${results.length} results (total: ${event.data.total || 0})`;
        if (results.length > 0) {
          const top = results[0];
          details = `Top match: ${top.file} (similarity: ${(1 - (top.distance || 0)).toFixed(2)}, score: ${(top.score || 0).toFixed(2)})`;
        }
        break;
      case 'rerank_start':
        message = 'Re-ranking results...';
        break;
      case 'rerank_done':
        message = `Re-rank completed (${event.data.reranked_count || 0} results)`;
        break;
      case 'generation_start':
        message = `Generating response (${event.data.num_docs || 0} docs)...`;
        break;
      case 'generation_chunk':
        message = event.data.chunk || '';
        break;
      case 'generation_done':
        message = 'Generation completed';
        break;
      case 'answer_complete':
        message = 'Answer delivered';
        details = `Session: ${event.data.session_id || 'N/A'}`;
        break;

      // 上传阶段事件
      case 'upload_start':
        message = `Starting upload: ${event.data.file_count || 0} file(s)`;
        break;
      case 'file_received':
        message = `File received: ${event.data.filename || ''}`;
        details = `Size: ${event.data.size || 0} bytes`;
        break;
      case 'parsing_start':
        message = `Parsing: ${event.data.filename || ''}`;
        break;
      case 'parsing_done':
        message = `Parsed ${event.data.chars || 0} characters`;
        break;
      case 'chunking_start':
        message = 'Chunking text...';
        break;
      case 'chunking_done':
        message = `Created ${event.data.chunk_count || 0} chunks`;
        break;
      case 'embedding_progress':
        message = `Embedding progress: ${event.data.current || 0}/${event.data.total || 0} (${event.data.percentage || 0}%)`;
        break;
      case 'storing_start':
        message = 'Storing vectors...';
        break;
      case 'storing_done':
        message = `Stored ${event.data.chunk_count || 0} vectors`;
        break;
      case 'indexing_done':
        message = 'Indexing completed';
        break;
      case 'upload_complete':
        message = `Upload complete: ${event.data.filename || ''}`;
        details = `Total chunks: ${event.data.chunk_count || 0}`;
        break;
      case 'all_complete':
        message = `All uploads complete (${event.data.total_files || 0} files)`;
        break;

      case 'error':
        message = `Error: ${event.data.error || 'Unknown error'}`;
        details = event.data.detail || '';
        break;

      default:
        message = `[${event.type}] ${JSON.stringify(event.data)}`;
    }

    return {
      id: `log-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      type,
      stage,
      message,
      details
    };
  };

  /**
   * 根据事件类型更新 RAG 状态
   */
  const updateStatusFromEvent = (eventType: string) => {
    if (eventType.includes('embedding')) {
      currentStatus.value = 'EMBEDDING';
    } else if (eventType.includes('retrieval') || eventType.includes('search')) {
      currentStatus.value = 'SEARCHING';
    } else if (eventType.includes('generation') || eventType.includes('generating')) {
      currentStatus.value = 'GENERATING';
    } else if (eventType.includes('upload')) {
      currentStatus.value = 'UPLOADING';
    } else if (eventType === 'answer_complete' || eventType === 'all_complete') {
      currentStatus.value = 'COMPLETE';
      setTimeout(() => {
        currentStatus.value = 'IDLE';
      }, 2000);
    }
  };

  return {
    isStreaming,
    currentStatus,
    eventToLog,
    updateStatusFromEvent
  };
}
