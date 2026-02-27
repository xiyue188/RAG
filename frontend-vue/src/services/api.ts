import { API_BASE_URL } from '@/constants';
import type { BackendDocumentListResponse, SSEEvent } from '@/types';

// API 服务类
class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // ==================== 文档管理 API ====================

  /**
   * 获取所有文档列表
   * @param includeChunks 是否包含切片详情（默认 false）
   */
  async getDocuments(includeChunks: boolean = false): Promise<BackendDocumentListResponse> {
    const url = includeChunks
      ? `${this.baseUrl}/documents?include_chunks=true`
      : `${this.baseUrl}/documents`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch documents: ${response.statusText}`);
    }
    return response.json();
  }

  /**
   * 删除指定文档
   */
  async deleteDocument(docId: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/documents/${docId}`, {
      method: 'DELETE'
    });
    if (!response.ok) {
      throw new Error(`Failed to delete document: ${response.statusText}`);
    }
  }

  /**
   * 上传文档（SSE 流式版本）
   * @param files 文件列表
   * @param onEvent SSE 事件回调
   */
  async uploadDocumentsStream(
    files: File[],
    onEvent: (event: SSEEvent) => void
  ): Promise<void> {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${this.baseUrl}/documents/upload/stream`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    // 处理 SSE 流
    await this.handleSSEStream(response, onEvent);
  }

  // ==================== 聊天 API ====================

  /**
   * 发送聊天消息（SSE 流式版本）
   * @param question 用户问题
   * @param sessionId 会话 ID
   * @param useRetrieval 是否使用知识库检索（RAG开关）
   * @param onEvent SSE 事件回调
   */
  async sendChatStream(
    question: string,
    sessionId: string,
    useRetrieval: boolean,
    onEvent: (event: SSEEvent) => void
  ): Promise<void> {
    const requestBody = {
      question,
      session_id: sessionId,
      use_retrieval: useRetrieval,
      enable_citation: true  // 保持启用引用显示
    };

    const response = await fetch(`${this.baseUrl}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestBody)
    });

    if (!response.ok) {
      throw new Error(`Chat failed: ${response.statusText}`);
    }

    // 处理 SSE 流
    await this.handleSSEStream(response, onEvent);
  }

  // ==================== SSE 流处理 ====================

  /**
   * 处理 SSE 流式响应
   */
  private async handleSSEStream(
    response: Response,
    onEvent: (event: SSEEvent) => void
  ): Promise<void> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim() || !line.startsWith('data: ')) continue;

          try {
            const jsonStr = line.slice(6); // 移除 "data: " 前缀
            const event: SSEEvent = JSON.parse(jsonStr);
            onEvent(event);
          } catch (e) {
            console.error('Failed to parse SSE event:', line, e);
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// 导出单例
export const apiService = new ApiService();
