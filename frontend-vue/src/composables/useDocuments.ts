import { ref, onMounted, type Ref } from 'vue';
import { apiService } from '@/services/api';
import type { KnowledgeFile, LogEntry, BackendDocumentListResponse, SSEEvent } from '@/types';
import { useSSEStream } from './useSSEStream';

/**
 * 文档管理 Composable
 */
export function useDocuments(logs: Ref<LogEntry[]>) {
  const files: Ref<KnowledgeFile[]> = ref([]);
  const isUploading = ref(false);
  const uploadProgress = ref(0);
  // 🎯 当前高亮的chunk IDs（用于问答时显示引用）
  const highlightedChunkIds: Ref<Set<string>> = ref(new Set());

  const { eventToLog, updateStatusFromEvent } = useSSEStream();

  /**
   * 加载文档列表（包含切片详情）
   */
  const loadDocuments = async () => {
    try {
      // 调用带 include_chunks=true 的 API
      const response: BackendDocumentListResponse = await apiService.getDocuments(true);

      // 转换为前端格式
      files.value = response.documents.map((doc, index) => {
        // 防御性编程：确保所有字段都存在
        const fileName = doc.file || 'unknown';
        const chunks = Array.isArray(doc.chunks) ? doc.chunks : [];

        return {
          id: `doc-${index}`, // 后端没有 id，生成一个
          name: fileName,
          type: inferFileType(fileName),
          size: 'N/A', // 后端没有返回大小
          isEnabled: true, // 默认启用
          // 映射后端返回的 chunks，添加错误处理
          chunks: chunks.map((chunk, idx) => ({
            id: chunk?.id || `chunk-${index}-${idx}`,
            content: chunk?.content || '',
            sourceId: fileName
          })),
          description: `Category: ${doc.category || 'unknown'}`,
          uploadTime: new Date().toISOString() // 后端没有返回时间
        };
      });

      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'success',
        stage: 'SYS',
        message: `Loaded ${response.total} documents with ${files.value.reduce((sum, f) => sum + f.chunks.length, 0)} chunks`
      });
    } catch (error) {
      console.error('Failed to load documents:', error);
      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'error',
        stage: 'SYS',
        message: `Failed to load documents: ${error}`
      });
    }
  };

  /**
   * 切换文件启用状态
   */
  const toggleFile = (id: string) => {
    const file = files.value.find(f => f.id === id);
    if (file) {
      file.isEnabled = !file.isEnabled;
      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'info',
        stage: 'SYS',
        message: `File ${file.name} ${file.isEnabled ? 'enabled' : 'disabled'}`
      });
    }
  };

  /**
   * 上传文档
   */
  const uploadDocuments = async (selectedFiles: File[]) => {
    if (selectedFiles.length === 0) return;

    isUploading.value = true;
    uploadProgress.value = 0;

    try {
      await apiService.uploadDocumentsStream(selectedFiles, (event: SSEEvent) => {
        handleUploadEvent(event);
      });

      // 上传完成后重新加载文档列表
      await loadDocuments();
    } catch (error) {
      console.error('Upload error:', error);
      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'error',
        stage: 'SYS',
        message: `Upload error: ${error}`
      });
    } finally {
      isUploading.value = false;
      uploadProgress.value = 0;
    }
  };

  /**
   * 处理上传 SSE 事件
   */
  const handleUploadEvent = (event: SSEEvent) => {
    // 添加日志
    logs.value.push(eventToLog(event));

    // 更新状态
    updateStatusFromEvent(event.type);

    // 更新进度
    if (event.type === 'embedding_progress' && event.data.percentage) {
      uploadProgress.value = event.data.percentage;
    } else if (event.type === 'all_complete') {
      uploadProgress.value = 100;
    }
  };

  /**
   * 删除文档
   * @param fileName 文档文件名（如 "policies/pet_policy.md"）
   */
  const deleteDocument = async (fileName: string) => {
    try {
      // 后端 DELETE 端点使用文件名
      await apiService.deleteDocument(fileName);

      // 从前端列表中移除
      files.value = files.value.filter(f => f.name !== fileName);

      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'success',
        stage: 'SYS',
        message: `Document deleted: ${fileName}`
      });
    } catch (error) {
      console.error('Delete error:', error);
      logs.value.push({
        id: `log-${Date.now()}`,
        timestamp: Date.now(),
        type: 'error',
        stage: 'SYS',
        message: `Delete error: ${error}`
      });
    }
  };

  // 推断文件类型
  const inferFileType = (filename: string): 'pdf' | 'txt' | 'md' | 'docx' => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return 'pdf';
    if (ext === 'md') return 'md';
    if (ext === 'docx') return 'docx';
    return 'txt';
  };

  // 格式化文件大小
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  /**
   * 设置高亮的chunks（根据引用信息）
   * @param chunkIds 要高亮的chunk IDs
   */
  const setHighlightedChunks = (chunkIds: string[]) => {
    highlightedChunkIds.value = new Set(chunkIds);
    console.log('[Chunk高亮]', chunkIds);
  };

  /**
   * 清除高亮
   */
  const clearHighlights = () => {
    highlightedChunkIds.value = new Set();
  };

  // 组件挂载时加载文档
  onMounted(() => {
    loadDocuments();
  });

  return {
    files,
    isUploading,
    uploadProgress,
    highlightedChunkIds,
    toggleFile,
    uploadDocuments,
    deleteDocument,
    loadDocuments,
    setHighlightedChunks,
    clearHighlights
  };
}
