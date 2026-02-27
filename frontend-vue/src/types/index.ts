// 文档分块
export interface Chunk {
  id: string;
  content: string;
  sourceId: string;
  score?: number;
}

// 知识库文件
export interface KnowledgeFile {
  id: string;
  name: string;
  type: 'pdf' | 'txt' | 'md' | 'docx';
  size: string;
  isEnabled: boolean;
  chunks: Chunk[];
  description: string;
  uploadTime?: string;
}

// 聊天消息
export interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string;
  timestamp: number;
  citations?: string[];
  /** 按引用出现顺序的完整引用信息（来自后端 citations 事件） */
  citationDetails?: CitationDetail[];
  isThinking?: boolean;
  ragEnabled?: boolean;  // 发送时的 RAG 状态快照
}

/** 后端引用的完整信息（对应 _stream_with_citations 发送的每条 citation） */
export interface CitationDetail {
  chunkId: string;
  file: string;
  category: string;
  content: string;
}

// 日志条目
export interface LogEntry {
  id: string;
  timestamp: number;
  type: 'info' | 'success' | 'warning' | 'error' | 'process';
  stage: 'USER' | 'EMBED' | 'SEARCH' | 'HIT' | 'GEN' | 'SYS' | 'UPLOAD' | 'PARSE' | 'CHUNK' | 'STORE';
  message: string;
  details?: string;
}

// RAG 状态枚举
export enum RagStatus {
  IDLE = 'IDLE',
  EMBEDDING = 'EMBEDDING',
  SEARCHING = 'SEARCHING',
  GENERATING = 'GENERATING',
  UPLOADING = 'UPLOADING',
  COMPLETE = 'COMPLETE'
}

// SSE 事件类型
export type SSEEventType =
  // 查询阶段事件（17种）
  | 'query_received' | 'session_start' | 'conversation_context_loaded'
  | 'hybrid_check_start' | 'query_rewrite_start' | 'query_rewrite_done'
  | 'embedding_start' | 'embedding_done' | 'retrieval_start' | 'retrieval_done'
  | 'rerank_start' | 'rerank_done' | 'generation_start' | 'generation_chunk'
  | 'generation_done' | 'citation_generated' | 'answer_complete'
  // 摄入阶段事件（14种）
  | 'upload_start' | 'file_received' | 'parsing_start' | 'parsing_done'
  | 'chunking_start' | 'chunking_done' | 'embedding_start_ingestion'
  | 'embedding_progress' | 'embedding_done_ingestion' | 'storing_start'
  | 'storing_done' | 'indexing_done' | 'upload_complete' | 'all_complete'
  | 'error';

// SSE 事件数据结构
export interface SSEEvent {
  type: SSEEventType;
  data: any;
}

// 后端切片信息 (对应 backend/schemas ChunkInfo)
export interface BackendChunkInfo {
  id: string;
  content: string;
  index: number;
}

// 后端文档响应（单个文档，对应 backend/schemas DocumentInfo）
export interface BackendDocument {
  file: string;
  category: string;
  chunks?: BackendChunkInfo[];  // 可选的切片列表
}

// 后端文档列表响应
export interface BackendDocumentListResponse {
  documents: BackendDocument[];
  total: number;
}
