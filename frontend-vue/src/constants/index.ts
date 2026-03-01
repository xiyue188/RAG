import type { LogEntry, KnowledgeFile } from '@/types';

// 初始欢迎日志
export const INITIAL_LOGS_WELCOME: LogEntry[] = [
  {
    id: 'init-1',
    timestamp: Date.now(),
    type: 'info',
    stage: 'SYS',
    message: 'System Initialized. Kernel v4.2.0 active.'
  },
  {
    id: 'init-2',
    timestamp: Date.now() + 100,
    type: 'success',
    stage: 'SYS',
    message: 'Vector Database connected.'
  },
  {
    id: 'init-3',
    timestamp: Date.now() + 200,
    type: 'info',
    stage: 'SYS',
    message: 'FastAPI Backend ready at http://localhost:8000'
  }
];

// 示例场景数据（医疗指南示例）
const MEDICAL_GUIDE_TEXT = `
[SECTION 4.2] INTOXICATION BY LEGUMES (BEANS)
Symptoms: Nausea, vomiting, abdominal pain, diarrhea.
Treatment:
1. DO NOT induce vomiting if the patient is unconscious.
2. If conscious, induce vomiting immediately using Ipecac syrup or mechanical stimulation.
3. Administer activated charcoal (50g).
4. Isolate patient and monitor vitals every 15 minutes.
Note: Raw kidney beans contain Phytohaemagglutinin. Cooking at boiling point is required to destroy the toxin.
`;

// 初始示例文件（可选）
export const SCENARIO_FILES: KnowledgeFile[] = [
  {
    id: 'demo-1',
    name: 'Ship_Medical_Guide.pdf',
    type: 'pdf',
    size: '2.4 MB',
    isEnabled: false,
    description: 'Standard medical procedures for merchant vessels.',
    chunks: [
      { id: 'c-1-1', sourceId: 'demo-1', content: 'General hygiene: Wash hands frequently to prevent infection spread.' },
      { id: 'c-1-2', sourceId: 'demo-1', content: MEDICAL_GUIDE_TEXT.trim() }
    ]
  }
];

// API 基础地址
export const API_BASE_URL = '/api/v1';

// SSE 事件类型到日志阶段的映射
export const SSE_EVENT_TO_STAGE: Record<string, LogEntry['stage']> = {
  // 后端实际事件类型
  'connected': 'SYS',
  'retrieval_status': 'SEARCH',
  'retrieval_results': 'HIT',
  'answer_chunk': 'GEN',
  'citations': 'HIT',
  'done': 'SYS',
  'error': 'SYS',

  // 查询解析与增强
  'resolved': 'USER',
  'query_rewritten': 'USER',

  // 多查询扩展
  'multi_query_start': 'SEARCH',
  'multi_query_done': 'SEARCH',

  // 混合检索
  'hybrid_search_start': 'SEARCH',
  'bm25_indexing': 'SEARCH',
  'bm25_indexed': 'SEARCH',

  // 查询阶段
  'query_received': 'USER',
  'embedding_start': 'EMBED',
  'embedding_done': 'EMBED',
  'retrieval_start': 'SEARCH',
  'retrieval_done': 'SEARCH',
  'rerank_start': 'SEARCH',
  'rerank_done': 'HIT',
  'generation_start': 'GEN',
  'generation_chunk': 'GEN',
  'generation_done': 'GEN',
  'answer_complete': 'GEN',

  // 摄入阶段
  'upload_start': 'UPLOAD',
  'file_received': 'UPLOAD',
  'parsing_start': 'PARSE',
  'parsing_done': 'PARSE',
  'chunking_start': 'CHUNK',
  'chunking_done': 'CHUNK',
  'embedding_start_ingestion': 'EMBED',
  'embedding_progress': 'EMBED',
  'embedding_done_ingestion': 'EMBED',
  'storing_start': 'STORE',
  'storing_done': 'STORE',
  'indexing_done': 'STORE',
  'upload_complete': 'SYS',
  'all_complete': 'SYS'
};

// SSE 事件类型到日志类型的映射
export const SSE_EVENT_TO_TYPE: Record<string, LogEntry['type']> = {
  'error': 'error',
  'upload_complete': 'success',
  'all_complete': 'success',
  'answer_complete': 'success',
  'done': 'success',
  'bm25_indexed': 'success',
  'multi_query_done': 'info',
  'query_rewritten': 'info',
  'hybrid_search_start': 'info',
  'generation_chunk': 'process',
  'answer_chunk': 'process',
  'embedding_progress': 'process'
};
