<template>
  <div class="flex h-screen w-screen overflow-hidden font-sans bg-deep-950 text-slate-200">
    <!-- Left: Library (25%) -->
    <div class="w-1/4 min-w-[320px]">
      <LibraryPanel
        :files="files"
        :highlighted-chunk-ids="highlightedChunkIds"
        :is-uploading="isUploading"
        :upload-progress="uploadProgress"
        @toggle-file="toggleFile"
        @upload="handleUpload"
        @delete-file="handleDeleteFile"
      />
    </div>

    <!-- Center: Chat (45%) -->
    <div class="flex-1">
      <ChatPanel
        :messages="messages"
        :is-rag-enabled="isRagEnabled"
        :is-thinking="isThinking"
        :chunk-map="chunkMap"
        @toggle-rag="toggleRag"
        @send-message="handleSendMessage"
        @citation-hover="handleCitationHover"
      />
    </div>

    <!-- Right: Brain (30%) -->
    <div class="w-[420px]">
      <BrainPanel
        :logs="logs"
        :current-similarity="currentSimilarity"
        :last-prompt="lastPrompt"
        :rag-status="currentStatus"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import LibraryPanel from './components/LibraryPanel.vue';
import ChatPanel from './components/ChatPanel.vue';
import BrainPanel from './components/BrainPanel.vue';
import { useDocuments } from './composables/useDocuments';
import { useChat } from './composables/useChat';
import { useSSEStream } from './composables/useSSEStream';
import { INITIAL_LOGS_WELCOME } from './constants';
import type { LogEntry } from './types';

// 全局日志状态
const logs = ref<LogEntry[]>([...INITIAL_LOGS_WELCOME]);

// 文档管理
const {
  files,
  isUploading,
  uploadProgress,
  highlightedChunkIds,
  toggleFile,
  uploadDocuments,
  deleteDocument,
  setHighlightedChunks,
  clearHighlights
} = useDocuments(logs);

// 聊天功能（传入 chunk 高亮回调）
const {
  messages,
  isThinking,
  isRagEnabled,
  currentSimilarity,
  lastPrompt,
  highlightedChunkId,
  sendMessage,
  toggleRag,
  handleCitationHover
} = useChat(logs, (chunkIds: string[]) => {
  // 🎯 接收到citations时立即高亮
  console.log('[App] 高亮chunks:', chunkIds);
  setHighlightedChunks(chunkIds);
});

// SSE 流状态
const { currentStatus } = useSSEStream();

// 构建 chunkId → { content, sourceName } 映射，供 ChatPanel 显示悬浮卡片
const chunkMap = computed(() => {
  const map: Record<string, { content: string; sourceName: string }> = {};
  for (const file of files.value) {
    for (const chunk of file.chunks) {
      map[chunk.id] = { content: chunk.content, sourceName: file.name };
    }
  }
  return map;
});

// 事件处理
const handleUpload = (selectedFiles: File[]) => {
  uploadDocuments(selectedFiles);
};

const handleSendMessage = (text: string) => {
  // 新消息时清除旧的高亮
  clearHighlights();
  sendMessage(text);
};

const handleDeleteFile = (fileName: string) => {
  deleteDocument(fileName);
};
</script>
