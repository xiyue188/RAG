<template>
  <div class="flex flex-col h-full bg-deep-900 border-r border-deep-800 text-slate-300">
    <!-- Header -->
    <div class="p-4 border-b border-deep-800 flex justify-between items-center bg-deep-950/50 backdrop-blur">
      <h2 class="font-mono text-base font-bold tracking-wider text-neon-blue flex items-center gap-2">
        <Layers :size="18" />
        知识库
      </h2>
      <span class="text-sm bg-deep-800 px-2 py-0.5 rounded text-slate-500 font-mono">
        {{ enabledCount }}/{{ files.length }} 激活
      </span>
    </div>

    <!-- Upload Zone -->
    <div class="p-4">
      <label
        class="w-full h-24 border-2 border-dashed border-deep-700 hover:border-neon-blue/50 rounded-lg flex flex-col items-center justify-center gap-2 transition-all group bg-deep-950/30 cursor-pointer"
      >
        <UploadCloud :size="24" class="text-slate-500 group-hover:text-neon-blue transition-colors" />
        <span class="text-sm text-slate-500 font-mono group-hover:text-slate-300">
          上传文档
        </span>
        <input
          type="file"
          multiple
          accept=".pdf,.txt,.md,.docx"
          class="hidden"
          @change="handleFileSelect"
          :disabled="isUploading"
        />
      </label>

      <!-- Upload Progress -->
      <div v-if="isUploading" class="mt-3">
        <div class="flex justify-between text-xs text-slate-500 mb-1">
          <span>Uploading...</span>
          <span>{{ uploadProgress }}%</span>
        </div>
        <div class="w-full h-2 bg-deep-800 rounded-full overflow-hidden">
          <div
            class="h-full bg-neon-blue transition-all duration-300"
            :style="{ width: uploadProgress + '%' }"
          />
        </div>
      </div>
    </div>

    <!-- File List -->
    <div class="flex-1 overflow-y-auto px-4 pb-4 space-y-3">
      <div
        v-for="file in files"
        :key="file.id"
        :class="[
          'rounded border transition-all duration-300',
          file.isEnabled
            ? 'border-deep-700 bg-deep-800/30'
            : 'border-deep-800 bg-deep-950 opacity-60'
        ]"
      >
        <!-- File Header -->
        <div class="p-3 flex items-center gap-3">
          <component :is="getFileIcon(file.type)" :size="16" :class="getFileIconColor(file.type)" />

          <div
            class="flex-1 min-w-0 cursor-pointer"
            @click="toggleExpand(file.id)"
          >
            <div class="text-sm font-semibold truncate text-slate-200">{{ file.name }}</div>
            <div class="text-xs text-slate-500 font-mono">
              <span>{{ file.chunks.length }} 片段</span>
            </div>
          </div>

          <!-- Toggle Switch -->
          <button
            @click="$emit('toggleFile', file.id)"
            :class="[
              'w-8 h-4 rounded-full relative transition-colors',
              file.isEnabled ? 'bg-neon-blue/20' : 'bg-slate-700'
            ]"
          >
            <div
              :class="[
                'absolute top-0.5 w-3 h-3 rounded-full transition-all',
                file.isEnabled
                  ? 'left-4 bg-neon-blue shadow-[0_0_8px_rgba(6,182,212,0.8)]'
                  : 'left-0.5 bg-slate-400'
              ]"
            />
          </button>

          <!-- Delete Button -->
          <button
            @click.stop="$emit('deleteFile', file.name)"
            class="p-1 rounded hover:bg-red-500/20 transition-colors group"
            title="Delete document"
          >
            <Trash2
              :size="14"
              class="text-slate-500 group-hover:text-red-400 transition-colors"
            />
          </button>
        </div>

        <!-- Expanded Chunk View -->
        <div
          v-if="expandedFileIds.has(file.id) || file.chunks.some(c => highlightedChunkIds?.has(c.id))"
          class="border-t border-deep-800 bg-black/20 p-2 space-y-1"
        >
          <div class="text-xs uppercase text-slate-500 font-mono mb-2 px-1">文档片段</div>
          <div
            v-for="(chunk, idx) in file.chunks"
            :key="chunk.id"
            :class="[
              'text-xs p-2 rounded font-mono leading-relaxed transition-all duration-500',
              highlightedChunkIds?.has(chunk.id)
                ? 'bg-neon-blue/20 border border-neon-blue/50 text-neon-blue shadow-[0_0_15px_rgba(6,182,212,0.2)]'
                : 'text-slate-500 border border-transparent hover:bg-deep-800'
            ]"
          >
            <span class="opacity-50 select-none mr-2">[{{ idx + 1 }}]</span>
            {{ chunk.content.substring(0, 80) }}...
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { Layers, UploadCloud, FileText, FileCode, File, Trash2 } from 'lucide-vue-next';
import type { KnowledgeFile } from '@/types';

interface Props {
  files: KnowledgeFile[];
  highlightedChunkIds?: Set<string>;  // 🎯 改为Set支持多个chunk高亮
  isUploading: boolean;
  uploadProgress: number;
}

interface Emits {
  (e: 'toggleFile', id: string): void;
  (e: 'upload', files: File[]): void;
  (e: 'deleteFile', fileName: string): void;
}

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

// 🎯 改为Set，支持多个文档同时展开
const expandedFileIds = ref<Set<string>>(new Set());

const enabledCount = computed(() => props.files.filter(f => f.isEnabled).length);

const toggleExpand = (id: string) => {
  // 切换展开/折叠状态
  if (expandedFileIds.value.has(id)) {
    expandedFileIds.value.delete(id);
  } else {
    expandedFileIds.value.add(id);
  }
  // 触发响应式更新
  expandedFileIds.value = new Set(expandedFileIds.value);
};

const handleFileSelect = (event: Event) => {
  const target = event.target as HTMLInputElement;
  if (target.files && target.files.length > 0) {
    emit('upload', Array.from(target.files));
    target.value = ''; // 重置输入
  }
};

const getFileIcon = (type: string) => {
  switch (type) {
    case 'pdf': return FileText;
    case 'md': return FileCode;
    default: return File;
  }
};

const getFileIconColor = (type: string) => {
  switch (type) {
    case 'pdf': return 'text-red-400';
    case 'txt': return 'text-blue-400';
    case 'md': return 'text-yellow-400';
    default: return 'text-slate-400';
  }
};
</script>
