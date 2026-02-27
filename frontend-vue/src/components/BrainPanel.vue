<template>
  <div class="flex flex-col h-full bg-deep-900 border-l border-deep-800 font-mono text-sm">
    <!-- Header -->
    <div class="p-4 border-b border-deep-800 flex justify-between items-center bg-deep-950/50 backdrop-blur">
      <h2 class="font-bold tracking-wider text-base text-neon-blue flex items-center gap-2">
        <Terminal :size="18" />
        运行日志
      </h2>
      <div class="flex items-center gap-2">
        <div
          :class="[
            'w-2 h-2 rounded-full',
            ragStatus === 'IDLE' ? 'bg-slate-500' : 'bg-neon-green animate-pulse'
          ]"
        />
        <span class="text-slate-500 text-sm">{{ ragStatus }}</span>
      </div>
    </div>

    <!-- Similarity Gauge (纯 SVG，无第三方依赖) -->
    <div class="p-4 border-b border-deep-800 bg-deep-950/30">
      <div class="flex items-center justify-between mb-2">
        <span class="text-slate-400 tracking-widest text-sm">检索质量</span>
        <Cpu :size="18" class="text-slate-600" />
      </div>

      <div class="h-32 relative flex items-center justify-center">
        <!-- SVG 半圆仪表盘 -->
        <svg viewBox="0 0 120 70" class="w-full max-w-[180px]">
          <!-- 背景轨道 -->
          <path
            d="M 10 65 A 50 50 0 0 1 110 65"
            fill="none"
            stroke="#1a1f2e"
            stroke-width="10"
            stroke-linecap="round"
          />
          <!-- 彩色进度弧 -->
          <path
            d="M 10 65 A 50 50 0 0 1 110 65"
            fill="none"
            :stroke="gaugeColor"
            stroke-width="10"
            stroke-linecap="round"
            :stroke-dasharray="`${gaugeArcLength} 157`"
            style="transition: stroke-dasharray 0.6s ease, stroke 0.6s ease;"
          />
        </svg>

        <!-- 数值显示，居中叠在 SVG 上 -->
        <div class="absolute bottom-2 left-1/2 -translate-x-1/2 text-center">
          <div class="text-3xl font-bold text-slate-100" style="transition: all 0.3s ease;">
            {{ similarityPercent }}%
          </div>
          <div class="text-xs text-slate-500">相似度</div>
        </div>
      </div>
    </div>

    <!-- Log Terminal -->
    <div ref="logsContainer" class="flex-1 overflow-y-auto p-4 space-y-2 bg-black/40 font-mono">
      <div v-for="log in logs" :key="log.id" class="flex gap-2">
        <span class="text-slate-600 shrink-0 text-sm">
          {{ formatTime(log.timestamp) }}
        </span>
        <div class="flex-1 break-words">
          <span :class="['font-bold mr-2', getStageColor(log.stage)]">
            [{{ log.stage }}]
          </span>
          <span :class="getTypeColor(log.type)">
            {{ log.message }}
          </span>
          <div
            v-if="log.details"
            class="ml-4 mt-1 text-slate-500 text-xs border-l border-slate-700 pl-2"
          >
            {{ log.details }}
          </div>
        </div>
      </div>
    </div>

    <!-- Prompt Inspector -->
    <div class="border-t border-deep-800 bg-deep-950">
      <button
        @click="isPromptOpen = !isPromptOpen"
        class="w-full p-3 flex items-center justify-between text-slate-400 hover:text-slate-200 hover:bg-deep-800 transition-colors"
      >
        <div class="flex items-center gap-2">
          <Code :size="18" />
          <span class="text-sm tracking-wider">提示词检视</span>
        </div>
        <component :is="isPromptOpen ? ChevronDown : ChevronRight" :size="16" />
      </button>

      <div
        v-if="isPromptOpen"
        class="p-4 bg-black/60 border-t border-deep-800 h-48 overflow-y-auto"
      >
        <pre class="whitespace-pre-wrap text-xs text-green-400/80 font-mono">{{ lastPrompt || '// 暂无提示词...' }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';
import { Terminal, Code, ChevronDown, ChevronRight, Cpu } from 'lucide-vue-next';
import type { LogEntry } from '@/types';

interface Props {
  logs: LogEntry[];
  currentSimilarity: number;
  lastPrompt: string | null;
  ragStatus: string;
}

const props = defineProps<Props>();

const isPromptOpen = ref(false);
const logsContainer = ref<HTMLDivElement>();

// 相似度百分比（0-100）
const similarityPercent = computed(() => Math.round(props.currentSimilarity * 100));

// SVG 半圆弧长：半径50，弧度180° = π*50 ≈ 157
// 进度弧长 = 157 * percent / 100
const gaugeArcLength = computed(() => (157 * similarityPercent.value) / 100);

// 根据分数决定颜色
const gaugeColor = computed(() => {
  const p = similarityPercent.value;
  if (p >= 70) return '#06b6d4'; // neon-blue：高相似度
  if (p >= 30) return '#f59e0b'; // 橙色：中等
  return '#ef4444';              // 红色：低相似度
});

// 日志自动滚动到底部
watch(() => props.logs.length, async () => {
  await nextTick();
  logsContainer.value?.scrollTo({
    top: logsContainer.value.scrollHeight,
    behavior: 'smooth'
  });
});

const formatTime = (timestamp: number): string => {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

const getStageColor = (stage: string): string => {
  const colors: Record<string, string> = {
    USER: 'text-blue-400',
    EMBED: 'text-cyan-400',
    SEARCH: 'text-cyan-400',
    HIT: 'text-neon-green',
    GEN: 'text-purple-400',
    SYS: 'text-slate-400',
    UPLOAD: 'text-yellow-400',
    PARSE: 'text-orange-400',
    CHUNK: 'text-pink-400',
    STORE: 'text-green-400'
  };
  return colors[stage] || 'text-slate-400';
};

const getTypeColor = (type: string): string => {
  const colors: Record<string, string> = {
    error: 'text-red-500',
    success: 'text-emerald-400',
    warning: 'text-yellow-500',
    info: 'text-slate-300',
    process: 'text-slate-400'
  };
  return colors[type] || 'text-slate-300';
};
</script>
