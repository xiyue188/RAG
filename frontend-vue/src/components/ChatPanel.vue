<template>
  <div class="flex flex-col h-full bg-deep-950 relative overflow-hidden" @click="closeCitationCard">
    <!-- Background Grid Effect -->
    <div
      class="absolute inset-0 opacity-10 pointer-events-none"
      style="
        background-image: linear-gradient(#1e293b 1px, transparent 1px),
          linear-gradient(90deg, #1e293b 1px, transparent 1px);
        background-size: 40px 40px;
      "
    />

    <!-- Header / Mode Switch -->
    <div class="relative z-10 p-4 border-b border-deep-800 flex justify-between items-center bg-deep-950/80 backdrop-blur">
      <div class="flex items-center gap-3">
        <div class="w-2 h-2 rounded-full bg-neon-green animate-pulse" />
        <h1 class="font-bold text-lg text-slate-100 tracking-tight">深蓝智脑</h1>
      </div>

      <!-- RAG Toggle -->
      <button
        @click.stop="$emit('toggleRag')"
        :class="[
          'flex items-center gap-3 px-4 py-2 rounded-full border transition-all duration-300',
          isRagEnabled
            ? 'bg-neon-blue/10 border-neon-blue text-neon-blue shadow-[0_0_15px_rgba(6,182,212,0.3)]'
            : 'bg-deep-800 border-deep-700 text-slate-400 hover:border-slate-500'
        ]"
      >
        <span class="text-sm font-mono font-bold">
          {{ isRagEnabled ? 'RAG系统：在线' : 'RAG系统：离线' }}
        </span>
        <component :is="isRagEnabled ? Zap : Shield" :size="16" :class="{ 'fill-current': isRagEnabled }" />
      </button>
    </div>

    <!-- Messages Area -->
    <div ref="messagesContainer" class="flex-1 overflow-y-auto p-6 space-y-6 relative z-10">
      <div
        v-for="msg in messages"
        :key="msg.id"
        :class="['flex gap-4', msg.role === 'user' ? 'justify-end' : 'justify-start']"
      >
        <!-- AI Avatar -->
        <div
          v-if="msg.role === 'ai'"
          class="w-8 h-8 rounded bg-deep-800 border border-deep-700 flex items-center justify-center shrink-0"
        >
          <Bot :size="20" class="text-neon-blue" />
        </div>

        <!-- Message Bubble -->
        <div
          :class="[
            'max-w-[80%] rounded-lg p-4 border shadow-xl',
            msg.role === 'user'
              ? 'bg-deep-800 border-deep-700 text-slate-200'
              : 'bg-black/40 border-deep-800 text-slate-300'
          ]"
        >
          <!-- AI Status Badge -->
          <div
            v-if="msg.role === 'ai' && !msg.ragEnabled"
            class="flex items-center gap-2 mb-2 text-amber-500/80 text-xs font-mono border-b border-amber-500/20 pb-1"
          >
            <AlertTriangle :size="12" />
            通用知识
          </div>
          <div
            v-if="msg.role === 'ai' && msg.ragEnabled"
            class="flex items-center gap-2 mb-2 text-neon-blue/80 text-xs font-mono border-b border-neon-blue/20 pb-1"
          >
            <Shield :size="12" />
            知识库检索
          </div>

          <!-- Message Content with Citations -->
          <div class="text-sm leading-relaxed whitespace-pre-wrap">
            <template v-for="(part, i) in parseMessageContent(msg.content, msg.citations, msg.citationDetails, props.chunkMap)" :key="i">
              <span
                v-if="part.type === 'text'"
                v-html="part.content"
              />
              <!-- 来源角标：可点击，弹出悬浮卡片 -->
              <span
                v-else
                :class="[
                  'inline-block mx-1 text-xs font-bold cursor-pointer rounded px-1.5 py-0.5 transition-all duration-200 select-none',
                  activeCitation?.chunkId === part.chunkId
                    ? 'bg-neon-blue text-deep-950 shadow-[0_0_8px_rgba(6,182,212,0.5)]'
                    : 'text-neon-blue border border-neon-blue/40 hover:bg-neon-blue/20 hover:border-neon-blue'
                ]"
                @click.stop="handleCitationClick($event, part)"
                @mouseenter="part.chunkId && $emit('citationHover', part.chunkId)"
                @mouseleave="$emit('citationHover', null)"
                :title="part.chunkId ? '点击查看文档证据' : ''"
              >
                {{ part.content }}
              </span>
            </template>
          </div>
        </div>

        <!-- User Avatar -->
        <div
          v-if="msg.role === 'user'"
          class="w-8 h-8 rounded bg-slate-700 border border-slate-600 flex items-center justify-center shrink-0"
        >
          <User :size="20" class="text-slate-300" />
        </div>
      </div>

      <!-- Thinking Indicator -->
      <div v-if="isThinking" class="flex gap-4 justify-start animate-pulse">
        <div class="w-8 h-8 rounded bg-deep-800 border border-deep-700 flex items-center justify-center shrink-0">
          <Bot :size="20" class="text-neon-blue" />
        </div>
        <div class="bg-black/40 border border-deep-800 rounded-lg p-4 flex items-center gap-2">
          <div
            v-for="i in 3"
            :key="i"
            class="w-2 h-2 bg-neon-blue rounded-full animate-bounce"
            :style="{ animationDelay: `${(i - 1) * 150}ms` }"
          />
        </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="p-4 bg-deep-900 border-t border-deep-800 relative z-10">
      <form @submit.prevent="handleSubmit" class="flex gap-3 relative">
        <input
          v-model="inputText"
          type="text"
          :disabled="isThinking"
          :placeholder="isRagEnabled ? '请提问关于文档的问题...' : '请提问通用问题...'"
          class="flex-1 bg-deep-950 border border-deep-700 rounded-lg px-4 py-3 text-base text-slate-200 focus:outline-none focus:border-neon-blue focus:shadow-[0_0_15px_rgba(6,182,212,0.1)] transition-all placeholder:text-slate-600"
        />
        <button
          type="submit"
          :disabled="!inputText.trim() || isThinking"
          class="bg-neon-blue hover:bg-cyan-400 text-deep-950 font-bold p-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send :size="20" />
        </button>
      </form>
    </div>

    <!-- 来源悬浮卡片（Teleport 到 body，避免被 overflow:hidden 裁切） -->
    <Teleport to="body">
      <Transition name="citation-card">
        <div
          v-if="activeCitation"
          class="fixed z-[9999] w-96 max-w-[calc(100vw-2rem)] bg-deep-900 border border-neon-blue/40 rounded-xl shadow-2xl shadow-black/60 overflow-hidden"
          :style="cardPositionStyle"
          @click.stop
        >
          <!-- 卡片顶部：来源信息 -->
          <div class="flex items-center justify-between px-4 py-3 bg-neon-blue/10 border-b border-neon-blue/20">
            <div class="flex items-center gap-2 min-w-0">
              <FileText :size="14" class="text-neon-blue shrink-0" />
              <span class="text-xs font-mono text-neon-blue truncate" :title="activeCitation.sourceName">
                {{ activeCitation.sourceName }}
              </span>
              <span class="text-xs text-slate-500 shrink-0">·</span>
              <span class="text-xs text-slate-500 shrink-0">{{ activeCitation.index }}号引用</span>
            </div>
            <button
              @click="closeCitationCard"
              class="w-6 h-6 flex items-center justify-center rounded hover:bg-deep-700 transition-colors text-slate-400 hover:text-slate-200 shrink-0 ml-2"
              title="关闭"
            >
              <X :size="14" />
            </button>
          </div>

          <!-- 卡片主体：文档内容 -->
          <div class="p-4 max-h-64 overflow-y-auto">
            <p class="text-sm text-slate-300 leading-relaxed font-mono whitespace-pre-wrap">{{ activeCitation.content }}</p>
          </div>

          <!-- 卡片底部：chunk ID -->
          <div class="px-4 py-2 bg-black/30 border-t border-deep-800">
            <span class="text-xs text-slate-600 font-mono">chunk: {{ activeCitation.chunkId }}</span>
          </div>

          <!-- 向上三角箭头（指向角标） -->
          <div
            class="absolute -top-2 border-8 border-transparent border-b-neon-blue/40"
            :style="{ left: `${arrowLeft}px` }"
          />
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue';
import { Send, Zap, Shield, User, Bot, AlertTriangle, FileText, X } from 'lucide-vue-next';
import type { Message, CitationDetail } from '@/types';

// ─── Types ────────────────────────────────────────────────────────────────────

interface ChunkInfo {
  content: string;
  sourceName: string;
}

interface Props {
  messages: Message[];
  isRagEnabled: boolean;
  isThinking: boolean;
  /** chunkId → { content, sourceName } 映射（降级用，主要路径是 citationDetails） */
  chunkMap?: Record<string, ChunkInfo>;
}

interface Emits {
  (e: 'toggleRag'): void;
  (e: 'sendMessage', text: string): void;
  (e: 'citationHover', id: string | null): void;
}

interface ContentPart {
  type: 'text' | 'citation';
  /** 标记原文，如 "[来源: filename.md]" 或 "[1]" */
  content: string;
  chunkId?: string;
  /** 角标序号（从1开始） */
  index?: number;
  /** 引用内容（直接来自后端 citationDetails，精准对应每条引用） */
  citationContent?: string;
  /** 引用文件名（来自后端 citationDetails） */
  citationFile?: string;
}

interface ActiveCitation {
  chunkId: string;
  content: string;
  sourceName: string;
  index: number;
  triggerRect: DOMRect;
}

// ─── Props & Emits ─────────────────────────────────────────────────────────────

const props = defineProps<Props>();
const emit = defineEmits<Emits>();

// ─── State ────────────────────────────────────────────────────────────────────

const inputText = ref('');
const messagesContainer = ref<HTMLDivElement>();
const activeCitation = ref<ActiveCitation | null>(null);

// ─── 卡片定位计算（fixed，相对于 viewport）────────────────────────────────────

/** 卡片宽度（px），需与模板宽度 w-96(384px) 对应 */
const CARD_WIDTH = 384;
/** 卡片出现在角标下方的间距 */
const CARD_OFFSET_Y = 10;

const cardPositionStyle = computed(() => {
  if (!activeCitation.value) return {};
  const r = activeCitation.value.triggerRect;

  // 水平：优先居中于角标，超出屏幕右侧则靠右
  let left = r.left + r.width / 2 - CARD_WIDTH / 2;
  left = Math.max(8, Math.min(left, window.innerWidth - CARD_WIDTH - 8));

  // 垂直：默认在角标下方；空间不足时改为上方
  const spaceBelow = window.innerHeight - r.bottom;
  const top = spaceBelow > 280
    ? r.bottom + CARD_OFFSET_Y
    : r.top - CARD_OFFSET_Y - 300; // 大约 max-h-64 + header + footer

  return { left: `${left}px`, top: `${Math.max(8, top)}px` };
});

/** 三角箭头在卡片内的水平偏移 */
const arrowLeft = computed(() => {
  if (!activeCitation.value) return 16;
  const r = activeCitation.value.triggerRect;
  const cardLeft = parseFloat(cardPositionStyle.value.left || '0');
  return Math.max(8, r.left + r.width / 2 - cardLeft - 8);
});

// ─── Citation 点击处理 ─────────────────────────────────────────────────────────

const handleCitationClick = (event: MouseEvent, part: ContentPart) => {
  const el = event.currentTarget as HTMLElement;
  const triggerRect = el.getBoundingClientRect();
  const chunkId = part.chunkId || '';

  // 点击同一个角标 → 关闭
  if (activeCitation.value?.chunkId === chunkId && chunkId) {
    closeCitationCard();
    return;
  }

  // 优先1：后端直接发来的 citationDetails（精准、不受路径格式影响）
  let content = part.citationContent || '';
  let sourceName = part.citationFile || '';

  // 优先2：chunkMap（降级，通过 chunkId 查找）
  if (!content && chunkId && props.chunkMap?.[chunkId]) {
    const info = props.chunkMap[chunkId];
    content = info.content;
    sourceName = sourceName || info.sourceName;
  }

  if (!content) {
    // 无内容可显示，只触发高亮
    if (chunkId) emit('citationHover', chunkId);
    return;
  }

  activeCitation.value = {
    chunkId,
    content,
    sourceName,
    index: part.index ?? 0,
    triggerRect,
  };

  // 同步触发左侧面板高亮
  if (chunkId) emit('citationHover', chunkId);
};

const closeCitationCard = () => {
  activeCitation.value = null;
};

// ─── 消息内容解析 ──────────────────────────────────────────────────────────────

/**
 * 解析消息内容，将引用标记拆分为可点击的 citation 部分。
 *
 * 支持两种后端格式：
 *   [来源: filename.md]  — _stream_with_citations 输出（主路径）
 *   [1] [2] ...         — 备用数字角标格式
 *
 * citationDetails 按引用出现顺序一一对应（后端不去重），
 * 直接携带 content 和 file 信息，无需通过 chunkMap 路径匹配。
 */
const parseMessageContent = (
  content: string,
  citations?: string[],
  citationDetails?: CitationDetail[],
  chunkMap?: Record<string, ChunkInfo>
): ContentPart[] => {
  const parts: ContentPart[] = [];
  const regex = /(\[来源:\s*[^\]]+\]|\[\d+\])/g;
  let lastIndex = 0;
  let match;
  let citationCounter = 0;

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: content.substring(lastIndex, match.index) });
    }

    let chunkId: string | undefined;
    let index: number;
    let citationContent: string | undefined;
    let citationFile: string | undefined;

    if (match[0].startsWith('[来源:')) {
      // 按出现顺序从 citationDetails 取（精准，不去重，与文本标记一一对应）
      const detail = citationDetails?.[citationCounter];
      chunkId = detail?.chunkId ?? citations?.[citationCounter];
      citationContent = detail?.content;
      citationFile = detail?.file;

      // 降级：chunkId 存在但 content 为空时，从 chunkMap 补充
      if (!citationContent && chunkId && chunkMap?.[chunkId]) {
        citationContent = chunkMap[chunkId].content;
        citationFile = citationFile || chunkMap[chunkId].sourceName;
      }

      index = citationCounter + 1;
      citationCounter++;
    } else {
      // [1] 数字角标格式
      const n = parseInt(match[0].replace(/[\[\]]/g, ''));
      const detail = citationDetails?.[n - 1];
      chunkId = detail?.chunkId ?? citations?.[n - 1];
      citationContent = detail?.content;
      citationFile = detail?.file;
      index = n;
    }

    parts.push({ type: 'citation', content: match[0], chunkId, index, citationContent, citationFile });
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push({ type: 'text', content: content.substring(lastIndex) });
  }

  return parts;
};

// ─── 自动滚动 ──────────────────────────────────────────────────────────────────

// 深度监听消息内容变化（流式输出时内容在变但数量不变）
watch(() => props.messages, async () => {
  await nextTick();
  const el = messagesContainer.value;
  if (el) el.scrollTop = el.scrollHeight;
}, { deep: true });

// ─── 输入处理 ──────────────────────────────────────────────────────────────────

const handleSubmit = () => {
  if (inputText.value.trim() && !props.isThinking) {
    closeCitationCard();
    emit('sendMessage', inputText.value);
    inputText.value = '';
  }
};
</script>

<style scoped>
/* 悬浮卡片入场/离场动画 */
.citation-card-enter-active,
.citation-card-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.citation-card-enter-from,
.citation-card-leave-to {
  opacity: 0;
  transform: translateY(-6px) scale(0.97);
}
</style>
