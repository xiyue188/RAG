# DeepBlue Command Deck - Vue 3 Frontend

基于 Vue 3 + TypeScript + Vite 构建的 RAG 系统前端界面。

## 技术栈

- **Vue 3** - 使用 Composition API
- **TypeScript** - 类型安全
- **Vite** - 快速构建工具
- **Tailwind CSS** - 实用优先的 CSS 框架
- **Lucide Vue Next** - 图标库
- **ECharts + vue-echarts** - 相似度仪表盘可视化

## 项目结构

```
frontend-vue/
├── src/
│   ├── components/          # Vue 组件
│   │   ├── LibraryPanel.vue    # 文档管理面板
│   │   ├── ChatPanel.vue       # 聊天界面面板
│   │   └── BrainPanel.vue      # 日志监控面板
│   ├── composables/         # Vue Composables
│   │   ├── useSSEStream.ts     # SSE 流处理逻辑
│   │   ├── useChat.ts          # 聊天功能逻辑
│   │   └── useDocuments.ts     # 文档管理逻辑
│   ├── services/            # API 服务层
│   │   └── api.ts              # FastAPI 后端接口
│   ├── types/               # TypeScript 类型定义
│   │   └── index.ts
│   ├── constants/           # 常量配置
│   │   └── index.ts
│   ├── App.vue              # 主应用组件
│   ├── main.ts              # 应用入口
│   └── style.css            # 全局样式
├── public/                  # 静态资源
├── index.html               # HTML 入口
├── vite.config.ts           # Vite 配置
├── tsconfig.json            # TypeScript 配置
├── tailwind.config.js       # Tailwind 配置
└── package.json             # 项目依赖
```

## 功能特性

### 三栏布局

1. **LibraryPanel (左侧 20%)**
   - 文档列表展示
   - 文件启用/禁用切换
   - 拖拽上传文档
   - Chunk 高亮显示

2. **ChatPanel (中间 50%)**
   - 实时消息流
   - RAG 模式切换
   - 引用标记解析 `[1] [2]`
   - 引用悬停高亮

3. **BrainPanel (右侧 30%)**
   - 实时日志流（31 种 SSE 事件）
   - 相似度仪表盘（ECharts 圆形进度条）
   - RAG 状态指示器
   - Prompt Inspector（可折叠）

### SSE 事件支持

完整支持后端的 31 种 SSE 事件：

**查询阶段（17 种）**
- query_received, session_start, conversation_context_loaded
- hybrid_check_start, query_rewrite_start, query_rewrite_done
- embedding_start, embedding_done
- retrieval_start, retrieval_done
- rerank_start, rerank_done
- generation_start, generation_chunk, generation_done
- citation_generated, answer_complete

**摄入阶段（14 种）**
- upload_start, file_received
- parsing_start, parsing_done
- chunking_start, chunking_done
- embedding_start_ingestion, embedding_progress, embedding_done_ingestion
- storing_start, storing_done, indexing_done
- upload_complete, all_complete

## 开发指南

### 安装依赖

```bash
cd frontend-vue
npm install
```

### 启动开发服务器

```bash
npm run dev
```

前端将在 http://localhost:3000 启动，并自动代理 `/api` 请求到后端 `http://localhost:8001`。

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## API 集成

### 后端接口

- `POST /api/v1/chat/stream` - SSE 流式聊天
- `POST /api/v1/documents/upload/stream` - SSE 流式文档上传
- `GET /api/v1/documents` - 获取文档列表
- `DELETE /api/v1/documents/{doc_id}` - 删除文档

### 配置代理

在 `vite.config.ts` 中已配置：

```typescript
proxy: {
  '/api': {
    target: 'http://localhost:8001',
    changeOrigin: true
  }
}
```

## 颜色主题

**DeepBlue Command Deck 配色方案**

- `deep-950`: #0a0e1a（深色背景）
- `deep-900`: #0f1420（面板背景）
- `deep-800`: #1a1f2e（边框/次级背景）
- `neon-blue`: #06b6d4（主题蓝色）
- `neon-green`: #10b981（成功/在线状态）
- `neon-purple`: #a855f7（生成阶段）

## 最佳实践

1. **Composables 优先** - 使用 Vue 3 Composables 进行逻辑复用
2. **类型安全** - 所有接口和数据结构都有 TypeScript 类型定义
3. **响应式设计** - 使用 Tailwind CSS 实现响应式布局
4. **组件解耦** - 通过 props 和 emits 进行组件通信
5. **性能优化** - SSE 流式处理，避免阻塞 UI

## 开发注意事项

1. 确保后端服务已启动在 `http://localhost:8001`
2. 上传文件支持格式：`.pdf`, `.txt`, `.md`, `.docx`
3. 日志会自动滚动到底部
4. 引用悬停会高亮对应的文档 chunk

## License

MIT
