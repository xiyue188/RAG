# DeepBlue RAG 系统部署指南

## 📋 目录

1. [快速开始](#快速开始)
2. [系统要求](#系统要求)
3. [本地部署](#本地部署)
4. [云服务器部署](#云服务器部署)
5. [配置说明](#配置说明)
6. [常见问题](#常见问题)
7. [维护与监控](#维护与监控)

---

## 🚀 快速开始

### 一键部署（推荐）

```bash
# Linux/Mac
./deploy.sh

# Windows
deploy.bat
```

**预期结果**：
- ✅ 后端 API: http://localhost:8000
- ✅ API 文档: http://localhost:8000/docs
- ✅ 前端界面: http://localhost:3000

---

## 💻 系统要求

### 硬件要求

| 配置项 | 最低配置 | 推荐配置 |
|--------|----------|----------|
| CPU | 2核 | 4核 |
| 内存 | 4GB | 8GB |
| 硬盘 | 10GB | 20GB+ |
| 网络 | 1Mbps | 10Mbps+ |

### 软件要求

- **Docker**: 20.10+ ([安装指南](https://docs.docker.com/get-docker/))
- **Docker Compose**: 1.29+ (Docker Desktop 自带)
- **操作系统**:
  - Linux (Ubuntu 20.04+)
  - macOS (10.15+)
  - Windows 10/11 (需启用 WSL2)

### API 密钥

至少需要以下一个 LLM 提供商的 API 密钥：

- 智谱AI（推荐）: https://open.bigmodel.cn/
- 阿里通义: https://dashscope.console.aliyun.com/
- OpenAI: https://platform.openai.com/
- Anthropic: https://console.anthropic.com/

---

## 🏠 本地部署

### 步骤 1: 克隆项目

```bash
git clone <your-repo-url>
cd rag-project
```

### 步骤 2: 配置环境变量

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件
nano .env
```

**必填配置**：

```env
# LLM 配置（至少选择一个）
LLM_PROVIDER=zhipuai                    # zhipuai / qwen / openai / anthropic
ZHIPUAI_API_KEY=your_api_key_here      # 智谱AI密钥

# 可选配置
TOP_K_RESULTS=5                         # 检索结果数量
CHUNK_SIZE=800                          # 文档切片大小
CHUNK_OVERLAP=200                       # 切片重叠大小
```

### 步骤 3: 一键部署

```bash
# Linux/Mac
chmod +x deploy.sh
./deploy.sh

# Windows
deploy.bat
```

### 步骤 4: 访问系统

打开浏览器访问：

- **前端界面**: http://localhost:3000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/health

---

## ☁️ 云服务器部署

### 方案 A: 阿里云轻量应用服务器

#### 1. 购买服务器

推荐配置：
- **CPU**: 2核
- **内存**: 2GB
- **带宽**: 3Mbps
- **系统**: Ubuntu 20.04
- **费用**: ¥99/年

购买地址: https://www.aliyun.com/product/swas

#### 2. 连接服务器

```bash
ssh root@your_server_ip
```

#### 3. 安装 Docker

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 启动 Docker
systemctl start docker
systemctl enable docker

# 安装 Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

#### 4. 部署应用

```bash
# 克隆项目
git clone <your-repo-url>
cd rag-project

# 配置环境变量
cp .env.example .env
nano .env

# 部署
./deploy.sh
```

#### 5. 配置防火墙

```bash
# 开放端口
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 3000/tcp
ufw allow 8000/tcp
```

#### 6. 配置域名（可选）

如果你有域名，可以配置 Nginx 反向代理：

```nginx
# /etc/nginx/sites-available/deepblue
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

---

## ⚙️ 配置说明

### 环境变量详解

```env
# ==================== LLM 配置 ====================
LLM_PROVIDER=zhipuai              # LLM 提供商
ZHIPUAI_API_KEY=xxx               # 智谱AI密钥
QWEN_API_KEY=xxx                  # 通义千问密钥（可选）
OPENAI_API_KEY=xxx                # OpenAI密钥（可选）

LLM_MODEL=glm-4-flash             # 模型名称
LLM_TEMPERATURE=0.7               # 温度参数 (0.0-1.0)
LLM_MAX_TOKENS=1000               # 最大生成长度

# ==================== RAG 配置 ====================
TOP_K_RESULTS=5                   # 检索K个最相关结果
SIMILARITY_THRESHOLD=0.3          # 相似度阈值 (0.0-1.0)

CHUNK_SIZE=800                    # 文档切片大小
CHUNK_OVERLAP=200                 # 切片重叠大小

ENABLE_HYBRID=true                # 启用混合检索
ENABLE_MULTI_QUERY=true           # 启用多查询
ENABLE_RERANK=false               # 启用重排序（可选）

# ==================== 向量模型 ====================
EMBEDDING_MODEL=bge-small-zh-v1.5 # Embedding模型
EMBEDDING_DEVICE=cpu              # 设备: cpu / cuda

# ==================== 数据库配置 ====================
CHROMA_PERSIST_DIR=./chroma_db    # ChromaDB持久化目录
COLLECTION_NAME=deepblue_docs     # 集合名称

# ==================== 后端配置 ====================
HOST=0.0.0.0                      # 监听地址
PORT=8000                         # 监听端口
CORS_ORIGINS=http://localhost:3000 # 允许的前端地址
LOG_LEVEL=INFO                    # 日志级别

# ==================== 速率限制 ====================
RATE_LIMIT=10/minute              # 每IP每分钟请求限制
```

### Docker Compose 配置

如需自定义 Docker 部署，编辑 `docker-compose.yml`：

```yaml
services:
  backend:
    ports:
      - "8000:8000"  # 修改端口映射
    environment:
      - LOG_LEVEL=DEBUG  # 修改日志级别
    volumes:
      - ./chroma_db:/app/chroma_db  # 数据持久化

  frontend:
    ports:
      - "3000:80"  # 修改前端端口
    environment:
      - VITE_API_BASE_URL=http://localhost:8000  # API地址
```

---

## ❓ 常见问题

### Q1: 端口被占用

**问题**: `Bind for 0.0.0.0:8000 failed: port is already allocated`

**解决方案**:

```bash
# 查看占用端口的进程
lsof -i :8000

# 停止冲突的容器
docker-compose down

# 或修改 docker-compose.yml 中的端口映射
ports:
  - "8001:8000"  # 改为8001
```

### Q2: API 密钥配置错误

**问题**: `Error: API key is missing or invalid`

**解决方案**:

1. 检查 `.env` 文件中的 API 密钥是否正确
2. 重启服务使配置生效：

```bash
docker-compose down
docker-compose up -d
```

### Q3: ChromaDB 数据库初始化失败

**问题**: `Could not initialize ChromaDB`

**解决方案**:

```bash
# 清空数据库重新初始化
rm -rf chroma_db/
docker-compose restart backend
```

### Q4: 前端无法连接后端

**问题**: `Network Error` 或 `CORS Error`

**解决方案**:

1. 检查后端是否正常运行：
   ```bash
   curl http://localhost:8000/api/health
   ```

2. 检查 CORS 配置（`.env` 文件）：
   ```env
   CORS_ORIGINS=http://localhost:3000
   ```

3. 重启服务：
   ```bash
   docker-compose restart
   ```

### Q5: 内存不足

**问题**: `Out of memory error`

**解决方案**:

1. 减小 Embedding 模型（`.env` 文件）：
   ```env
   EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
   ```

2. 限制 Docker 内存使用（`docker-compose.yml`）：
   ```yaml
   services:
     backend:
       deploy:
         resources:
           limits:
             memory: 2G
   ```

3. 升级服务器配置

---

## 🔧 维护与监控

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 仅查看后端日志
docker-compose logs -f backend

# 仅查看前端日志
docker-compose logs -f frontend

# 查看最近100行日志
docker-compose logs --tail=100
```

### 重启服务

```bash
# 重启所有服务
docker-compose restart

# 仅重启后端
docker-compose restart backend

# 完全重新部署
docker-compose down
docker-compose up -d --build
```

### 备份数据

```bash
# 备份向量数据库
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz chroma_db/

# 备份上传的文档
tar -czf documents_backup_$(date +%Y%m%d).tar.gz data/documents/

# 备份日志
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### 更新系统

```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose up -d
```

### 监控资源使用

```bash
# 查看容器资源使用
docker stats

# 查看磁盘使用
du -sh chroma_db/ data/ logs/

# 清理无用镜像
docker system prune -a
```

---

## 📊 性能优化

### 1. 提升检索速度

```env
# 减少检索结果数量
TOP_K_RESULTS=3

# 禁用重排序
ENABLE_RERANK=false

# 提高相似度阈值
SIMILARITY_THRESHOLD=0.5
```

### 2. 提升生成速度

```env
# 使用更快的模型
LLM_MODEL=glm-4-flash

# 减少生成长度
LLM_MAX_TOKENS=500
```

### 3. 减少内存占用

```env
# 使用更小的Embedding模型
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# 减少切片大小
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

---

## 🔒 安全建议

### 1. 保护 API 密钥

- ⚠️ **永远不要提交 `.env` 文件到 Git**
- ✅ 使用 `.env.example` 作为模板
- ✅ 定期轮换 API 密钥

### 2. 配置防火墙

```bash
# 仅开放必要端口
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp  # 不直接暴露后端端口
ufw deny 3000/tcp  # 不直接暴露前端端口
```

### 3. 使用 HTTPS

配置 SSL 证书（推荐 Let's Encrypt）：

```bash
# 安装 Certbot
sudo apt-get install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d your-domain.com
```

### 4. 限制请求速率

已内置速率限制（10次/分钟/IP），如需调整：

```env
RATE_LIMIT=20/minute  # 修改为20次/分钟
```

---

## 📞 获取帮助

- **问题反馈**: 提交 GitHub Issue
- **功能建议**: 提交 Pull Request
- **部署支持**: 查看 [DEPLOYMENT_COMPARISON.md](./DEPLOYMENT_COMPARISON.md)
- **API 文档**: http://localhost:8000/docs

---

## 🎓 微课演示建议

### 场景 1: 课堂现场演示

- **部署方式**: 教师笔记本 + Docker 本地部署
- **优点**: 完全离线可用，无需网络
- **启动命令**: `./deploy.sh` 或 `deploy.bat`

### 场景 2: 实验室共享

- **部署方式**: 实验室服务器 + 局域网访问
- **优点**: 多人同时使用，数据集中管理
- **访问方式**: http://192.168.x.x:3000

### 场景 3: 在线演示

- **部署方式**: 阿里云轻量服务器 + 域名
- **优点**: 随时随地访问，适合远程教学
- **费用**: ¥99/年（学生价更低）

---

**最后更新**: 2026-02-25
**维护者**: DeepBlue Team
