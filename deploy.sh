#!/bin/bash

# =========================================
#   DeepBlue RAG 系统一键部署脚本
# =========================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# =========================================
# 主流程
# =========================================

print_header "DeepBlue RAG 系统一键部署"

# 步骤 1: 检查 Docker
echo ""
print_info "[1/6] 检查 Docker 环境..."

if ! command_exists docker; then
    print_error "未检测到 Docker，请先安装 Docker"
    echo ""
    echo "安装指南:"
    echo "  macOS: brew install docker"
    echo "  Ubuntu: sudo apt-get install docker.io"
    echo "  官网: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "未检测到 Docker Compose，请先安装"
    echo ""
    echo "安装指南:"
    echo "  macOS: brew install docker-compose"
    echo "  Ubuntu: sudo apt-get install docker-compose"
    exit 1
fi

print_success "Docker 环境检查通过"
docker --version
docker-compose --version

# 步骤 2: 检查环境变量
echo ""
print_info "[2/6] 检查环境变量..."

if [ ! -f .env ]; then
    print_error "未找到 .env 文件"
    if [ -f .env.example ]; then
        cp .env.example .env
        print_info "已从 .env.example 创建 .env 文件"
        print_error "请编辑 .env 文件，填入 API 密钥后重新运行"
        exit 1
    else
        print_error "请创建 .env 文件并配置 API 密钥"
        exit 1
    fi
fi

print_success "环境变量文件已存在"

# 步骤 3: 创建必要目录
echo ""
print_info "[3/6] 创建数据目录..."

mkdir -p chroma_db data/documents logs

print_success "数据目录创建完成"

# 步骤 4: 构建镜像
echo ""
print_info "[4/6] 构建 Docker 镜像..."
echo ""

docker-compose build --no-cache

print_success "镜像构建完成"

# 步骤 5: 启动服务
echo ""
print_info "[5/6] 启动服务..."
echo ""

docker-compose up -d

print_success "服务启动成功"

# 步骤 6: 健康检查
echo ""
print_info "[6/6] 等待服务就绪..."

# 等待后端启动
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/api/health >/dev/null 2>&1; then
        print_success "后端服务已就绪"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "后端服务启动超时"
    echo ""
    echo "查看日志: docker-compose logs backend"
    exit 1
fi

# 等待前端启动
if curl -f http://localhost:3000/health >/dev/null 2>&1; then
    print_success "前端服务已就绪"
else
    print_info "前端服务正在启动中..."
fi

# =========================================
# 部署成功
# =========================================

echo ""
print_header "🎉 部署成功！"
echo ""
echo -e "${GREEN}📡 前端地址:${NC} http://localhost:3000"
echo -e "${GREEN}📡 后端API:${NC}  http://localhost:8000"
echo -e "${GREEN}📚 API文档:${NC}  http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}常用命令:${NC}"
echo "  查看日志:   docker-compose logs -f"
echo "  停止服务:   docker-compose down"
echo "  重启服务:   docker-compose restart"
echo "  查看状态:   docker-compose ps"
echo ""
echo -e "${YELLOW}数据目录:${NC}"
echo "  向量数据库:  ./chroma_db/"
echo "  上传文档:    ./data/documents/"
echo "  系统日志:    ./logs/"
echo ""
print_info "按 Ctrl+C 停止脚本（服务将继续在后台运行）"
