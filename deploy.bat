@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM =========================================
REM   DeepBlue RAG 系统一键部署脚本 (Windows)
REM =========================================

echo ========================================
echo    DeepBlue RAG 系统一键部署
echo ========================================
echo.

REM 步骤 1: 检查 Docker
echo [1/6] 检查 Docker 环境...

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ 未检测到 Docker，请先安装 Docker Desktop
    echo.
    echo 下载地址: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

where docker-compose >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ 未检测到 Docker Compose
    echo Docker Desktop 应该已包含 Docker Compose
    pause
    exit /b 1
)

echo ✓ Docker 环境检查通过
docker --version
docker-compose --version
echo.

REM 步骤 2: 检查环境变量
echo [2/6] 检查环境变量...

if not exist .env (
    echo ✗ 未找到 .env 文件
    if exist .env.example (
        copy .env.example .env >nul
        echo ℹ 已从 .env.example 创建 .env 文件
        echo ✗ 请编辑 .env 文件，填入 API 密钥后重新运行
        pause
        exit /b 1
    ) else (
        echo ✗ 请创建 .env 文件并配置 API 密钥
        pause
        exit /b 1
    )
)

echo ✓ 环境变量文件已存在
echo.

REM 步骤 3: 创建必要目录
echo [3/6] 创建数据目录...

if not exist chroma_db mkdir chroma_db
if not exist data\documents mkdir data\documents
if not exist logs mkdir logs

echo ✓ 数据目录创建完成
echo.

REM 步骤 4: 构建镜像
echo [4/6] 构建 Docker 镜像...
echo.

docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ✗ 镜像构建失败
    pause
    exit /b 1
)

echo ✓ 镜像构建完成
echo.

REM 步骤 5: 启动服务
echo [5/6] 启动服务...
echo.

docker-compose up -d
if %errorlevel% neq 0 (
    echo ✗ 服务启动失败
    pause
    exit /b 1
)

echo ✓ 服务启动成功
echo.

REM 步骤 6: 健康检查
echo [6/6] 等待服务就绪...

set RETRY_COUNT=0
set MAX_RETRIES=30

:wait_backend
if %RETRY_COUNT% geq %MAX_RETRIES% (
    echo ✗ 后端服务启动超时
    echo.
    echo 查看日志: docker-compose logs backend
    pause
    exit /b 1
)

curl -f http://localhost:8000/api/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ 后端服务已就绪
    goto check_frontend
)

set /a RETRY_COUNT+=1
echo | set /p=.
timeout /t 2 /nobreak >nul
goto wait_backend

:check_frontend
curl -f http://localhost:3000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ 前端服务已就绪
) else (
    echo ℹ 前端服务正在启动中...
)

echo.

REM =========================================
REM 部署成功
REM =========================================

echo ========================================
echo    🎉 部署成功！
echo ========================================
echo.
echo 📡 前端地址: http://localhost:3000
echo 📡 后端API:  http://localhost:8000
echo 📚 API文档:  http://localhost:8000/docs
echo.
echo 常用命令:
echo   查看日志:   docker-compose logs -f
echo   停止服务:   docker-compose down
echo   重启服务:   docker-compose restart
echo   查看状态:   docker-compose ps
echo.
echo 数据目录:
echo   向量数据库:  .\chroma_db\
echo   上传文档:    .\data\documents\
echo   系统日志:    .\logs\
echo.
echo 按任意键退出（服务将继续在后台运行）
pause >nul
