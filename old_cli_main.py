"""
RAG 项目主程序入口
提供命令行界面，调用各个脚本
"""

import sys
import os
from pathlib import Path


# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

import argparse
from config import validate_config, ensure_directories
from dotenv import load_dotenv


def main():
    """主程序"""
    # 加载环境变量 - 指定 .env 文件的完整路径
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)

    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="RAG 系统 - 检索增强生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py init              # 初始化数据库
  python main.py ingest            # 摄入文档
  python main.py query             # 测试检索
  python main.py run               # 运行完整 RAG 系统
  python main.py config            # 检查配置
        """
    )

    parser.add_argument(
        'command',
        choices=['init', 'ingest', 'query', 'run', 'config'],
        help='要执行的命令'
    )

    args = parser.parse_args()

    # 显示欢迎信息
    print("\n" + "=" * 70)
    print(" RAG 系统 - Retrieval-Augmented Generation".center(70))
    print("=" * 70 + "\n")

    # 执行对应命令
    if args.command == 'init':
        print("执行: 初始化数据库\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location("init_db", Path(__file__).parent / "scripts" / "1_init_db.py")
        init_db = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(init_db)
        init_db.main()

    elif args.command == 'ingest':
        print("执行: 摄入文档\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location("ingest_docs", Path(__file__).parent / "scripts" / "2_ingest_docs.py")
        ingest_docs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ingest_docs)
        ingest_docs.main()

    elif args.command == 'query':
        print("执行: 测试检索\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_query", Path(__file__).parent / "scripts" / "3_test_query.py")
        test_query = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_query)
        test_query.main()

    elif args.command == 'run':
        print("执行: 运行 RAG 系统\n")
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_rag", Path(__file__).parent / "scripts" / "4_run_rag.py")
        run_rag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_rag)
        run_rag.main()

    elif args.command == 'config':
        print("执行: 检查配置\n")
        check_config()

    print()


def check_config():
    """检查配置"""
    from config import (
        PROJECT_ROOT, DATA_DIR, DB_DIR,
        EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP,
        LLM_PROVIDER, TOP_K_RESULTS, RETRIEVAL_MODE
    )

    print("=" * 70)
    print("配置信息")
    print("=" * 70)

    print(f"\n【路径配置】")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print(f"  数据目录: {DATA_DIR}")
    print(f"  数据库目录: {DB_DIR}")

    print(f"\n【Embedding 配置】")
    print(f"  模型: {EMBEDDING_MODEL_NAME}")

    print(f"\n【分块配置】")
    print(f"  块大小: {CHUNK_SIZE} 字符")
    print(f"  重叠: {CHUNK_OVERLAP} 字符")

    print(f"\n【检索配置】")
    print(f"  Top-K: {TOP_K_RESULTS}")
    print(f"  检索模式: {RETRIEVAL_MODE}")
    print(f"  高级检索: 已启用（Query Rewrite + Multi-Query）")

    print(f"\n【LLM 配置】")
    print(f"  提供商: {LLM_PROVIDER}")

    # 确保目录存在
    print(f"\n【目录状态】")
    ensure_directories()

    data_exists = DATA_DIR.exists()
    db_exists = DB_DIR.exists()

    print(f"  数据目录: {'[OK] 存在' if data_exists else '[FAIL] 不存在'}")
    print(f"  数据库目录: {'[OK] 存在' if db_exists else '[FAIL] 不存在'}")

    # 验证配置
    print(f"\n【配置验证】")
    try:
        validate_config()
        print("  [OK] 配置验证通过")
    except ValueError as e:
        print(f"  [FAIL] 配置验证失败:\n{e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
