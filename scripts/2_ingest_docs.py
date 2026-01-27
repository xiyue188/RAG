"""
脚本2: 摄入文档
只调用 rag 模块，不包含逻辑
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import DocumentIngestion, VectorDB, Embedder
from config import DATA_DIR, ensure_directories


def main():
    """批量摄入文档"""
    print("=" * 70)
    print("文档摄入系统")
    print("=" * 70)

    # 确保目录存在
    ensure_directories()

    # 检查数据目录
    data_dir = Path(DATA_DIR)
    if not data_dir.exists():
        print(f"\n✗ 数据目录不存在: {data_dir}")
        print(f"\n请将文档放入以下目录:")
        print(f"  {data_dir}")
        print(f"\n目录结构示例:")
        print(f"  data/documents/")
        print(f"  ├── policies/")
        print(f"  │   ├── pet_policy.md")
        print(f"  │   └── remote_work.md")
        print(f"  └── benefits/")
        print(f"      └── health_insurance.md")
        return

    # 初始化组件
    print(f"\n数据目录: {data_dir}\n")

    vectordb = VectorDB()
    embedder = Embedder()
    ingestion = DocumentIngestion(vectordb, embedder)

    # 摄入文档
    stats = ingestion.ingest_directory(
        directory_path=str(data_dir),
        recursive=True
    )

    print(f"\n✓ 摄入完成")
    print(f"  成功: {stats['success_files']} 文件")
    print(f"  失败: {stats['failed_files']} 文件")
    print(f"  总块数: {stats['total_chunks']}")


if __name__ == "__main__":
    main()
