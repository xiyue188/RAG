"""
脚本1: 初始化向量数据库
只调用 rag 模块，不包含逻辑
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from rag import VectorDB
from config import ensure_directories

def main():
    """初始化数据库"""
    print("=" * 70)
    print("初始化向量数据库")
    print("=" * 70)

    # 确保目录存在
    ensure_directories()

    # 创建数据库连接
    db = VectorDB()

    # 创建集合（是否重置）
    reset = input("\n是否重置数据库？这将删除所有现有数据 (y/N): ").strip().lower() == 'y'

    collection = db.create_collection(reset=reset)

    print(f"\n✓ 数据库初始化完成")
    print(f"  集合名称: {collection.name}")
    print(f"  当前文档数: {db.count()}")


if __name__ == "__main__":
    main()
