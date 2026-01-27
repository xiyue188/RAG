"""
文档摄入模块
负责文档处理和入库逻辑
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .chunker import chunk_text
from .embedder import Embedder
from .vectordb import VectorDB
from config import DATA_DIR, SUPPORTED_FILE_TYPES, ENCODING
from typing import Optional, Dict, List
from tqdm import tqdm


class DocumentIngestion:
    """
    文档摄入器
    负责文档的读取、分块、向量化和入库
    """

    def __init__(self, vectordb: Optional[VectorDB] = None,
                 embedder: Optional[Embedder] = None):
        """
        初始化文档摄入器

        参数:
            vectordb: VectorDB - 向量数据库实例
            embedder: Embedder - 向量化器实例
        """
        self.vectordb = vectordb or VectorDB()
        self.embedder = embedder or Embedder()

        # 确保集合已创建
        self.vectordb.get_collection()

    def ingest_text(self, text: str, doc_id: str,
                    metadata: Optional[Dict] = None,
                    chunk_size: int = None, chunk_overlap: int = None):
        """
        摄入单个文本

        参数:
            text: str - 文本内容
            doc_id: str - 文档ID前缀
            metadata: Dict - 元数据（可选）
            chunk_size: int - 块大小（可选）
            chunk_overlap: int - 重叠大小（可选）

        返回:
            int - 添加的块数量
        """
        # 1. 分块
        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)

        # 2. 向量化
        embeddings = self.embedder.encode(chunks, to_list=True)

        # 3. 准备数据
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {} for _ in range(len(chunks))]

        # 4. 入库
        self.vectordb.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return len(chunks)

    def ingest_file(self, file_path: str, category: Optional[str] = None) -> int:
        """
        摄入单个文件

        参数:
            file_path: str - 文件路径
            category: str - 类别（可选）

        返回:
            int - 添加的块数量
        """
        file_path = Path(file_path)

        # 检查文件类型
        if file_path.suffix not in SUPPORTED_FILE_TYPES:
            print(f"⚠ 跳过不支持的文件类型: {file_path}")
            return 0

        # 读取文件
        try:
            with open(file_path, 'r', encoding=ENCODING) as f:
                content = f.read()
        except Exception as e:
            print(f"✗ 读取文件失败 {file_path}: {e}")
            return 0

        # 准备元数据
        metadata = {
            "file": file_path.name,
            "category": category or "uncategorized"
        }

        # 摄入
        doc_id = file_path.stem  # 文件名（不含扩展名）
        chunk_count = self.ingest_text(content, doc_id, metadata)

        return chunk_count

    def ingest_directory(self, directory_path: str = None,
                         recursive: bool = True) -> Dict:
        """
        摄入整个目录的文档

        参数:
            directory_path: str - 目录路径（默认使用 DATA_DIR）
            recursive: bool - 是否递归处理子目录

        返回:
            Dict - 统计信息
        """
        if directory_path is None:
            directory_path = DATA_DIR

        directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        print(f"开始摄入文档: {directory_path}")
        print("=" * 70)

        stats = {
            "total_files": 0,
            "total_chunks": 0,
            "success_files": 0,
            "failed_files": 0,
            "categories": {}
        }

        # 遍历目录
        if recursive:
            # 递归处理：每个子目录作为一个类别
            for category_dir in directory_path.iterdir():
                if not category_dir.is_dir():
                    continue

                category = category_dir.name
                print(f"\n处理类别: {category}")

                file_list = list(category_dir.glob("*"))
                file_count = 0
                chunk_count = 0

                for file_path in file_list:
                    if file_path.is_file() and file_path.suffix in SUPPORTED_FILE_TYPES:
                        print(f"  {file_path.name}", end="")

                        try:
                            chunks = self.ingest_file(str(file_path), category)
                            print(f" ({chunks} chunks) ✓")

                            file_count += 1
                            chunk_count += chunks
                            stats["success_files"] += 1

                        except Exception as e:
                            print(f" ✗ 失败: {e}")
                            stats["failed_files"] += 1

                stats["total_files"] += file_count
                stats["total_chunks"] += chunk_count
                stats["categories"][category] = {
                    "files": file_count,
                    "chunks": chunk_count
                }

        else:
            # 非递归：只处理当前目录的文件
            file_list = list(directory_path.glob("*"))

            for file_path in tqdm(file_list, desc="Processing files"):
                if file_path.is_file() and file_path.suffix in SUPPORTED_FILE_TYPES:
                    try:
                        chunks = self.ingest_file(str(file_path))
                        stats["success_files"] += 1
                        stats["total_chunks"] += chunks
                    except Exception as e:
                        print(f"✗ 处理失败 {file_path.name}: {e}")
                        stats["failed_files"] += 1

            stats["total_files"] = stats["success_files"] + stats["failed_files"]

        # 打印统计
        print("\n" + "=" * 70)
        print("摄入完成！")
        print("=" * 70)
        print(f"  • 处理文件数: {stats['total_files']}")
        print(f"  • 成功: {stats['success_files']}, 失败: {stats['failed_files']}")
        print(f"  • 生成块数: {stats['total_chunks']}")

        if stats['categories']:
            print(f"\n按类别统计:")
            for cat, cat_stats in stats['categories'].items():
                print(f"  • {cat}: {cat_stats['files']} 文件, {cat_stats['chunks']} 块")

        print(f"\n数据库总文档数: {self.vectordb.count()}")

        return stats

    def __repr__(self):
        return f"DocumentIngestion(db={self.vectordb}, embedder={self.embedder})"


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("DocumentIngestion 模块测试")
    print("=" * 70)

    # 初始化
    ingestion = DocumentIngestion()
    print(f"\n{ingestion}\n")

    # 测试文本摄入
    print("测试文本摄入:")
    test_text = """
    TechCorp 宠物政策

    员工可以在每周五带宠物来办公室。
    宠物必须：
    - 性格温顺，不攻击人
    - 已接种所有必要疫苗
    - 在公共区域需要牵引

    CEO 的金毛寻回犬是公司吉祥物。
    """

    chunk_count = ingestion.ingest_text(
        text=test_text,
        doc_id="test_policy",
        metadata={"category": "policies", "file": "test.md"}
    )

    print(f"✓ 文本摄入完成，生成 {chunk_count} 个块")

    # 查看数据库状态
    print(f"\n数据库文档数: {ingestion.vectordb.count()}")

    print("\n✓ 测试完成")
