"""
向量数据库模块
只负责连接和基础 CRUD 操作（符合单一职责原则）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import chromadb
from chromadb.config import Settings
from config import CHROMA_DB_PATH, COLLECTION_NAME, SIMILARITY_METRIC
from typing import List, Dict, Optional


class VectorDB:
    """
    向量数据库封装类
    只提供基础的连接和 CRUD 操作，不包含业务逻辑
    """

    def __init__(self, db_path=None, collection_name=None):
        """
        初始化向量数据库连接

        参数:
            db_path: str - 数据库路径，默认从 config 读取
            collection_name: str - 集合名称，默认从 config 读取
        """
        self.db_path = db_path or CHROMA_DB_PATH
        self.collection_name = collection_name or COLLECTION_NAME

        # 连接数据库
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        self.collection = None

    def create_collection(self, reset=False):
        """
        创建集合

        参数:
            reset: bool - 是否重置（删除已存在的集合）
        """
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"✓ 已删除旧集合: {self.collection_name}")
            except Exception as e:
                # 集合不存在时会抛出异常，这是正常的
                print(f"删除集合跳过（可能不存在）: {e}")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": SIMILARITY_METRIC}
        )
        print(f"✓ 集合已创建/获取: {self.collection_name}")
        return self.collection

    def get_collection(self):
        """获取集合"""
        if self.collection is None:
            self.collection = self.client.get_collection(name=self.collection_name)
        return self.collection

    def add(self, ids: List[str], embeddings: List[List[float]],
            documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        添加文档到数据库

        参数:
            ids: List[str] - 文档ID列表
            embeddings: List[List[float]] - 向量列表
            documents: List[str] - 原始文本列表
            metadatas: List[Dict] - 元数据列表（可选）
        """
        if self.collection is None:
            self.get_collection()

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_embeddings: List[List[float]], n_results=3,
              where: Optional[Dict] = None, where_document: Optional[Dict] = None):
        """
        查询相似文档

        参数:
            query_embeddings: List[List[float]] - 查询向量
            n_results: int - 返回结果数量
            where: Dict - 元数据过滤条件（可选）
            where_document: Dict - 文档内容过滤条件（可选）

        返回:
            dict - 查询结果（包含 ids, documents, metadatas, distances）
        """
        if self.collection is None:
            self.get_collection()

        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances']  # 显式包含距离信息
        )

    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None,
            limit: Optional[int] = None):
        """
        获取文档

        参数:
            ids: List[str] - 文档ID列表（可选）
            where: Dict - 元数据过滤条件（可选）
            limit: int - 限制返回数量（可选）

        返回:
            dict - 文档数据
        """
        if self.collection is None:
            self.get_collection()

        return self.collection.get(ids=ids, where=where, limit=limit)

    def update(self, ids: List[str], embeddings: Optional[List[List[float]]] = None,
               documents: Optional[List[str]] = None, metadatas: Optional[List[Dict]] = None):
        """
        更新文档

        参数:
            ids: List[str] - 文档ID列表
            embeddings: List[List[float]] - 新向量（可选）
            documents: List[str] - 新文本（可选）
            metadatas: List[Dict] - 新元数据（可选）
        """
        if self.collection is None:
            self.get_collection()

        self.collection.update(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None):
        """
        删除文档

        参数:
            ids: List[str] - 文档ID列表（可选）
            where: Dict - 元数据过滤条件（可选）
        """
        if self.collection is None:
            self.get_collection()

        self.collection.delete(ids=ids, where=where)

    def count(self):
        """获取文档数量"""
        if self.collection is None:
            self.get_collection()
        return self.collection.count()

    def peek(self, limit=10):
        """查看部分数据（用于调试）"""
        if self.collection is None:
            self.get_collection()
        return self.collection.peek(limit=limit)

    def __repr__(self):
        count = self.count() if self.collection else "未连接"
        return f"VectorDB(collection={self.collection_name}, documents={count})"


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("VectorDB 模块测试")
    print("=" * 70)

    # 初始化
    db = VectorDB()
    print(f"\n初始化: {db}\n")

    # 创建集合（测试用，reset=True）
    collection = db.create_collection(reset=True)

    # 添加测试数据
    print("\n添加测试数据...")
    test_ids = ["doc1", "doc2", "doc3"]
    test_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]  # 模拟向量
    test_documents = [
        "员工可以在每周五带宠物来办公室",
        "远程办公政策允许每周3天在家工作",
        "公司提供全面的健康保险"
    ]
    test_metadatas = [
        {"category": "policies", "file": "pet.md"},
        {"category": "policies", "file": "remote.md"},
        {"category": "benefits", "file": "health.md"}
    ]

    db.add(
        ids=test_ids,
        embeddings=test_embeddings,
        documents=test_documents,
        metadatas=test_metadatas
    )
    print(f"✓ 已添加 {len(test_ids)} 条文档")

    # 查询
    print(f"\n当前文档数: {db.count()}")

    # 查看数据
    print("\n查看数据 (peek):")
    data = db.peek(limit=3)
    for i, (doc_id, doc, meta) in enumerate(zip(data['ids'], data['documents'], data['metadatas'])):
        print(f"  {i+1}. ID={doc_id}, category={meta['category']}, doc={doc[:30]}...")

    # 测试查询
    print("\n测试查询（模拟向量）:")
    query_result = db.query(
        query_embeddings=[[0.15] * 384],  # 模拟查询向量
        n_results=2
    )
    print(f"  找到 {len(query_result['documents'][0])} 个结果")

    # 测试元数据过滤
    print("\n测试元数据过滤（只查询 policies 类别）:")
    filtered = db.get(where={"category": "policies"})
    print(f"  找到 {len(filtered['ids'])} 条文档")
    for doc_id, meta in zip(filtered['ids'], filtered['metadatas']):
        print(f"    - {doc_id}: {meta['file']}")

    print("\n✓ 测试完成")
