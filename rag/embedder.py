"""
向量化（Embedding）模块
负责将文本转换为向量表示
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME, BATCH_SIZE
from typing import List, Union
from .logger import get_logger

# 初始化logger
logger = get_logger(__name__)


class Embedder:
    """
    文本向量化器
    封装 SentenceTransformer，提供统一的向量化接口
    """

    def __init__(self, model_name=None):
        """
        初始化 Embedder

        参数:
            model_name: str - 模型名称，默认从 config 读取
        """
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        logger.info(f"加载 Embedding 模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding 模型加载完成")

    def encode(self, texts: Union[str, List[str]], to_list=True, batch_size=None):
        """
        将文本转换为向量

        参数:
            texts: str 或 List[str] - 单个文本或文本列表
            to_list: bool - 是否转换为 Python list（默认True，兼容 ChromaDB）
            batch_size: int - 批处理大小

        返回:
            ndarray 或 list - 向量或向量列表
        """
        if batch_size is None:
            batch_size = BATCH_SIZE

        # 确保输入是列表
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]

        # 向量化
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,  # 只在处理多个文本时显示进度条
            convert_to_numpy=True
        )

        # 转换为 list（ChromaDB 需要）
        if to_list:
            embeddings = embeddings.tolist()

        # 如果输入是单个文本，返回单个向量
        if is_single:
            return embeddings[0]

        return embeddings

    def get_embedding_dim(self):
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()

    def __repr__(self):
        return f"Embedder(model={self.model_name}, dim={self.get_embedding_dim()})"


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("Embedding 模块测试")
    print("=" * 70)

    # 初始化
    embedder = Embedder()
    print(f"\n{embedder}\n")

    # 测试单个文本
    text = "可以带宠物来公司吗？"
    print(f"测试文本: \"{text}\"")

    embedding = embedder.encode(text)
    print(f"向量维度: {len(embedding)}")
    print(f"向量前10维: {embedding[:10]}\n")

    # 测试多个文本
    texts = [
        "员工可以在每周五带宠物来办公室",
        "远程办公政策允许每周3天在家工作",
        "公司提供全面的健康保险"
    ]

    print(f"测试批量向量化 ({len(texts)} 个文本):")
    embeddings = embedder.encode(texts)

    print(f"  结果形状: {len(embeddings)} x {len(embeddings[0])}")

    # 计算相似度
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    emb_array = np.array(embeddings)
    sim_matrix = cosine_similarity(emb_array)

    print(f"\n相似度矩阵:")
    print("       文本1  文本2  文本3")
    for i in range(len(texts)):
        row = f"  文本{i+1}"
        for j in range(len(texts)):
            row += f"  {sim_matrix[i][j]:.3f}"
        print(row)

    print("\n✓ 测试完成")
