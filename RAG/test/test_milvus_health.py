"""
Milvus 健康检查测试：
1) 连接 Milvus
2) 创建临时集合并建立向量索引
3) 插入测试向量并检索
4) 校验检索结果
5) 清理临时集合

运行方式：
- pytest: pytest RAG/test/test_milvus_health.py -q
- 直接运行: python RAG/test/test_milvus_health.py
"""

import time
from typing import List

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
ALIAS = "default"


def _build_test_schema(dim: int = 4) -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="label", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(fields=fields, description="Milvus health-check test schema")


def _insert_sample_data(collection: Collection) -> None:
    labels: List[str] = ["near_origin", "far_point"]
    vectors: List[List[float]] = [
        [0.0, 0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0, 10.0],
    ]
    collection.insert([labels, vectors])
    collection.flush()


def test_milvus_basic_functionality() -> None:
    collection_name = f"milvus_health_test_{int(time.time())}"

    # 1) 连接
    connections.connect(alias=ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)

    try:
        # 2) 创建临时集合与索引
        assert not utility.has_collection(collection_name, using=ALIAS)
        collection = Collection(name=collection_name, schema=_build_test_schema(), using=ALIAS)
        collection.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}},
        )

        # 3) 插入和加载
        _insert_sample_data(collection)
        collection.load()

        # 4) 检索并断言
        query = [[0.1, 0.1, 0.1, 0.1]]
        search_result = collection.search(
            data=query,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {}},
            limit=1,
            output_fields=["label"],
        )

        assert len(search_result) == 1
        assert len(search_result[0]) == 1
        top_hit = search_result[0][0]
        assert top_hit.entity.get("label") == "near_origin"
    finally:
        # 5) 清理
        if utility.has_collection(collection_name, using=ALIAS):
            utility.drop_collection(collection_name, using=ALIAS)
        connections.disconnect(alias=ALIAS)


if __name__ == "__main__":
    # 便于不装 pytest 时快速自测
    test_milvus_basic_functionality()
    print("Milvus 健康检查通过：连接/写入/检索/清理均正常。")
