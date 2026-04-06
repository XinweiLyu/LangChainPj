"""
向量数据库管理模块
基于LangChain和Milvus实现文档切分、向量化存储和检索功能
- 1.
MinIO (对象存储服务) : Milvus 使用 MinIO 来存储数据。它提供了一个网页管理界面。

- 登录页面 : http://localhost:9001
- 账号 (Access Key) : minioadmin
- 密码 (Secret Key) : minioadmin
- 2.
Milvus (向量数据库) : 在我们当前的配置中，Milvus 本身 没有 提供一个用于登录的网页界面。您可以通过代码（例如使用 pymilvus 库）连接到 Milvus 服务来进行操作。

- 连接地址 : localhost:19530
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load env vars from explicit path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# LangChain and Milvus imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from pymilvus import (
    utility,
    connections,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDatabaseManager:
    """向量数据库管理器 (Milvus后端)"""

    def __init__(self,
                 milvus_host: str = None,
                 milvus_port: int = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 dashscope_api_key: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        初始化向量数据库管理器

        Args:
            milvus_host: Milvus 服务主机
            milvus_port: Milvus 服务端口
            collection_name: Milvus 集合名称
            embedding_model: DashScope嵌入模型名称
            dashscope_api_key: DashScope API密钥
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
        """
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        # Ensure port is a string as pymilvus might expect it, or handle int gracefully
        self.milvus_port = str(milvus_port or os.getenv("MILVUS_PORT", "19530"))
        self.milvus_alias = "default"
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型
        self._init_embeddings()

        # 初始化文档切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        # 连接到Milvus
        self._connect_to_milvus()

        # 向量数据库实例
        self.vectorstore = None
        # 延迟加载：不要在 __init__ 中调用 _load_existing_db，避免启动时连接未就绪或报错
        # self._load_existing_db()

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 确保 API Key 存在
            if not self.dashscope_api_key:
                logger.warning("未提供 DashScope API Key，将尝试从环境变量获取")
                self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key
            )
            # 简单测试 embedding 是否工作
            try:
                self.embeddings.embed_query("test")
                logger.info(f"成功加载并验证 DashScope嵌入模型: {self.embedding_model}")
            except Exception as e:
                logger.error(f"DashScope 模型验证失败: {e}")
                raise e

        except Exception as e:
            logger.error(f"加载DashScope模型失败: {e}")
            logger.warning("使用备用HuggingFace模型")
            # 备用HuggingFace模型,加载本机模型 sentence-transformers/all-MiniLM-L6-v2，并在 CPU 上
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def _connect_to_milvus(self):
        """连接到Milvus服务"""
        try:
            logger.info(f"Connecting to Milvus: host={self.milvus_host}, port={self.milvus_port}")
            connections.connect(alias="default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"成功连接到Milvus: {self.milvus_host}:{self.milvus_port}, alias=default")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _ensure_connection(self):
        """确保 default 连接可用；若断开则自动重连。"""
        try:
            # 先检查 alias 是否存在，再用一次轻量操作验证连接真的可用
            addr = connections.get_connection_addr("default")
            if not addr:
                raise RuntimeError("Milvus alias=default 连接地址为空")
            utility.list_collections(timeout=3, using="default")
        except Exception:
            self._connect_to_milvus()

    def _connection_args(self) -> Dict[str, str]:
        """供 LangChain Milvus 使用（当前 pymilvus 2.6 + langchain_milvus 在 __init__ 中有 ORM 别名 bug）。"""
        return {"host": self.milvus_host, "port": self.milvus_port, "alias": "default"}

    def _infer_orm_search_schema(self, col: Collection) -> Tuple[str, str, str]:
        """
        从集合 schema 推断向量字段、文本字段与 metric_type（用于 ORM search）。
        """
        vector_field: Optional[str] = None
        varchar_specs: List[Tuple[str, int]] = []

        for field in col.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                vector_field = field.name
            elif field.dtype in (DataType.VARCHAR, DataType.JSON):
                varchar_specs.append(
                    (field.name, int(field.params.get("max_length", 0) or 0))
                )

        if not vector_field:
            raise ValueError("集合中未找到 FLOAT_VECTOR 字段，无法检索")

        text_field: Optional[str] = None
        for preferred in ("text", "langchain_text", "page_content"):
            if any(n == preferred for n, _ in varchar_specs):
                text_field = preferred
                break
        if text_field is None and varchar_specs:
            varchar_specs.sort(key=lambda x: -x[1])
            text_field = varchar_specs[0][0]

        if text_field is None:
            raise ValueError("集合中未找到可用的文本字段（VARCHAR）")

        metric_type = "L2"
        for idx in col.indexes:
            if idx.field_name == vector_field and idx.params:
                metric_type = idx.params.get("metric_type") or metric_type
                break

        return vector_field, text_field, metric_type

    def _create_collection_orm(self, collection_name: str) -> None:
        """使用 ORM 创建与 upload_document / 现有项目一致的集合（避免 langchain_milvus 连接 bug）。"""
        dim = len(self.embeddings.embed_query("ping"))
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description="文本嵌入集合")
        col = Collection(name=collection_name, schema=schema, using="default")
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "AUTOINDEX", "metric_type": "L2", "params": {}},
        )
        logger.info(f"已创建集合 {collection_name}，向量维度={dim}")

    def _insert_documents_via_orm(
        self, documents: List[Document], collection_name: str
    ) -> None:
        """使用 default 连接与 Collection.insert 写入，不经过 langchain_milvus.Milvus。"""
        self._ensure_connection()
        col = Collection(collection_name, using="default")
        vector_field, text_field, _ = self._infer_orm_search_schema(col)

        dim = int(
            next(
                f.params["dim"]
                for f in col.schema.fields
                if f.dtype == DataType.FLOAT_VECTOR
            )
        )
        texts_raw = [(doc.page_content or "") for doc in documents]
        vectors = self.embeddings.embed_documents(texts_raw)
        if vectors and len(vectors[0]) != dim:
            raise ValueError(
                f"嵌入维度 {len(vectors[0])} 与集合向量字段维度 {dim} 不一致"
            )

        data_columns: List[List[Any]] = []
        for f in col.schema.fields:
            if f.is_primary and getattr(f, "auto_id", False):
                continue
            max_len = int(f.params.get("max_length", 65535))
            if f.dtype == DataType.FLOAT_VECTOR:
                data_columns.append(vectors)
            elif f.dtype == DataType.VARCHAR:
                if f.name == text_field:
                    data_columns.append(
                        [t if len(t) <= max_len else t[:max_len] for t in texts_raw]
                    )
                elif f.name == "name":
                    names = []
                    for doc in documents:
                        src = doc.metadata.get("source") or doc.metadata.get("name") or "document"
                        names.append(os.path.basename(str(src))[:max_len])
                    data_columns.append(names)
                else:
                    data_columns.append([""] * len(documents))
            elif f.dtype == DataType.JSON:
                data_columns.append(
                    [json.dumps(doc.metadata, ensure_ascii=False) for doc in documents]
                )
            else:
                raise ValueError(f"入库不支持字段: {f.name} ({f.dtype})")

        col.insert(data_columns)
        col.flush()
        logger.info(f"已向集合 {collection_name} ORM 插入 {len(documents)} 条，已 flush")

    def _entity_row_to_dict(self, entity: Any, field_names: List[str]) -> Dict[str, Any]:
        """将 search 返回的 entity 转为 dict。"""
        row: Dict[str, Any] = {}
        if entity is None:
            return row
        if hasattr(entity, "to_dict"):
            d = entity.to_dict()
            if isinstance(d, dict):
                # pymilvus 2.6：Hit.to_dict() 常为
                # {id, distance, entity: {name, text, ...}}，标量字段在嵌套 entity 里
                inner = d.get("entity")
                src: Dict[str, Any] = inner if isinstance(inner, dict) else d
                return {k: src.get(k) for k in field_names}
        for name in field_names:
            try:
                if hasattr(entity, "get"):
                    row[name] = entity.get(name)
                else:
                    row[name] = getattr(entity, name, None)
            except Exception:
                row[name] = None
        return row

    def _search_via_orm(
        self, query: str, k: int, collection_name: str
    ) -> List[Tuple[Document, float]]:
        """
        使用 pymilvus ORM（default 连接）检索，规避 langchain_milvus 与 MilvusClient 的别名不一致问题。
        """
        self._ensure_connection()
        col = Collection(collection_name, using="default")

        vector_field, text_field, metric_type = self._infer_orm_search_schema(col)
        output_fields: List[str] = []
        for f in col.schema.fields:
            if f.name == vector_field:
                continue
            output_fields.append(f.name)

        col.load()
        query_vector = self.embeddings.embed_query(query)
        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}

        raw = col.search(
            data=[query_vector],
            anns_field=vector_field,
            param=search_params,
            limit=k,
            output_fields=output_fields,
        )

        results: List[Tuple[Document, float]] = []
        for hits in raw:
            for hit in hits:
                row = self._entity_row_to_dict(hit.entity, output_fields)
                page = row.get(text_field)
                page_content = "" if page is None else str(page)
                metadata = {k: v for k, v in row.items() if k != text_field}
                results.append((Document(page_content=page_content, metadata=metadata), float(hit.distance)))
        return results

    def _has_collection_with_retry(self, collection_name: str) -> bool:
        """检查集合是否存在；若连接短暂失效则重连后重试一次。"""
        try:
            self._ensure_connection()
            return utility.has_collection(collection_name, using="default")
        except Exception as first_error:
            logger.warning(f"检查集合存在性失败，准备重试: {first_error}")
            try:
                self._connect_to_milvus()
                return utility.has_collection(collection_name, using="default")
            except Exception as retry_error:
                logger.error(f"重试检查集合失败: {retry_error}")
                return False

    def _load_existing_db(self):
        """加载已存在的Milvus集合"""
        try:
            self._ensure_connection()
            # 检查集合是否存在
            if self._has_collection_with_retry(self.collection_name):
                try:
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=self.collection_name,
                        connection_args=self._connection_args()
                    )
                    logger.info(f"成功加载现有Milvus集合: {self.collection_name}")
                except Exception as e:
                    logger.error(f"加载现有集合失败: {e}")
                    # 不再自动删除，防止误删数据。
                    # 如果 Schema 不兼容，将在写入时处理。
                    self.vectorstore = None
            else:
                logger.info(f"未找到集合 {self.collection_name}，将在添加文档时创建")
        except Exception as e:
            logger.error(f"加载Milvus集合失败: {e}")

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        # ... (代码与之前版本相同)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                logger.warning(f"未识别的文件类型 {file_extension}, 使用文本加载器")
            # 加载文档，返回Document列表，每个Document包含page_content: str和metadata: dict
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents

        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        # ... (代码与之前版本相同)
        try:
            # 调用text_splitter，把长文档切成几百个小片段
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            return documents

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
        """
        将文档添加到Milvus数据库

        Args:
            documents: 文档列表
            collection_name: 集合名称（可选，覆盖默认）
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return
        # 目标集合名
        target_collection = collection_name or self.collection_name

        try:
            self._ensure_connection()
            collection_exists = self._has_collection_with_retry(target_collection)
            if not collection_exists:
                logger.info(f"集合不存在，将创建: {target_collection}")
                self._create_collection_orm(target_collection)
            else:
                logger.info(f"加载现有集合: {target_collection}")

            self._insert_documents_via_orm(documents, target_collection)
            self.collection_name = target_collection
            self.vectorstore = None
            logger.info(f"成功向集合 '{target_collection}' 写入 {len(documents)} 条文档")

        except Exception as e:
            msg = str(e)
            if "non-exist field" in msg or "inconsistent with defined schema" in msg:
                logger.warning(f"检测到 Schema 不兼容 ({e})，尝试重建集合并 ORM 写入...")
                try:
                    if self._has_collection_with_retry(target_collection):
                        utility.drop_collection(target_collection, using="default")
                    self._create_collection_orm(target_collection)
                    self._insert_documents_via_orm(documents, target_collection)
                    self.collection_name = target_collection
                    self.vectorstore = None
                    logger.info(f"集合 '{target_collection}' 已重建并写入数据")
                except Exception as re:
                    logger.error(f"重建集合失败: {re}")
                    raise re
            else:
                logger.error(f"添加文档到Milvus失败: {e}")
                raise e

    def process_file(self, file_path: str, collection_name: str = None) -> bool:
        """
        处理单个文件：加载、切分、存储

        Args:
            file_path: 文件路径
            collection_name: 集合名称

        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            # 加载文档
            documents = self.load_document(file_path)
            if not documents:
                return False

            # 切分文档
            split_docs = self.split_documents(documents)
            # 将文档添加到Milvus数据库
            self.add_documents_to_db(split_docs, collection_name)

            logger.info(f"文件处理完成: {file_path}")
            return True

        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def process_csv_data(self, csv_path: str,
                         text_columns: List[str] = None,
                         metadata_columns: List[str] = None) -> bool:
        """
        处理CSV数据文件

        Args:
            csv_path: CSV文件路径
            text_columns: 需要向量化的文本列名列表
            metadata_columns: 需要向量化的元数据列名列表 ，知道「这条向量对应哪条业务记录」

        Returns:
            处理是否成功
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
            # 如果text_columns为空，则从df的列名中找到所有object类型的列，并去掉以Unnamed开头的列
            if text_columns is None or not text_columns:
                object_cols = [c for c in df.columns if df[c].dtype == 'object']
                text_columns = [c for c in object_cols if not str(c).startswith('Unnamed')]
            # 如果metadata_columns为空，则从df的列名中找到所有不在text_columns中的列
            if metadata_columns is None:
                metadata_columns = [c for c in df.columns if c not in text_columns]
           
            documents = [] 
            # 遍历df的每一行 ，构建Document对象
            for idx, row in df.iterrows():
                content_parts = [] # 内容部分. 列名: 单元格内容
                metadata = {"source": csv_path, "row_index": idx} # 元数据
                
                # 循环获取每一行的单元格内容
                for col in text_columns:  # col 是表头
                    if pd.notna(row[col]): 
                        text = str(row[col]).strip()   # row[col] 是单元格内容
                        if text:
                            content_parts.append(f"{col}: {text}") # 表头: 单元格内容 eg row 1 => id: 1， category: 1， question: 1， answer: 1
                
                # metadata 
                for col in metadata_columns:
                    val = row.get(col) # row.get(col) 是单元格内容
                    if pd.notna(val):
                        metadata[str(col)] = str(val) 

                if content_parts: 
                    content = "\n".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata) # 创建一个Document对象  page_content: str和metadata: dict
                    documents.append(doc)

            logger.info(f"构建了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents) # 切分文档
            self.add_documents_to_db(split_docs) # 将文档添加到Milvus数据库

            return True

        except Exception as e:
            logger.error(f"处理CSV数据失败: {e}")
            return False

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """使用嵌入模型为一组文本生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []

    def search(self, query: str, k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = None) -> List[
        Tuple[Document, float]]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: Milvus目前不支持直接的元数据过滤，此参数保留但暂不使用
            collection_name: 指定要搜索的集合名称（可选）

        Returns:
            (文档, 相似度分数) 列表
        """
        target_collection = collection_name or self.collection_name
        self._ensure_connection()
        if filter_dict:
            logger.warning("Milvus集成当前不支持直接的元数据过滤，该过滤器将被忽略。")

        if not target_collection or not self._has_collection_with_retry(target_collection):
            logger.warning("向量数据库未初始化或集合不存在")
            return []

        try:
            # 使用 ORM default 连接检索。langchain_milvus.Milvus 在 pymilvus 2.6 下会因
            # MilvusClient._using（cm-xxx）与 ORM connections 不一致而在 __init__ 中报错。
            results = self._search_via_orm(query, k, target_collection)
            self.collection_name = target_collection
            logger.info(f"搜索查询: '{query}', 返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.exception(f"搜索失败: {e}")
            return []

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取数据库信息
        """
        target_collection = collection_name or self.collection_name

        info = {
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "collection_name": target_collection,
            "is_initialized": self.vectorstore is not None
        }

        try:
            self._ensure_connection()
            # 如果 vectorstore 未初始化，尝试临时连接检查
            if self._has_collection_with_retry(target_collection):
                # 使用 Collection 对象获取统计信息
                col = Collection(target_collection, using="default")
                # 刷新以确保获取最新数据量（刚插入的数据可能还在内存中）
                col.flush()
                info["document_count"] = col.num_entities
                # 集合存在即可视为已初始化（不强依赖 self.vectorstore 当前是否已绑定）
                info["is_initialized"] = True

                # 检索已改为 ORM 直连，不再依赖 langchain_milvus.Milvus 实例
            else:
                info["document_count"] = 0
        except Exception as e:
            logger.error(f"获取Milvus集合信息失败: {e}")
            info["error"] = str(e)

        return info

    def clear_database(self):
        """清空Milvus集合"""
        try:
            self._ensure_connection()
            if self._has_collection_with_retry(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.vectorstore = None
                logger.info(f"Milvus集合 '{self.collection_name}' 已被删除")
        except Exception as e:
            logger.error(f"清空Milvus集合失败: {e}")


def main():
    """测试函数"""
    # 确保Milvus服务正在运行
    try:
        test_alias = "default"
        connections.connect(alias=test_alias, host="127.0.0.1", port=19530)
        utility.get_collection_stats("agent_rag", using=test_alias)
        connections.disconnect(alias=test_alias)
    except Exception as e:
        logger.error("无法连接到Milvus服务，请确保您已通过 docker-compose up -d 启动了Milvus。")
        logger.error(f"错误: {e}")
        return

    # 创建向量数据库管理器
    db_manager = VectorDatabaseManager()

    # 清空现有数据
    print("清空现有数据库...")
    db_manager.clear_database()

    # 处理学科数据CSV文件
    csv_path = "../data.csv"
    if os.path.exists(csv_path):
        print("处理CSV数据...")
        success = db_manager.process_csv_data(csv_path)
        if success:
            print("CSV数据处理成功！")

            # 测试搜索
            print("\n测试搜索功能:")
            results = db_manager.search("示例查询", k=3)
            for i, (doc, score) in enumerate(results):
                print(f"\n结果 {i + 1} (相似度: {score:.4f}):")
                print(f"内容: {doc.page_content[:200]}...")
                print(f"元数据: {doc.metadata}")
        else:
            print("CSV数据处理失败！")
    else:
        print(f"未找到数据文件: {csv_path}")

    # 显示数据库信息
    print("\n数据库信息:")
    info = db_manager.get_database_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
