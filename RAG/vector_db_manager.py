import os
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from pymilvus import utility


class VectorDatabaseManager:
    def __init__(self, milvus_host, milvus_port, ...):
        # 1. 配置参数
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        self.milvus_port = str(milvus_port or os.getenv("MILVUS_PORT", "19530"))
        
        # 2. 初始化嵌入模型
        self._init_embeddings()
        
        # 3. 初始化文本切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 4. 连接到 Milvus
        self._connect_to_milvus()


    def process_file(self, file_path: str, collection_name: str = None) -> bool:
        """处理单个文件：加载 → 切分 → 存储"""
        
        # 步骤1: 加载文档
        documents = self.load_document(file_path)
        # 例如: PDF 文件会被解析为多个 Document 对象
        
        # 步骤2: 切分文档
        split_docs = self.split_documents(documents)
        # 将长文档切分为多个小块
        
        # 步骤3: 存入向量数据库
        self.add_documents_to_db(split_docs, collection_name)
        # 自动调用嵌入模型，生成向量并存储
        
        return True

def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
    """将文档添加到 Milvus"""
    
    target_collection = collection_name or self.collection_name # 目标集合名
    collection_exists = utility.has_collection(target_collection) # 检查集合是否存在
    
    if collection_exists:
        # 集合已存在：加载并追加数据
        self.vectorstore = Milvus(
            embedding_function=self.embeddings, # 嵌入函数
            collection_name=target_collection,  # 目标集合名
            connection_args={"host": self.milvus_host, "port": self.milvus_port} # 连接参数
        )
        self.vectorstore.add_documents(documents)  # 追加 
    else:
        # 集合不存在：创建新集合并插入
        self.vectorstore = Milvus.from_documents(
            documents=documents, 
            embedding=self.embeddings,
            collection_name=target_collection,
            connection_args={"host": self.milvus_host, "port": self.milvus_port}
        )