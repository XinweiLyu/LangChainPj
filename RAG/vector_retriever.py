"""
向量检索器模块
基于向量数据库实现智能问答和内容检索
"""
from openai import OpenAI
import logging
import sys
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# LangChain imports
from langchain_core.documents import Document

# 本地模块
from vector_db_manager import VectorDatabaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果数据类"""
    content: str 
    score: float 
    metadata: Dict[str, Any]
    source: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source
        }


@dataclass
class AnswerResult:
    """问答结果数据类"""
    answer: str
    confidence: float
    question_type: str
    source_documents: List[Document]
    scores: List[float]


class VectorRetriever:
    """向量检索器"""

    def __init__(self,
                 db_manager: VectorDatabaseManager,
                 similarity_threshold: Optional[float] = None,  # None 表示不过滤
                 max_results: int = 10):
        """
        初始化向量检索器

        Args:
            db_manager: 向量数据库管理器
            similarity_threshold: 相似度阈值
            max_results: 最大返回结果数
        """
        self.db_manager = db_manager
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        # LangChain + Milvus 默认 score 为距离（L2/IP 等），L2 下越小越相似
        self.distance_metric = "L2"

    def search_similar_content(self,
                               query: str,
                               collection_name: str,
                               k: int = None,
                               filter_expression: str = None,
                               include_scores: bool = True) -> List[Tuple[Document, float]]:
        """
        搜索相似内容

        Args:
            query: 查询文本
            collection_name: Milvus 集合名称
            k: 返回结果数量
            filter_expression: Milvus 过滤表达式
            include_scores: 是否包含相似度分数

        Returns:
            检索结果列表 (文档, 分数)
        """
        if k is None:
            k = self.max_results

        try:
            # 执行向量搜索
            search_results = self.db_manager.search(query=query, k=k, collection_name=collection_name)

            # 阈值过滤（可选）
            results = []
            for doc, score in search_results:
                if self.similarity_threshold is None:
                    results.append((doc, score))
                elif self.distance_metric == "L2":
                    if score <= self.similarity_threshold:
                        results.append((doc, score))
                else:
                    if score >= self.similarity_threshold:
                        results.append((doc, score))

            logger.info(f"在集合 '{collection_name}' 中检索查询: '{query}', 返回 {len(results)} 个高质量结果")
            return results

        except Exception as e:
            logger.error(f"在集合 '{collection_name}' 中检索失败: {e}")
            return []

    def answer_question(self,
                        question: str,
                        collection_name: str,
                        k: int = 5) -> AnswerResult:
        """
        回答问题

        Args:
            question: 问题文本
            collection_name: Milvus 集合名称
            k: 上下文文档数量

        Returns:
            回答结果
        """
        try:
            # 1. 分类问题
            question_type = QuestionClassifier.classify_question(question) 

            # 2. 检索相关文档
            relevant_docs_with_scores = self.search_similar_content(
                query=question,
                collection_name=collection_name,
                k=k
            )

            # 3. 构建上下文 (即使为空也构建空上下文)
            context_parts = []
            source_documents = []
            scores = []

            if relevant_docs_with_scores:
                for i, (doc, score) in enumerate(relevant_docs_with_scores):
                    context_parts.append(f"参考资料{i + 1}: {doc.page_content}")
                    source_documents.append(doc)
                    scores.append(score)

            context = "\n\n".join(context_parts)

            # 4. 生成回答 (使用 LLM)
            answer = self._generate_answer_with_llm(question, context)

            # 5. 计算置信度
            confidence = self._calculate_confidence(scores)

            return AnswerResult(
                answer=answer,
                confidence=confidence,
                question_type=question_type,
                source_documents=source_documents,
                scores=scores
            )

        except Exception as e:
            logger.error(f"回答问题失败: {e}")
            return AnswerResult(
                answer=f"处理问题时出现错误: {str(e)}",
                confidence=0.0,
                question_type="错误",
                source_documents=[],
                scores=[]
            )

    def _generate_answer_with_llm(self, question: str, context: str) -> str:
        """使用 LLM 生成回答"""
        try:

            import os

            # 使用与 query_system.py 相同的配置
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            model_name = os.environ.get("LLM_MODEL", "qwen-plus")

            client = OpenAI(api_key=api_key, base_url=base_url)

            if context.strip():
                system_prompt = (
                    "你是知识库问答助手。【参考资料】由向量检索从知识库取出，请优先严格依据其中的文字作答，"
                    "可做归纳、分点、转述，不要编造资料中不存在的事实。\n"
                    "不要使用「知识库中未找到相关内容」或「以下基于通用知识」等前缀。\n"
                    "仅当参考资料与问题主题完全无关时（例如资料只讲 RAG，问题问诺贝尔奖），"
                    "先一句话说明资料未覆盖该问题，再视情况补充常识。\n"
                    "回答简洁、准确。"
                )
                temperature = 0.2
            else:
                system_prompt = (
                    "你是一个智能助手。当前没有可用的知识库参考资料。\n"
                    "请用你的通用知识回答，并在开头写："
                    "「知识库中未找到相关内容，以下是基于通用知识的回答：」。\n"
                    "回答简洁、准确、有条理。"
                )
                temperature = 0.7

            user_prompt = f"问题：{question}\n\n"
            if context.strip():
                user_prompt += f"【参考资料】：\n{context}"
            else:
                user_prompt += "【参考资料】：(无)"

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM 生成失败: {e}")
            return "抱歉，生成回答时出现错误，请稍后再试。"

    def _calculate_confidence(self, scores: List[float]) -> float:
        """
        计算回答置信度

        Args:
            scores: 相似度分数列表

        Returns:
            置信度分数 (0-1)
        """
        if not scores:
            return 0.0

        # 检索分数为 Milvus L2 距离：越小越相似，不能当「越大越好」的相似度用
        def _l2_to_unit_similarity(d: float) -> float:
            return 1.0 / (1.0 + float(d) / 5000.0)

        sims = [_l2_to_unit_similarity(s) for s in scores]
        best_sim = max(sims)
        avg_sim = sum(sims) / len(sims)
        count_weight = min(len(scores) / 5.0, 1.0)
        confidence = (best_sim * 0.6 + avg_sim * 0.4) * count_weight
        return min(max(confidence, 0.0), 1.0)

    def get_statistics(self, collection_name: str) -> Dict[str, Any]:
        """
        获取检索统计信息

        Args:
            collection_name: Milvus 集合名称

        Returns:
            统计信息字典
        """
        db_info = self.db_manager.get_database_info()

        stats = {
            "database_info": db_info,
            "similarity_threshold": self.similarity_threshold,
            "max_results": self.max_results,
            "retriever_status": "active" if db_info.get("is_initialized") else "inactive"
        }

        return stats


class QuestionClassifier:
    @classmethod
    def classify_question(cls, question: str) -> str:
        return "通用查询"


def main():
    """测试函数"""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

    # 初始化 Milvus 连接
    try:
        db_manager = VectorDatabaseManager(
            milvus_host="localhost",
            milvus_port="19530"
        )
        retriever = VectorRetriever(db_manager)
        collection_name = "agent_rag"
        print("向量系统初始化成功")
    except Exception as e:
        print(f"向量系统初始化失败: {e}")
        return

    # 准备测试数据
    info = db_manager.get_database_info(collection_name=collection_name)
    if not info.get("is_initialized"):
        print(f"集合 '{collection_name}' 不存在，正在创建并添加数据...")
        # 此处可以添加一个示例文件上传的逻辑
        # 例如: db_manager.process_file("path/to/your/data.csv", collection_name)
        print("请先手动上传数据以进行测试。")
    else:
        print(f"集合 '{collection_name}' 已就绪，文档数: {info.get('document_count', 'unknown')}")

    # 测试问题回答
    test_questions = [
        "吕心炜是谁？"

    ]

    print("\n--- 测试问答功能 ---")
    for question in test_questions:
        print(f"\n问题: {question}")

        result = retriever.answer_question(question, collection_name=collection_name)
        print(f"回答: {result.answer}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"问题类型: {result.question_type}")
        print(f"参考来源数: {len(result.source_documents)}")


if __name__ == "__main__":
    main()
