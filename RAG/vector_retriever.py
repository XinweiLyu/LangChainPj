def search_similar_content(self, query: str, collection_name: str, k: int = None):
    """搜索相似内容"""
    
    # 执行向量搜索（内部会自动将 query 转为向量）
    search_results = self.db_manager.search(
        query=query, 
        k=k,  # 返回的相似文档数量
        collection_name=collection_name
    )
    
    # 过滤低相似度结果
    results = []
    for doc, score in search_results:
        if score >= self.similarity_threshold:  # 默认 0.5
            results.append((doc, score))
    
    return results

def answer_question(self, question: str, collection_name: str, k: int = 5):
    """回答问题 - RAG 的核心逻辑"""
    
    # 步骤1: 检索相关文档
    relevant_docs = self.search_similar_content(
        query=question,
        collection_name=collection_name, # 目标集合名是list 吗？ 是
        k=k
    )
    
    # 步骤2: 构建上下文
    context_parts = []
    for i, (doc, score) in enumerate(relevant_docs):
        context_parts.append(f"参考资料{i+1}: {doc.page_content}")
    context = "\n\n".join(context_parts)
    
    # 步骤3: 调用 LLM 生成回答
    answer = self._generate_answer_with_llm(question, context)
    
    return AnswerResult(
        answer=answer,
        source_documents=source_documents,
        scores=scores
    )

    def _generate_answer_with_llm(self, question: str, context: str) -> str:
    """使用 LLM 生成回答"""
    
    client = OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    system_prompt = """你是一个智能助手。请基于提供的【参考资料】回答用户的问题。
    如果参考资料为空或与问题无关，请利用你的通用知识进行回答..."""
    
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"问题：{question}\n\n【参考资料】：\n{context}"}
        ]
    )
    
    return response.choices[0].message.content