@vector_bp.route('/upload_document', methods=['POST'])
def upload_document():
    """POST /api/vector/upload_document"""
    
    data = request.get_json()
    file_path = data['file_path']
    collection_name = data['collection_name']
    
    # 调用核心处理逻辑
    success = vector_manager.process_file(file_path, collection_name)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'文档处理成功: {file_path}',
            'database_info': vector_manager.get_database_info(collection_name)
        })


@vector_bp.route('/query', methods=['POST'])
def query_documents():
    """POST /api/vector/query"""
    
    data = request.get_json()
    question = data['question']
    collection_name = data['collection_name']
    k = data.get('k', 5)
    
    # 执行 RAG 问答
    result = vector_retriever.answer_question(
        question, 
        k=k, 
        collection_name=collection_name
    )
    
    return jsonify({
        'success': True,
        'question': question,
        'answer': result.answer,
        'confidence': result.confidence,
        'sources': [...]  # 来源文档信息
    })