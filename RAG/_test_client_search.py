import os
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv(Path(__file__).parent / ".env")
uri = f"http://{os.getenv('MILVUS_HOST', '127.0.0.1')}:{os.getenv('MILVUS_PORT', '19530')}"
emb = DashScopeEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v1"),
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
)
v = emb.embed_query("Milvus")
client = MilvusClient(uri=uri)
r = client.search(
    "agent_rag",
    data=[v],
    anns_field="embedding",
    search_params={"metric_type": "L2", "params": {}},
    limit=3,
    output_fields=["name", "text"],
)
print(r)
if r and r[0]:
    print("keys", r[0][0].keys())
