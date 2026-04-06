import os
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility, DataType
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv(Path(__file__).parent / ".env")
connections.connect("default", host="127.0.0.1", port="19530")
name = "agent_rag"
if utility.has_collection(name):
    c = Collection(name)
    for f in c.schema.fields:
        if f.dtype == DataType.FLOAT_VECTOR:
            print("vector", f.name, "dim", f.params.get("dim"))
    emb = DashScopeEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v1"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    )
    v = emb.embed_query("test")
    print("query embedding len", len(v))
else:
    print("no collection")
