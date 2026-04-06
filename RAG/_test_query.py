from pymilvus import connections, Collection, utility

connections.connect("default", host="127.0.0.1", port="19530")
c = Collection("agent_rag", using="default")
c.load()
q = c.query(expr="id >= 0", output_fields=["name", "text"], limit=5)
print("query ok", len(q), q[:1])
