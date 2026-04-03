from dotenv import load_dotenv
import os

load_dotenv()

# 通义千问示例
from langchain_community.chat_models import ChatTongyi

llm = ChatTongyi(
    model="qwen-turbo",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

response = llm.invoke("你好，请用一句话介绍自己")
print(response.content)