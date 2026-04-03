from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
load_dotenv()

# 1. 初始化模型
llm = ChatTongyi(
    model="qwen-plus",
    temperature=0.7,
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
)

# 2. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_db = {
        "北京": "晴天，15-25度",
        "上海": "多云，18-28度",
        "深圳": "小雨，20-30度",
    }
    return weather_db.get(city, f"{city}的天气信息暂不可用")

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return "计算错误"

@tool
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    knowledge = {
        "LangChain": "LangChain是一个用于开发LLM应用的框架，支持工具、代理、内存管理等功能。",
        "机器学习": "机器学习是AI的子集，让系统能从数据中自动学习和改进。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于'{query}'的信息"

# 创建记忆存储
checkpointer = MemorySaver()

# 3. 创建带记忆的 Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, calculator, search_knowledge],
    system_prompt="你是一个专业助手，能记住之前的对话。",
    checkpointer=checkpointer,
)

# 4. 使用 Agent
def main() -> None:

    test_queries = [
        "北京今天天气怎么样？",
        "给我讲讲什么是机器学习",
        "计算 123 * 456",
    ]

    config_tools = {"configurable": {"thread_id": "session_tools_demo"}}

    # for query in test_queries:
    #     print(f"\n用户: {query}")

    #     result = agent.invoke(
    #         {"messages": [{"role": "user", "content": query}]},
    #         config_tools,
    #     )

    #     final_message = result["messages"][-1]  # 最后一条消息通常是模型的回答
    #     print(f"助手: {final_message.content}")
 

#   #####################################################################
    # 会话配置（同一个 thread_id 共享记忆）
    config = {"configurable": {"thread_id": "session_001"}}

    # # 第一轮对话
    # result1 = agent.invoke(
    #     {"messages": [{"role": "user", "content": "我叫小明"}]},
    #     config,
    # )
    # print(result1["messages"][-1].content)

    # # 第二轮对话（Agent 会记得用户叫小明）
    # result2 = agent.invoke(
    #     {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    #     config,
    # )
    # print(result2["messages"][-1].content)  # 会回答"小明"

#   #####################################################################
    # 流式调用
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气如何？"}]},
        config,
        stream_mode="values",
    ):
        messages = chunk.get("messages")
        if messages:
            last_message = messages[-1]
            # 如果 last_message 有 content 属性 和  content 的值为真。hasattr() 函数用于检查对象是否具有指定的属性。
            if hasattr(last_message, "content") and last_message.content:
                print(last_message.content, end="", flush=True)
    print()  # 换行


if __name__ == "__main__":
    main()