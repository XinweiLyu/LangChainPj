import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver

# 导入工具
from tools.weather import get_weather
from tools.calculator import calculator
from tools.translator import translate

# 加载环境变量
load_dotenv()
# 遇到奇怪问题时，用 LangSmith 看看 Agent 到底在想什么：
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")







# 系统提示词
SYSTEM_PROMPT = """
你是小呆，一个全能智能助手。

你可以帮用户：
• 查询天气：查询中国主要城市的天气
• 数学计算：进行各种数学运算
• 文本翻译：将中文翻译成其他语言

工作原则：
1. 先理解用户意图，选择合适的工具
2. 如果不确定，可以询问用户
3. 回答简洁明了，有帮助
"""

def create_assistant():
    """创建智能助手"""
    # 初始化模型
    llm = ChatTongyi(
        model="qwen-plus",
        temperature=0.7,
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
    )
    
    # 创建记忆
    checkpointer = MemorySaver()
    
    # 创建 Agent
    agent = create_agent(
        model=llm,
        tools=[get_weather, calculator, translate],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer
    )
    
    return agent

def main():
    """主函数"""
    print("=" * 50)
    print("      小呆 - 全能智能助手 v1.0")
    print("=" * 50)
    print("输入 'quit' 退出\n")
    
    agent = create_assistant()
    config = {"configurable": {"thread_id": "main"}}
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', '退出', 'exit']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                recursion_limit=10  # 最多10步的递归
            )
            
            response = result["messages"][-1].content
            print(f"小呆: {response}\n")
            
        except Exception as e:
            print(f"出错了: {str(e)}\n")

if __name__ == "__main__":
    main()