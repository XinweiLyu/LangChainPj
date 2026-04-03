import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional, Type

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """
    获取城市的天气信息

    Args:
        city: 城市名称，如"北京"、"上海"
    """
    # 模拟天气数据（实际项目中调用天气 API）
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "深圳": "小雨，28°C",
    }
    return weather_data.get(city, f"{city}的天气暂时无法获取")


@tool
def search_database(query: str, category: Optional[str] = None) -> str:
    """
    在数据库中搜索信息

    Args:
        query: 搜索关键词
        category: 可选的分类过滤器，如"科技"、"健康"、"教育"
    """
    database = {
        "科技": ["人工智能正在改变世界", "5G技术的应用"],
        "健康": ["健康饮食的重要性", "运动与长寿的关系"],
        "教育": ["在线学习的趋势", "终身学习的价值"]
    }

    results = []
    
    if category and category in database:
        for item in database[category]: 
            if query.lower() in item.lower(): 
                results.append(f"[{category}] {item}")
    else: 
        # 遍历 database 中的所有值，找到 query 在 item 中的所有值
        for cat, items in database.items():
            for item in items:
                if query.lower() in item.lower(): 
                    results.append(f"[{cat}] {item}")

    if results:
        return "找到以下结果:\n" + "\n".join(results)
    else:
        return f"没有找到关于 '{query}' 的结果"


class CalculatorInput(BaseModel):
    expression: str = Field(description="数学表达式，如 '2+2' 或 '10*5'")


class Calculator(BaseTool):
    name: str = "calculator"
    description: str = "执行数学计算"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """同步执行"""
        try:
            result = eval(expression) # eval() 函数用于执行字符串中的 Python 表达式
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """异步执行（可选实现）"""
        return self._run(expression)


def main() -> None:
    # 测试工具
    # result = get_weather.invoke({"city": "北京"})
    # print(result)  # 输出: 晴天，25°C

    # # 查看工具信息
    # print(f"工具名称: {get_weather.name}")
    # print(f"工具描述: {get_weather.description}")
    # print(f"参数结构: {get_weather.args}")

    # result = search_database.invoke({"query": "人工智能", "category": "科技"})
    # print(result)

    # result2 = search_database.invoke({"query": "智能"})
    # print(result2)
    #####################################################################
    # calc_result = Calculator().invoke({"expression": "123 * 456"})
    # print(calc_result)  # 计算结果: 123 * 456 = 56088

    #####################################################################
    # 创建工具列表
    tools = [get_weather, search_database, Calculator()]

    llm = ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    # 绑定到模型
    llm_with_tools = llm.bind_tools(tools)

    # 现在模型知道有这些工具可用了
    response = llm_with_tools.invoke("北京今天天气如何？")

    # 检查模型是否决定调用工具
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"模型想调用: {tool_call['name']}")
            print(f"传入参数: {tool_call['args']}")




if __name__ == "__main__":
    main()
