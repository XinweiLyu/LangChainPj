import os # os 模块用于操作操作系统
from datetime import datetime

from dotenv import load_dotenv # load_dotenv 模块用于加载环境变量
from langchain_community.chat_models import ChatTongyi # ChatTongyi 模块用于调用通义千问模型
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # ChatPromptTemplate 模块用于定义多轮对话模板，PromptTemplate 模块用于定义单轮对话模板

load_dotenv()


def build_llm() -> ChatTongyi:
    return ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )


def demo_prompt_template() -> None:
    #####################################################################
    # 定义模板，用 {} 占位
    template = PromptTemplate.from_template(
        "请用{language}语言介绍一下{topic}，不超过100字。"
    )

    # 填充变量
    prompt = template.format(language="中文", topic="人工智能")
    print(prompt)
    # 输出: 请用中文语言介绍一下人工智能，不超过100字。


def demo_partial_prompt_template() -> None:
    #####################################################################
    # partial_variables：固定/自动填充部分占位符，format 时只需传其余变量
    template = PromptTemplate(
        template="今天是{date}，请告诉我关于{topic}的最新消息。",
        input_variables=["topic"],
        partial_variables={
            "date": datetime.now().strftime("%Y年%m月%d日")  # 自动填充
        },
    )

    # 只需要传 topic
    prompt = template.format(topic="AI发展")
    print(prompt)


def demo_chat_prompt_template(llm: ChatTongyi) -> None:
    #####################################################################
    # 定义多轮对话模板
    # from_messages() 传入list表，每个元素是一个元组，包含角色和消息内容。
    # 方法用于定义多轮对话模板，每轮对话由一个角色和一个消息组成。
    # 角色可以是 "system"、"human" 或 "assistant"，分别表示系统消息、用户消息和助手消息。
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}，擅长用简洁易懂的方式解释复杂概念。"),
        ("human", "请解释一下：{concept}"),
    ])

    # 生成消息列表
    messages = chat_template.format_messages(
        role="物理学教授",
        concept="量子纠缠",
    )

    # 调用模型
    response = llm.invoke(messages)
    print(response.content)


def demo_few_shot_chat(llm: ChatTongyi) -> None:
    #####################################################################
    # 定义示例
    examples = [
        {"input": "开心", "output": "我今天非常开心！"},
        {"input": "难过", "output": "我感到有些难过..."},
    ]

    # 创建包含示例的模板
    example_block = "\n\n".join(
        f"情绪: {e['input']}\n表达: {e['output']}" for e in examples
    )

    few_shot_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个情绪表达助手。"),
        ("human", "示例：\n{example_block}"),
        ("human", "现在请表达这个情绪: {emotion}"),
    ])
    messages = few_shot_template.format_messages(
        example_block=example_block,
        emotion="兴奋",
    )
    response = llm.invoke(messages)
    print(response.content)


def main() -> None:
    llm = build_llm()
    demo_prompt_template()
    demo_partial_prompt_template()
    demo_chat_prompt_template(llm)
    demo_few_shot_chat(llm)


if __name__ == "__main__":
    main()
