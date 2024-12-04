from langchain_aws import ChatBedrock
from dotenv import load_dotenv
load_dotenv()

# 初始化模型
llm = ChatBedrock(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    model_kwargs=dict(temperature=0.5),
    region_name="us-east-1"  # 根據您的模型所在區域設定
)

# 定義訊息
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]

# 調用模型
ai_msg = llm.invoke(messages)

# 輸出回應
print(ai_msg.content)
