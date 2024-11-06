from langchain.chat_models import ChatOpenAI
import os

def Chat_OpenAI(model_id=os.getenv("OPENAI_MODEL", default="gpt-4o-mini")):
    # 初始化 OpenAI 聊天模型，從環境變數中讀取 API 金鑰
    return ChatOpenAI(model_name=model_id)
