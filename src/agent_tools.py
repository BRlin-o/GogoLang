import os
import boto3
import streamlit as st
from langchain.agents import Tool
# from langchain_aws import BedrockChat
from core.src.models.llm_bedrock import Chat_Bedrock
from .rag import get_rag_chain

from dotenv import load_dotenv
load_dotenv()

CAR_MODEL = "CrossOver"

model_id = "anthropic.claude-3-haiku-20240307-v1:0"
kb_id = os.getenv("KB_ID")

claude_llm = Chat_Bedrock(
    model_id=model_id,
    model_kwargs={
        "temperature": 0
    },
)

def call_search(query):
    global CAR_MODEL
    CAR_MODEL = st.session_state["car_model"]
    super_query = f"{query}？"
    print(f"[DEBUG] kb_id={kb_id}, CAR_MODEL={CAR_MODEL}, query={super_query}")
    return get_rag_chain(kb_id, claude_llm, CAR_MODEL)(super_query)

LLM_AGENT_TOOLS = [
    Tool(
        name="GogoroSearch",
        func=lambda query: call_search(query),
        description=(
            "This tool must be used as the primary method for handling any questions related to Gogoro. "
            "Always use it first to obtain accurate and up-to-date information before making decisions. "
            "The input should be a properly formatted question in Traditional Chinese, such as: "
            "'What is the price of Gogoro batteries?' or 'Comparison of Gogoro models.' "
            "If the response includes any Markdown syntax (e.g., `![]()`), display it directly in the chat, "
            "as the frontend can render Markdown to show images or other content. "
            "Using this tool ensures accurate and reliable information, avoiding incorrect answers."
        ),
        # description=(
        #     "當面對任何與 Gogoro 相關的問題時，務必優先使用此工具以獲取正確答案，"
        #     "並基於獲取的資訊做出決策。"
        #     "輸入應為格式正確的繁體中文問題，例如："
        #     "「Gogoro 電池價格」、「Gogoro 車款比較」等。"
        #     "若回應中包含任何 Markdown 語法（如 `![]()`），"
        #     "請直接顯示在聊天中，因前端會自動渲染 Markdown 以顯示圖片或其他資料。"
        #     "使用此工具能確保提供的資訊最新且準確，避免錯誤解答。"
        # ),
        # description=(
        #     "Use when you are asked any questions about Gogoro."
        #     " The Input should be a correctly formatted question, and using 繁體中文."
        #     " If the response contains any markdown syntax like `![]()`, please make sure to display it directly in the chat, as the frontend can render markdown to show image data."
        # ),
    )
]