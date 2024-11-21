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
    return get_rag_chain(kb_id, claude_llm, CAR_MODEL)(super_query)

LLM_AGENT_TOOLS = [
    Tool(
        name="GogoroSearch",
        func=lambda query: call_search(query),
        description=(
            "Use when you are asked any questions about Gogoro."
            " The Input should be a correctly formatted question, and using 繁體中文."
            " If the response contains any markdown syntax like `![]()`, please make sure to display it directly in the chat, as the frontend can render markdown to show image data."
        ),
    )
]