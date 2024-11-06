from pprint import pprint
import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chains import ConversationChain
from typing import List, Tuple, Union, Dict
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
load_dotenv()

# from src.llms.llm_openai import Chat_OpenAI
from src.ui import StreamHandler, display_chat_messages, langchain_messages_format, render_chat_interface, get_user_md, get_assistant_md

# llm = Chat_OpenAI()
from langchain_openai import ChatOpenAI
import os

def Chat_OpenAI(model_id=os.getenv("OPENAI_MODEL", default="gpt-4o-mini")):
    # 初始化 OpenAI 聊天模型，從環境變數中讀取 API 金鑰
    return ChatOpenAI(
        model=model_id,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

llm = Chat_OpenAI()

memory = ConversationBufferWindowMemory(
    k=10,
    ai_prefix="Assistant",
    human_prefix="Hu",
    chat_memory=StreamlitChatMessageHistory(),
    return_messages=False,
    memory_key="chat_history",
    input_key="input"
)

GENERIC_PROMPT = """
You are a highly intelligent and helpful assistant. Your purpose is to assist users by answering their questions, providing insightful responses, and offering support across various topics. Respond concisely, clearly, and politely. 
Ensure that your responses are both informative, accessible, and always using {language}.
If a question is outside your scope, kindly inform the user and provide guidance on where they might find more information.

Available tools: {tool_names}
{tools}
Scratchpad: {agent_scratchpad}

chat_history: {chat_history}
input: {input}

Instructions:
1. Only use tools if absolutely necessary to answer the question.
2. If the input question can be answered with the information you already know, directly respond with the answer and end the conversation.
3. If you cannot answer with current information and need more data, then and only then use the available tools to gather that information.
4. Avoid repeating tool use if the previous output is already sufficient to answer.

Answer with "Final Answer:" before your response when you are ready to provide a direct answer without tool usage.
"""

GENERIC_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["chat_history", "input", "language"],
    template=GENERIC_PROMPT
)

agent = create_react_agent(
    llm=llm, 
    tools=[],  # 移除GogoroSearch工具
    prompt=GENERIC_PROMPT_TEMPLATE
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[],
    verbose=True,
    return_intermediate_steps=False,
    memory=memory,
    handle_parsing_errors=True
)

# GENERIC_PROMPT = """
# You are a highly intelligent and helpful assistant. Your purpose is to assist users by answering their questions, providing insightful responses, and offering support across various topics. Respond concisely, clearly, and politely. 
# Ensure that your responses are both informative, accessible, and always using {language}.
# If a question is outside your scope, kindly inform the user and provide guidance on where they might find more information.

# chat_history: {chat_history}
# user_input: {input}
# """

# GENERIC_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["chat_history", "input", "language"],
#     template=GENERIC_PROMPT
# )

# from langchain.chains import LLMChain

# # 使用LLMChain，傳遞自訂的變數
# agent_chain = LLMChain(
#     llm=llm,
#     prompt=GENERIC_PROMPT_TEMPLATE,
#     memory=memory,
#     verbose=True
# )

from langchain_core.runnables import RunnableLambda

get_output = RunnableLambda(lambda x: x["output"])

chain = agent_chain | get_output
# chain = agent_chain

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your assistant. How can I help you?"),
    ]

if __name__ == "__main__":
    st.set_page_config(
        page_title="Gogoro Chatbot",
        page_icon="🤖",
        # layout="wide"
    )

    car_model, chat_lang = render_chat_interface()
    display_chat_messages()

    user_query = st.chat_input("Type your message here...")
    if len(st.session_state.chat_history) > 1:
        # hide all header
        st.markdown(
            """
            <style>
            h1, h2, h3, h4, h5, h6 {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        block = st.empty()
        block.markdown(get_user_md(HumanMessage(content=user_query)), unsafe_allow_html=True)

    st.session_state["langchain_messages"] = langchain_messages_format(
        st.session_state["langchain_messages"]
    )

    if isinstance(st.session_state.chat_history[-1], AIMessage) is False:
        # with st.chat_message("AI"):
        block = st.empty()
        block.markdown(get_assistant_md(AIMessage(content="")), unsafe_allow_html=True)
        response = chain.invoke(
            {
                "chat_history": st.session_state.chat_history,
                "input": user_query,
                # "car_model": car_model, 
                "language": chat_lang
            },
            {
                "callbacks": [
                    StreamlitCallbackHandler(st.container())
                ]
            },
        )
        print("[DEBUG] response:")
        pprint(response)
        message = AIMessage(content=response)
        st.session_state.chat_history.append(message)
        st.rerun() 
