## 2.1 new version for agent tools trying


from pprint import pprint
import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler, StreamingStdOutCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.manager import CallbackManager
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

from langchain.globals import set_debug
set_debug(True)

from src.ui import StreamHandler, display_chat_messages, langchain_messages_format, render_chat_interface, get_user_md, get_assistant_md

# from core.src.models.llm_openai import Chat_OpenAI
# llm = Chat_OpenAI()

# from src.agent_prompt import CLAUDE_AGENT_PROMPT
# PREFIX = '''
# You are an AI assistant specializing in Gogoro Smart Scooters. Your primary role is to provide accurate and helpful information to Gogoro scooter owners based on the knowledge base provided to you.
# '''
PREFIX = '''
You are an AI assistant specializing in Gogoro Smart Scooters. Your primary role is to provide accurate and helpful information to Gogoro scooter owners based on the knowledge base provided to you.

When responding to queries:
- **Always prioritize the `input` field over `chat_history`.** Do not let previous questions influence your current answer.
- If the question is unrelated to Gogoro or smart scooters, **politely decline to answer** with: "我只專注於提供 Gogoro 或智慧機車相關的技術資訊。"
- Retain all Markdown syntax exactly as provided, especially for images (e.g., `![image](url)`).
- Structure responses based on the provided `FORMAT_INSTRUCTIONS` (if applicable).

Strictly follow this sequence. Always validate if the question is within the Gogoro topic before proceeding.
'''

FORMAT_INSTRUCTIONS = '''
All responses must follow these formatting rules:
1. Use Markdown for all outputs, including headers (`#`), bullet points (`-`), and images (`![image](url)`).
2. If the query involves step-by-step instructions, use an ordered list (`1.`, `2.`, etc.).
3. For troubleshooting, structure your response as:
   - Problem Identification
   - Step-by-Step Solution
   - Final Check and Recommendation
4. If outputting technical data, format it in a table:
   | Parameter        | Value        |
   |------------------|--------------|
   | Example          | Data         |
5. If additional references are required, provide links in Markdown format: `[Link Text](URL)`.
'''

SUFFIX = '''
Begin!

Previous conversation history:
{chat_history}

Instructions:
- The user's Gogoro scooter model: {scooter_name}
- Language for the response: {language}
- Query: {input}

Guidelines:
- **Answer only Gogoro-related questions.** For unrelated questions, politely decline and do not provide an answer.
- Retain Markdown syntax as is for all outputs.
- Always prioritize the `input` question over `chat_history` to avoid context contamination.

Markdown Rendering Reminder:
- Retain all Markdown syntax exactly as provided, especially for image syntax (e.g., `![image](url)`).
- Do not alter or convert Markdown syntax to plain text descriptions.

{agent_scratchpad}
'''

from src.tools import create_gogoro_tool
from core.src.models.llm_bedrock import Chat_Bedrock
aws_knowledge_base_id = os.getenv("KB_ID")
llm = Chat_Bedrock(
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler(st.container())])
)

memory = ConversationBufferWindowMemory(
    k=5,
    ai_prefix="Assistant",
    human_prefix="Hu",
    chat_memory=StreamlitChatMessageHistory(),
    return_messages=False,
    memory_key="chat_history",
    input_key="input"
)
# from langchain.memory import ConversationSummaryBufferMemory

# memory = ConversationSummaryBufferMemory(
#     llm=llm,
#     max_token_limit=5000
# )

agent_chain = initialize_agent(
    llm=llm,
    tools=[create_gogoro_tool(aws_knowledge_base_id, llm)],
    agent = "chat-zero-shot-react-description",
    verbose=True,
    handle_parsing_errors = True,
    return_intermediate_steps=True,
    max_iterations=3,
    memory = memory,
    agent_kwargs={
        'prefix': PREFIX, 
        'format_instructions': FORMAT_INSTRUCTIONS,
        'suffix': SUFFIX
    },
    streaming=True,
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler(st.container())])
)

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

    scooter_name, chat_lang = render_chat_interface()
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
                "scooter_name": scooter_name, 
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
