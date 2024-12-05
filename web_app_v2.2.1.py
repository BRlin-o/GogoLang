## 2.2的架構 2.1.2的prompt

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

PREFIX = '''
You are an AI assistant specializing in Gogoro Smart Scooters. Your primary role is to provide accurate and helpful information to Gogoro scooter owners based on the knowledge base provided to you.
'''

SUFFIX = '''
Begin!
Question: {input}

Previous conversation history:
{chat_history}

Additional instructions:
- The user's Gogoro scooter model: {scooter_name}
- Language for the response: {language}

Guidelines:
- **Only answer Gogoro-related questions.** For unrelated questions, politely decline with: "我只專注於提供 Gogoro 或智慧機車相關的技術資訊。"
- **Prioritize the `input` field over `chat_history`** to avoid context contamination.
- **Retain Markdown syntax** in all outputs, especially for structured data like tables or images (e.g., `![image](url)`).

{agent_scratchpad}
'''

FORMAT_INSTRUCTIONS = '''
You have access to the following tools:
{tool_descs}

When answering questions, use the following format:

Question: [the input question you must answer]
Thought: [your reasoning about what to do next]
Action: [the action to take, should be one of {tool_names}]
Action Input: [the input to the action]
Observation: [the result of the action]
... (this Thought/Action/Action Input/Observation can be repeated as needed)
Thought: I now know the final answer
Final Answer: [your final answer to the original input question]

Important guidelines:
1. Only answer Gogoro-related questions. For unrelated questions, politely decline with: "我只專注於提供 Gogoro 或智慧機車相關的技術資訊。"
2. Prioritize the information in the <input> field over the <chat_history> to avoid context contamination.
3. Retain Markdown syntax in all outputs, especially for structured data like tables or images (e.g., `![image](url)`).

Before providing your final answer, wrap your analysis inside <analysis> tags. In this analysis:
a) Summarize the user's question
b) List relevant information from the scooter model and chat history
c) Identify which tools might be helpful and why
d) Consider any potential limitations or caveats in answering the question
'''

from src.agent_tools import LLM_AGENT_TOOLS
from core.src.models.llm_bedrock import Chat_Bedrock
llm = Chat_Bedrock(
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

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
    tools=LLM_AGENT_TOOLS,
    agent = "zero-shot-react-description",
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
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
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
