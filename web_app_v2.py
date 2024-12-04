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
PREFIX = '''
You are an AI assistant specializing in Gogoro Smart Scooters. Your primary role is to provide accurate and helpful information to Gogoro scooter owners based on the knowledge base provided to you.
'''

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
'''
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: [input for the tool, including Gogoro {scooter_name} if relevant]
End of response.
'''

When you have gathered all the necessary information about Gogoro Smart Scooters, formulate a comprehensive response for the user. Tailor the response based on the specific scooter model and the language specified.

'''
Thought: I now know the final answer
Action: Provide Final Answer
Final Answer: [your response]
End of response.
'''
"""

SUFFIX = '''

Begin!

Previous conversation history:
{chat_history}

Instructions:
- The user's Gogoro scooter model: {scooter_name}
- Language for the response: {language}
- Query: {input}

{agent_scratchpad}
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

agent_chain = initialize_agent(
    llm=llm,
    tools=LLM_AGENT_TOOLS,
    agent = "zero-shot-react-description",
    verbose=True,
    # output_key = "result",
    handle_parsing_errors = True,
    return_intermediate_steps=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory = memory,
    agent_kwargs={
        'prefix': PREFIX, 
        # 'format_instructions': FORMAT_INSTRUCTIONS,
        'suffix': SUFFIX
    },
    streaming=True
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
