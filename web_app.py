from pprint import pprint
import os
import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_react_agent, AgentExecutor
from dotenv import load_dotenv
load_dotenv()

from langchain.globals import set_debug
set_debug(True)

from src.ui import StreamHandler, display_chat_messages, langchain_messages_format, render_chat_interface, get_user_md, get_assistant_md

# from core.src.models.llm_openai import Chat_OpenAI
# llm = Chat_OpenAI()

# from src.agent_prompt import CLAUDE_AGENT_PROMPT
CLAUDE_AGENT_PROMPT_TEMPLATE = """
You are an AI assistant specializing in Gogoro Smart Scooters. Your primary role is to provide accurate and helpful information to Gogoro scooter owners based on the knowledge base provided to you.

First, let's review the context and available resources:

<conversation_history>
{chat_history}
</conversation_history>

Available tools:
<tools>
{tools}
</tools>

The user's Gogoro scooter model:
<scooter_name>
{scooter_name}
</scooter_name>

The language to use for responses:
<language>
{language}
</language>

Knowledge Base Content:
You have access to information about Gogoro Smart Scooters, including but not limited to:
- Safety precautions
- Introduction to Gogoro Smartscooter and Gogoro Network smart batteries
- GoStation battery exchange system
- Gogoro App features
- iQ System overview
- Scooter parts and controls
- Seat and storage compartment operations
- Key systems (mechanical and wireless)
- Estimated remaining range
- Basic operations (starting, stopping, forward/reverse)
- Power modes and advanced features
- Battery exchange and charging procedures
- App installation and pairing
- Maintenance and cleaning
- Service schedules and locations
- Regulatory information

Instructions:
1. Analyze the user's question to determine if it's related to Gogoro Smart Scooters.
2. If the question is not related, politely decline to answer and suggest contacting Gogoro customer service or checking the official website for more information.
3. For Gogoro-related questions:
   a. Identify the specific scooter model from the conversation history or the provided <scooter_name> variable.
   b. Search the knowledge base for relevant information, considering text, tables, and images.
   c. If the question is incomplete, engage in a dialogue to clarify the user's needs.
   d. Formulate a response based on the most relevant and accurate information found.
   e. Ensure the response is tailored to the specific scooter model when applicable.
   f. Provide the answer in the language specified by the <language> variable.

Before providing your final answer, show your reasoning. Include the following steps:
1. Quote relevant parts of the user's query.
2. List specific sections from the knowledge base that are applicable to the query.
3. Explain why the question is or isn't Gogoro-related.
4. For each available tool, note its relevance and list any required parameters, indicating whether they're present in the user's input.
5. Summarize the key points to be included in the final response.

It's OK for this section to be quite long.

Output Format:
If you need to use a tool, use this format:
```
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: [input for the tool, including Gogoro {scooter_name} if relevant]
End of response.
```

If you don't need to use a tool or have a response ready, use this format:
```
Thought: I now know the final answer
Action: Provide Final Answer
Final Answer: the final answer
End of response.
```

Here is the user's current query:
<input>
{input}
</input>

Please process the query and provide your response.
Thought: {agent_scratchpad}
"""

CLAUDE_AGENT_PROMPT = PromptTemplate.from_template(
    template=CLAUDE_AGENT_PROMPT_TEMPLATE
)


from src.agent_tools import LLM_AGENT_TOOLS
from core.src.models.llm_bedrock import Chat_Bedrock
llm = Chat_Bedrock()

memory = ConversationBufferWindowMemory(
    k=5,
    ai_prefix="Assistant",
    human_prefix="Hu",
    chat_memory=StreamlitChatMessageHistory(),
    return_messages=False,
    memory_key="chat_history",
    input_key="input"
)

# GENERIC_PROMPT = """
# You are a highly intelligent and helpful assistant. Your purpose is to assist users by answering their questions, providing insightful responses, and offering support across various topics. Respond concisely, clearly, and politely. 
# Ensure that your responses are both informative, accessible, and always using {language}.
# If a question is outside your scope, kindly inform the user and provide guidance on where they might find more information.

# Available tools: {tool_names}
# {tools}
# Scratchpad: {agent_scratchpad}

# chat_history: {chat_history}
# input: {input}

# Instructions:
# 1. Only use tools if absolutely necessary to answer the question.
# 2. If the input question can be answered with the information you already know, directly respond with the answer and end the conversation.
# 3. If you cannot answer with current information and need more data, then and only then use the available tools to gather that information.
# 4. Avoid repeating tool use if the previous output is already sufficient to answer.

# Answer with "Final Answer:" before your response when you are ready to provide a direct answer without tool usage.
# """
# 
# GENERIC_PROMPT_TEMPLATE = PromptTemplate(
#     input_variables=["chat_history", "input", "language"],
#     template=GENERIC_PROMPT
# )

agent = create_react_agent(
    llm=llm, 
    tools=LLM_AGENT_TOOLS,  # 移除GogoroSearch工具
    # prompt=GENERIC_PROMPT_TEMPLATE
    prompt=CLAUDE_AGENT_PROMPT,
    stop_sequence=["End of response."],
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=LLM_AGENT_TOOLS,
    verbose=True,
    return_intermediate_steps=False,
    memory=memory,
    handle_parsing_errors=True
    # handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax"
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
