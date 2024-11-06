import os
from typing import List, Tuple, Union, Dict
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from GoLang.src.llms.llm_openai import Chat_OpenAI

llm = Chat_OpenAI()

# Initialize conversation history in session state if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app layout
st.title("AI Chat Interface")
st.write("Interact with an AI using Streamlit")

# User input
if prompt := st.chat_input("Say something"):
    # Add user message to chat history
    st.session_state.chat_history.append(("user", prompt))
    
    # Get response from AI model
    ai_response = llm(prompt)
    
    # Add AI response to chat history
    st.session_state.chat_history.append(("assistant", ai_response))

# Display chat history using for loop
for speaker, message in st.session_state.chat_history:
    st.chat_message(speaker).write(message)