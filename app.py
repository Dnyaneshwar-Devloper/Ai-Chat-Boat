import os
import streamlit as st
from langchain_groq import ChatGroq

# ---- PAGE CONFIG ----
st.set_page_config(page_title=" AI Assistant", page_icon="🐼", layout="wide")

# ---- CUSTOM CSS ----
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 1em; background-color: #ff4b4b; color: white; }
    .stTextInput>div>div>input { border-radius: 5px; }
            
            .sidebar-img {
        display: block;
        margin-left: 0;
        margin-right: auto;
        width: 100px; /* Adjust size as needed */
        border-radius: 0px;
        margin-bottom: 0px;
            height: 1em;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- SIDEBAR SETTINGS ----
with st.sidebar:
    st.image(r"D:\ChatBot-1\ai emoji.png")
   
    st.title("⚙️Settings")
    api_key = st.text_input("Enter GROQ API Key", type="password", placeholder="Enter Key...")
    
    st.divider()
    model = st.selectbox("Select Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"])
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.7)
    
    uploaded_file = st.file_uploader("📂 Upload context file", type=["txt", "pdf", "md"])
    if uploaded_file:
        st.success(f"📄 {uploaded_file.name} ready")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []


# ---- CHAT INTERFACE ----
# Centering the title and caption using a container or direct HTML
st.markdown("<h1 style='text-align: center;'>🤖 AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>🚀 Meet your new AI sidekick.</p>", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me anything....."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if not api_key:
        st.error("Please provide a GROQ API Key in the sidebar.")
    else:
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    os.environ["GROQ_API_KEY"] = api_key
                    llm = ChatGroq(model=model, temperature=temperature)
                    
                    # Include file info if exists
                    context = f"[File: {uploaded_file.name}] " if uploaded_file else ""
                    full_query = context + prompt
                    
                    response = llm.invoke(full_query)
                    answer = response.content
                    st.markdown(answer)
            
            # Add assistant message to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
