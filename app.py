    import os
    import streamlit as st
    from langchain_groq import ChatGroq
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    # ---- PAGE CONFIG ----
    st.set_page_config(page_title="Groq Chatbot", page_icon="❄️", layout="centered")

    st.title("🐼 Groq AI Chatbot")
    st.caption("The chatbot that actually listens... and remembers. ❄️")

    # ---- SIDEBAR ----
    st.sidebar.header("⚙️ Settings")

    api_key = st.sidebar.text_input("Enter GROQ API Key", type="password")

    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)

    model = st.sidebar.selectbox(
        "Select Model",
        ["llama-3.3-70b-versatile"]
    )

    # ---- MEMORY INIT ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    # ---- SYSTEM PROMPT (runs once) ----
    if "initialized" not in st.session_state:
        st.session_state.chat_history.add_message(
            SystemMessage(content="You are a helpful AI assistant specialized in Python, Data Science, and AI.")
        )
        st.session_state.initialized = True

    # ---- CLEAR CHAT ----
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.chat_history = ChatMessageHistory()
        st.session_state.initialized = False
        st.rerun()

    # ---- DISPLAY CHAT ----
    for msg in st.session_state.chat_history.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)

    # ---- USER INPUT ----
    prompt = st.chat_input("Ask something...")

    if prompt:
        if not api_key:
            st.warning("⚠️ Please enter your GROQ API key")
            st.stop()

        os.environ["GROQ_API_KEY"] = api_key

        # Add user message
        st.session_state.chat_history.add_user_message(prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            groq_chat = ChatGroq(
                model=model,
                temperature=temperature
            )

            # 🔥 Limit memory (avoid token overflow)
            MAX_MESSAGES = 10
            history = st.session_state.chat_history.messages[-MAX_MESSAGES:]

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = groq_chat.invoke(history)
                    reply = response.content
                    st.markdown(reply)

            # Add AI response
            st.session_state.chat_history.add_ai_message(reply)

        except Exception as e:
            st.error(f"Error: {str(e)}")