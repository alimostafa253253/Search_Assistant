import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# 🎨 Streamlit Page Config
st.set_page_config(page_title="🔎 AI Search Assistant", page_icon="🤖")

# 🏗️ Sidebar for Settings
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("🔑 Enter your Groq API Key:", type="password")
serp_api_key = st.sidebar.text_input("🔑 Enter your SerpAPI Key:", type="password")

# 🛠️ Arxiv, Wikipedia, and Web Search Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

#  Web Search with SerpAPI (Dynamic API Key)
if serp_api_key:
    search = Tool(
        name="SerpAPI Search",
        func=SerpAPIWrapper(serpapi_api_key=serp_api_key).run,  # Pass the user-provided API key
        description="Search the web using SerpAPI"
    )
else:
    search = None

# 🎤 Chat Interface Title
st.title("🤖 Chat with AI-Powered Search")

st.markdown("""
✨ **Welcome to the AI Search Assistant!**  
Ask me anything, and I'll find the best answer from **Wikipedia**, **Arxiv**, or **the Web**. 🌍📚🔬
""")

# 📝 Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi! I'm an AI-powered assistant. How can I help you today?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ✍️ User Input
if prompt := st.chat_input(placeholder="🔍 Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(f"**🗣️ You:** {prompt}")

    if api_key:
        # 🤖 AI Model Selection
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
        tools = [arxiv, wiki]
        if search:  # Only add search tool if API key is provided
            tools.append(search)

        # 🔎 AI Search Agent
        search_agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True
        )

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])

            # Save response in chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(f"**🤖 AI:** {response}")

    else:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
