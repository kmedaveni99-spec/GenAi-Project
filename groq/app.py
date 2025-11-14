# app.py
import streamlit as st
import os
import time
from dotenv import load_dotenv

# -------------------------------------------------
# 1. Load environment (Groq API key)
# -------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# -------------------------------------------------
# 2. Imports – **LangChain 1.0+ only**
# -------------------------------------------------
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# -------------------------------------------------
# 3. Build the vector store (cached in session_state)
# -------------------------------------------------
if "vectors" not in st.session_state:
    with st.spinner("Downloading & embedding LangSmith docs…"):
        # 3.1 Load
        loader = WebBaseLoader("https://docs.smith.langchain.com/")
        docs = loader.load()

        # 3.2 Split
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs[:50])          # limit for demo speed

        # 3.3 Embed + store
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectors = FAISS.from_documents(chunks, embeddings)

# -------------------------------------------------
# 4. LLM & Prompt
# -------------------------------------------------
llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen/qwen3-32b", 
    temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    """Answer the question using **only** the provided context.

<context>
{context}
</context>

Question: {input}"""
)

# -------------------------------------------------
# 5. Modern RAG chain (no langchain.chains!)
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = st.session_state.vectors.as_retriever()

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------------------------------
# 6. Streamlit UI
# -------------------------------------------------
st.title("ChatGroq + LangSmith RAG Demo")

user_input = st.text_input("Ask a question about LangSmith", placeholder="e.g. How do I create an API key?")

if user_input:
    start = time.process_time()
    answer = rag_chain.invoke(user_input)
    elapsed = time.process_time() - start

    st.write(f"**Answer** (took {elapsed:.2f}s):")
    st.write(answer)

    # -------------------------------------------------
    # 7. Show retrieved chunks
    # -------------------------------------------------
    with st.expander("Retrieved source chunks"):
        retrieved = retriever.invoke(user_input)
        for i, doc in enumerate(retrieved, 1):
            st.markdown(f"**Chunk {i}**")
            st.caption(doc.metadata.get("source", "unknown"))
            st.write(doc.page_content)
            st.divider()