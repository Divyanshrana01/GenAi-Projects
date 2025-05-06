import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import LangChainTracer
import tempfile
import os

# LangChain Tracing (Optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "legal-chat-upload"

# Streamlit UI setup
st.set_page_config(page_title="📄 Legal RAG Assistant", layout="wide")
st.title("⚖️ Legal RAG Assistant")
st.markdown("Upload a legal PDF (like the Constitution, contracts, laws, etc.) and ask questions about it.")

# Sidebar for Groq API Key
st.sidebar.title("🔐 Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
    st.stop()

# Set the key in the environment
os.environ["GROQ_API_KEY"] = groq_api_key

uploaded_file = st.file_uploader("Upload a legal or informative PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing your legal document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectordb = FAISS.from_documents(docs, embedding_model)
        retriever = vectordb.as_retriever()

        # 🧠 Legal Assistant Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful legal assistant specialized in interpreting legal documents. "
             "Use only the provided context (from the uploaded PDF) to answer the user's query. "
             "If the context does not contain enough information or the question is too complex, "
             "politely recommend that the user seek professional legal advice. "
             "Do not make up answers. Respond clearly and concisely."),
            ("user", "context:\n{context}\n\nuser_query: {input}\n\nAnswer:")
        ])

        llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(message["user"])
            with st.chat_message("assistant"):
                st.markdown(message["bot"])

        user_query = st.chat_input("Ask a legal question based on the uploaded document...")
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            tracer = LangChainTracer(project_name="legal-chat-upload")

            try:
                response = retrieval_chain.invoke({"input": user_query}, callbacks=[tracer])
                answer = response["answer"]
            except Exception as e:
                answer = "⚠️ Sorry, an error occurred while processing your request."

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.chat_history.append({"user": user_query, "bot": answer})
else:
    st.info("Please upload a PDF document to start chatting.")
