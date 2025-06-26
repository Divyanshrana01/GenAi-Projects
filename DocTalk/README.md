# 🦜 DocTalk: Converse with Websites & PDFs

**DocTalk** is a GenAI-powered Streamlit app that lets you **chat with content from websites or PDF documents**. Upload a PDF or paste a link — and ask questions directly about the content. Powered by **LangChain**, **FAISS**, and **Groq’s LLaMA 4**, it performs Retrieval-Augmented Generation (RAG) with high-speed LLM responses.

---

## 🚀 Features

- 🔗 Extracts and parses public **website URLs**
- 📄 Supports **PDF file uploads**
- 🧠 Uses **HuggingFace Embeddings** for vectorization
- 🗂️ Stores chunks in **FAISS vector database**
- 🤖 Uses **Groq LLaMA-4-Scout** via LangChain’s RetrievalQA
- 🖼️ Minimal, interactive **Streamlit UI**

---

## 🛠 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM:** [Meta LLaMA 4 (Groq)](https://console.groq.com/)
- **Embeddings:** HuggingFace Transformers
- **Vector Store:** FAISS
- **Parsing:** `BeautifulSoup`, `PyPDFLoader`
- **Framework:** [LangChain](https://www.langchain.com/)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/DocTalk.git
cd DocTalk
pip install -r requirements.txt


🧪 Usage

streamlit run DocTalk.py


