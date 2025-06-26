🦜 DocTalk: Converse with Websites & PDFs
DocTalk is a lightweight, GenAI-powered Streamlit app that lets you chat with content from websites or PDF documents. Whether it’s a legal PDF, a blog article, or a documentation page — just upload or paste the link, and start asking questions. Powered by LangChain, FAISS, and Groq’s LLaMA 4, it performs retrieval-augmented generation (RAG) to give grounded, contextual answers.

🚀 Features
🔗 Load and parse content from any public website

📄 Upload PDF documents for instant analysis

🧠 Automatically chunk, embed, and index text using HuggingFace Embeddings

📚 Uses FAISS vector store for fast retrieval

🤖 Queries are answered using Groq LLaMA 4 via LangChain RetrievalQA

⚡ Simple and clean Streamlit interface

🛠 Tech Stack
Frontend/UI: Streamlit

LLM: Meta LLaMA 4 Scout via Groq

Embeddings: HuggingFace Transformers

Vector Store: FAISS

Document Parsing: BeautifulSoup (Web), PyPDFLoader (PDF)

Framework: LangChain

📦 Installation
bash
Copy code
git clone https://github.com/yourusername/doctalk.git
cd doctalk
pip install -r requirements.txt
Make sure you have a valid Groq API key.

🧪 Usage
bash
Copy code
streamlit run DocTalk.py
Then:

Paste a website URL or upload a PDF

Enter your Groq API key in the sidebar

Click “Analyze”

Ask natural language questions based on the content

🔐 API Key
You need a Groq API key to access the LLaMA 4 model.

