# ⚖️ Legal RAG Assistant – Chat with Your Legal PDFs!

The Legal RAG Assistant is a web-based application built to assist users with legal inquiries. It allows individuals to upload legal documents (such as the Constitution of India, contracts, or other legal texts) and ask legal questions. The system uses Retrieval-Augmented Generation (RAG) with Generative AI to extract relevant information from the uploaded document and provide answers grounded in the provided context.

---

## 🚀 Features

- 🧠 RAG-based question answering using LLaMA 4 Scout (via Groq API)
- 📄 Upload any legal or informative PDF
- 🔍 Semantic document retrieval with FAISS & HuggingFace embeddings
- 💬 Chat-like interface with scrollable history
- 📊 LangSmith tracing support for debugging and analysis
- 🛡️ Adheres to safe AI principles (no hallucinations, warns on insufficient data)

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Groq API](https://console.groq.com/)
- [Meta LLaMA 4 Scout 17B](https://huggingface.co/meta-llama/Meta-Llama-4-Scout)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/legal-rag-assistant.git
cd legal-rag-assistant
pip install -r requirements.txt
