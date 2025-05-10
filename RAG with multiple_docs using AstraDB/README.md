# 📚 Multi-Document RAG Chatbot with AstraDB & LangChain

This project implements a Generative AI chatbot that performs Retrieval-Augmented Generation (RAG) over multiple document formats (`.txt`, `.pdf`, `.pptx`). Built with LangChain and integrated with AstraDB as the vector store, it enables accurate, context-aware responses based on custom document collections.

---

## 🚀 Features

- 📄 Loads `.txt`, `.pdf`, and `.pptx` documents from a local folder
- ✂️ Chunks documents using LangChain's `RecursiveCharacterTextSplitter`
- 📦 Stores embeddings in **AstraDB** for scalable retrieval
- 💬 (Assumed) Query interface using a Large Language Model (e.g., LLaMA, OpenAI, etc.)

---

## 🧱 Tech Stack

- Python
- LangChain
- AstraDB
- LLMs (Groq, OpenAI, etc.)
- Document Loaders (`TextLoader`, `PyPDFLoader`, `UnstructuredPowerPointLoader`)

---

## 📁 Project Structure

