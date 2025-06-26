import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os

#  Sidebar: API Key
st.sidebar.title(" GROQ API Key")
groq_api_key = st.sidebar.text_input("Enter your GROQ API key:", type="password")
st.sidebar.markdown("[Get your API key here](https://console.groq.com)")

if not groq_api_key:
    st.warning("Please enter your GROQ API key in the sidebar.")
    st.stop()

#  Load Model & Embeddings 
try:
    model = ChatGroq(
        model="Meta-Llama/Llama-4-Scout-17b-16e-Instruct",
        groq_api_key=groq_api_key
    )
except Exception as e:
    st.error(f" Failed to load ChatGroq model: {e}")
    st.stop()

embedding = HuggingFaceEmbeddings(model_name='avsolatorio/GIST-small-Embedding-v0')

# Load FAISS Vector Store 
@st.cache_resource
def load_vectorstore():
    try:
        vs = FAISS.load_local(
            "faiss_index",
            embeddings=embedding,
            allow_dangerous_deserialization=True
        )
        return vs
    except Exception as e:
        st.error(f" Failed to load FAISS index: {e}")
        return None

vectorstore = load_vectorstore()
if vectorstore is None:
    st.stop()

retriever = vectorstore.as_retriever()

# Prompt Template 
prompt = ChatPromptTemplate.from_template("""
You are a nutrition AI assistant that gives dietary recommendations based on peer-reviewed research. 

Given a patient's health condition(s) and allergy profile, provide specific and research-backed nutrition advice.
Use only medically accurate, peer-reviewed information. Cite nutritional reasoning if available.

Patient Condition(s): {condition}
Allergies: {allergies}

Context from medical literature:
{context}

What are the best dietary recommendations for this patient?
""")

document_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

#  User Input 
st.title(" Koyl AI: Nutrition Advisor")

condition = st.text_input("Enter patient condition(s):", placeholder="e.g., high blood pressure, diabetes")
allergies = st.text_input("Enter allergy profile:", placeholder="e.g., dairy, gluten")

if st.button("Get Dietary Recommendations"):
    if not condition or not allergies:
        st.warning(" Please fill in both condition and allergy fields.")
        st.stop()

    query = f"{condition} {allergies}"

    with st.spinner(" Retrieving relevant documents..."):
        retrieved_docs = retriever.invoke(query)

    with st.spinner(" Generating dietary recommendations..."):
        response = document_chain.invoke({
            "condition": condition,
            "allergies": allergies,
            "context": retrieved_docs
        })

    st.markdown("##  Dietary Recommendation")
    st.write(response)
