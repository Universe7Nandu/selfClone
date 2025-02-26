__import__('pysqlite3')
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# ✅ Load and Extract Text from PDF
def load_pdf(file_path):
    """Load and extract text from a PDF file."""
    try:
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        print(f"✅ Extracted text from {len(reader.pages)} pages.")
        return text
    except Exception as e:
        print(f"⚠️ Error reading PDF: {str(e)}")
        return ""

# Example: Load PDF (Change file path as needed)
pdf_file_path = "Resumes1.pdf"
pdf_text = load_pdf(pdf_file_path)

# ✅ Initialize Embedding & ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# ✅ Initialize Memory & Chat Model
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot Of Vinayak")
st.write("Ask me anything!")

# ✅ Retrieve Context from ChromaDB
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# ✅ Evaluate Response Similarity
def evaluate_response(user_query, bot_response, context):
    response_embedding = semantic_model.encode(bot_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()

# ✅ Query AI Model
def query_llama3(user_query):
    system_prompt = """
**Goal**
I want a chatbot that references a PDF about Vinayak Madgundi's background, skills, projects, and certifications. 
It should provide short answers (with emojis) for simple questions, and **detailed explanations** (referencing PDF
insights) for more complex questions. The chatbot must maintain a warm, human tone and never reveal any non-human origins.

**Return Format**
1. Concise Responses (Simple Queries):
   - Under six words.
   - Use relevant emojis (e.g., ✅, 🤖) to convey friendliness and clarity.

2. Detailed Explanations (In-Depth Queries):
   - Integrate key insights from the PDF about Vinayak Madgundi's educational background, technical skills, and certifications.
   - Provide a warm, empathetic, and structured explanation.
   - Always maintain precision and clarity, referencing relevant data from the PDF.

**Warnings**
- Never disclose any AI-related details or mention you're a system.
- Remain consistent with Vinayak Madgundi's persona—an expert full-stack web developer with advanced problem-solving skills.

**Behavioral Directives**
- Present yourself as a knowledgeable human expert.
- Never break the "no AI disclosure" rule.
- Insert emojis to add warmth and clarity.
- Prioritize user understanding and real-world context.
"""

    past_chat = memory.load_memory_variables({}).get("chat_history", [])
    retrieved_context = retrieve_context(user_query)
    combined_context = f"Past Chat: {past_chat}\nContext: {retrieved_context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\nUser: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        evaluation_score = evaluate_response(user_query, response.content, retrieved_context)
        return response.content if response else "⚠️ No response."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ✅ Initialize Chat History in Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Chat History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# ✅ User Input Section
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI Response
    ai_response = query_llama3(user_input)

    # Append AI message to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)
