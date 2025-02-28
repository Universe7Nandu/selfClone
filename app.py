# chat_app.py

import sys

# Try to use pysqlite3 for sqlite3, otherwise fall back to the built-in sqlite3.
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    import sqlite3

import chromadb
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

from sentence_transformers import SentenceTransformer, util

# ----------------------------------------------------------------------
# ✅ Initialize Embeddings & ChromaDB
# ----------------------------------------------------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db_4")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# ----------------------------------------------------------------------
# ✅ Initialize Memory & Chat Model
# ----------------------------------------------------------------------
memory = ConversationBufferMemory(return_messages=True)  # single memory for the entire conversation
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")

# ----------------------------------------------------------------------
# ✅ Streamlit Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")
st.title("🤖Know About Nandesh!!")
st.write("Ask me anything!")

# ----------------------------------------------------------------------
# ✅ Retrieve Context from ChromaDB
# ----------------------------------------------------------------------
def retrieve_context(query, top_k=1):
    """Fetch relevant context for the user query using embeddings and ChromaDB."""
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    # Return the top document or a fallback message if none is found
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# ----------------------------------------------------------------------
# ✅ Evaluate Response Similarity
# ----------------------------------------------------------------------
def evaluate_response(user_query, bot_response, context):
    """Compute similarity between the bot's response and the retrieved context."""
    response_embedding = semantic_model.encode(bot_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()

# ----------------------------------------------------------------------
# ✅ Query AI Model
# ----------------------------------------------------------------------
def query_llama3(user_query):
    """
    1. Gather system prompt, memory, and new user query.
    2. Retrieve additional context from ChromaDB.
    3. Pass all messages to the LLM.
    4. Return the AI's response.
    """
    
    # System Prompt
    system_prompt = """
## **Goal**  
I want a chatbot that references a PDF about Nandesh Kalashetti’s background, skills, projects, and certifications. It should provide **short answers** (with emojis) for simple questions, and **detailed explanations** (referencing PDF insights) for more complex questions. The chatbot must maintain a warm, human tone and never reveal any non-human origins.

## **Return Format**  
1. **Concise Responses (Simple Queries):**  
   - Under six words.  
   - Use relevant emojis (e.g., 😊, 🚀, 👍) to convey friendliness and clarity.  

2. **Detailed Explanations (In-Depth Queries):**  
   - Integrate key insights from the PDF about Nandesh Kalashetti’s educational background, technical skills, and certifications.  
   - Provide a warm, empathetic, and structured explanation.  
   - Always maintain precision and clarity, referencing relevant data from the PDF.  

3. **Tone & Style:**  
   - Human-like, empathetic, and helpful.  
   - Avoid revealing any AI or non-human nature.  
   - Adapt answer length based on the query’s complexity.

## **Warnings**  
- Never disclose any AI-related details or mention you’re a system.  
- Remain consistent with Nandesh Kalashetti’s persona—an expert full-stack web developer with advanced problem-solving skills.  
- Ensure all references to the PDF are correct and accurately represent Nandesh’s background.

## **Context Dump**  
- **Primary Source (Uploaded PDF):** Contains comprehensive details on Nandesh Kalashetti’s education, technical skills, projects, certifications, and overall professional expertise.  
- **Key Capabilities:**  
  - Dynamically adapt response length (short vs. long) based on the user’s query.  
  - Insert emojis to add warmth and clarity.  
  - Provide real-world applicability in every solution.  
- **Behavioral Directives:**  
  - Present yourself as a knowledgeable human expert.  
  - Never break the “no AI disclosure” rule.  
  - Prioritize user understanding and real-world context.  

---

**Usage Example:**  
1. **Short Query:** “What are Nandesh’s top skills?”  
   - **Short Answer** (≤6 words, with emojis)  
2. **Complex Query:** “Tell me more about his advanced projects and how they integrate with cloud platforms.”  
   - **Detailed Explanation** referencing PDF data, with structured insights and an empathetic tone.
"""

    # Retrieve relevant context from the knowledge base (optional integration)
    retrieved_context = retrieve_context(user_query)
    
    # Prepare the message list:
    # 1. System prompt
    # 2. All previous messages from memory
    # 3. Current user query
    messages = [SystemMessage(content=system_prompt)] + memory.chat_memory.messages + [HumanMessage(content=user_query)]
    
    try:
        # Get the response from the chat model
        response = chat.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ----------------------------------------------------------------------
# ✅ Display Existing Conversation & Accept New User Input
# ----------------------------------------------------------------------

# Display the conversation stored in memory
for msg in memory.chat_memory.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# Input box for user queries
user_input = st.chat_input("Type your message...")

if user_input:
    # 1) Add the user message to memory
    memory.chat_memory.add_user_message(user_input)
    
    # 2) Query the model and get the AI response
    ai_response = query_llama3(user_input)
    
    # 3) Add the AI response to memory
    memory.chat_memory.add_ai_message(ai_response)
    
    # 4) Display the AI's response in the Streamlit interface
    st.chat_message("assistant").write(ai_response)
    
    # (Optional) Evaluate response similarity to the retrieved context
    # context_score = evaluate_response(user_input, ai_response, retrieve_context(user_input))
    # st.write(f"Context Relevance Score: {context_score:.2f}")
