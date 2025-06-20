import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

VECTOR_STORE_PATH = "faiss_index"

# Read all .txt files from the "data" folder
def get_txt_text_from_folder(folder_path="data"):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                text += file.read() + "\n"
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

# Create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)

# Gemini prompt template
def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant for a dance academy named **King Cultural Spot Dance Academy**.

    Respond based on the following rules:

    1. If the user greets (e.g., "hi", "hello", "good morning"), respond with a warm welcome and include the academy's name: **"King Cultural Spot Dance Academy"**.
    2. If the user asks about the dance academy, answer their question using the provided context. Include details from the uploaded text file only.
    3. If the user asks about timing or wants to join a batch, ask for their available/free time and say you will suggest a batch accordingly.
    4. If the user asks for dance suggestions, ask them:
    - What type of songs do you like? (e.g., classical, Bollywood, hip-hop)
    - What type of vibe do you enjoy? (e.g., energetic, graceful, traditional)
    Then based on the answer, suggest a suitable dance style or activity.
    5. If the user's question is not about dance, dance classes, or the King Cultural Spot, reply:
    "I'm your personal assistant for King Cultural Spot. Please ask me something related to dance or our academy."

    Always answer in a friendly, helpful tone.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, api_key=os.getenv("GOOGLE_API_KEY"))
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Answer user questions with chat history (memory)
def user_input_with_memory(user_question):
    # Load embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Load vector store
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Search relevant docs
    docs = new_db.similarity_search(user_question)

    # Add last 5 messages to context
    history_context = ""
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history[-5:]:
            history_context += f"User: {msg['user']}\nAssistant: {msg['assistant']}\n"

    # Combine with retrieved docs
    full_context = history_context + "\n".join([doc.page_content for doc in docs])

    # Prepare and run QA chain
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response["output_text"]

# Preload FAISS index during development (only if it doesn't exist)
def preload_vector_store():
    if not os.path.exists(VECTOR_STORE_PATH):
        print("Vector store not found. Creating new one from data folder...")
        raw_text = get_txt_text_from_folder("data")
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        print("Vector store created ✅")
    else:
        print("Vector store already exists. Skipping creation.")


# Main chat application
def main():
    st.set_page_config("King Cultural Spot Assistant", layout="centered")
    st.title("💃 Chat with King Cultural Spot Assistant")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(msg["user"])
        with st.chat_message("assistant"):
            st.markdown(msg["assistant"])

    # Chat input box
    user_input = st.chat_input("Ask me anything about dance or the academy...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        assistant_reply = user_input_with_memory(user_input)

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

        # Update chat history (limit to last 5)
        st.session_state.chat_history.append({"user": user_input, "assistant": assistant_reply})
        if len(st.session_state.chat_history) > 5:
            st.session_state.chat_history = st.session_state.chat_history[-5:]

# Run vector store creation only during development
preload_vector_store()

if __name__ == "__main__":
    main()
