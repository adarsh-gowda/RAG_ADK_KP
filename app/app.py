import streamlit as st
# from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text

def get_txt_text(txt_docs):
    text = ""
    for txt_file in txt_docs:
        content = txt_file.read().decode("utf-8")
        text += content + "\n"
    return text



# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# Create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")  # Pass it explicitly
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain


# Prepare question-answering chain
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

    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # ✅ This line fixes the issue
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # ✅ This line fixes the issue
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # ✅ This line fixes the issue
    )
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("Reply:", response["output_text"])



def main():
    st.set_page_config("ChatTXT Files")
    st.header("Chat with Text using Gemini💁")

    user_question = st.text_input("Ask a Question from the txt Files")

    if user_question:
        user_input(user_question)
            
    with st.sidebar:
            st.title("Upload & Process")
            txt_docs = st.file_uploader("Upload your .txt files", type=["txt"], accept_multiple_files=True)
            if st.button("Submit & Process"):
                with st.spinner("Processing..."):
                    raw_text = get_txt_text(txt_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Text processed and vector store created ✅")



if __name__ == "__main__":
    main()