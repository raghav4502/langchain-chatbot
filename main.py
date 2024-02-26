from dotenv import load_dotenv
import os
import streamlit as st
import pathlib
from langchain_experimental.agents import create_csv_agent
from langchain_openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']

def processor(filename):
    return pathlib.Path(filename).suffix
    

    
    
# what is the transaction name for id number 114

st.title("Extractor Chatbot :computer:")



uploaded_file = st.file_uploader("Upload files here")

if uploaded_file is not None:

    ext = processor(uploaded_file.name)
    user_question = st.text_input("Which file do you want to search?")

    if ext == ".csv":
        agent = create_csv_agent(
                OpenAI(temperature=0), uploaded_file, verbose=True)
        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                st.write(agent.run(user_question))

    if ext == ".pdf":
       if uploaded_file is not None:
           pdf_reader = PdfReader(uploaded_file)
           text = ""
           for page in pdf_reader.pages:
               text += page.extract_text()

           text_splitter = CharacterTextSplitter(
           separator="\n", chunk_size = 1000, chunk_overlap = 200, length_function = len
           )  
       
           chunks = text_splitter.split_text(text)

           embeddings = OpenAIEmbeddings()
           knowledge_base = FAISS.from_texts(chunks, embeddings)

           if user_question:
               docs = knowledge_base.similarity_search(user_question)

               llm = OpenAI()
               chain = load_qa_chain(llm, chain_type="stuff")
               response = chain.run(input_documents = docs, question = user_question)
               st.write(response)