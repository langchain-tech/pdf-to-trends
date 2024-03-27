import os
import streamlit as st
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]
    if text and 35 < y < 770:
        page_contents.append(text)

def convert_to_json(document_content):
    messages = [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(
            content=document_content
        )
    ]
    answer = chat.invoke(messages)
    return answer.content

def prepare_pdfs(pdf_files):
    document_content = ""
    for pdf_file in pdf_files:
        with pdf_file as file:
            pdf_reader = PdfReader(file)
            page_contents = []
            for page in pdf_reader.pages:
                page_contents.append(page.extract_text())
            document_content += "\n".join(page_contents)
    return document_content


st.title("PDF Chatbot")

pdf_files = st.file_uploader("Upload PDF files:", accept_multiple_files=True)

openai_key = st.text_input("Enter your OpenAI API key:")
if openai_key:
  os.environ["OPENAI_API_KEY"] = openai_key
  chat = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0)
  embeddings = OpenAIEmbeddings()

query = st.text_input("Enter your query for pdf data:")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

if st.button("Get Answer to Query"):
	if pdf_files and openai_key and query:
		docs = prepare_pdfs(pdf_files)
		chunks = text_splitter.split_text(docs)
		db = FAISS.from_texts(chunks, embeddings)
		chain = load_qa_chain(chat, chain_type="stuff", verbose=True)
		docs = db.similarity_search(query)
		print("docsearch",docs)
		response = chain.run(input_documents=docs, question=query)
		st.write("Query Answer:")
		st.write(response)
	else:
		st.warning("Please upload PDF files, enter your OpenAI API key and query")
