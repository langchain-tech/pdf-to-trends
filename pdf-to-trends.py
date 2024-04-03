import os
import pathlib
import streamlit as st
from pypdf import PdfReader
from tempfile import NamedTemporaryFile
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter


### Uncomment import 'pdb' this to use debugger in the app
### Use this code in between any file or function to stop debugger at any point pdb.set_trace()
import pdb

def convert_to_json(document_content):
    """
    Convert document content to JSON format.
    
    Args:
        document_content (str): Content of the document.
    
    Returns:
        str: JSON formatted document content.
    """
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

def prepare_files(files):
    """
    Prepare files for processing by extracting their content.
    
    Args:
        files (list): List of uploaded files.
    
    Returns:
        str: Concatenated content of all files.
    """
    document_content = ""
    for file in files:
        if file.type == 'application/pdf':
            page_contents = handle_pdf_file(file)
        elif file.type == 'text/csv':
            page_contents = handle_csv_file(file)
        else:
            st.write('File type is not supported!')
        document_content += "".join(page_contents)
    return document_content

def handle_pdf_file(pdf_file):
    """
    Handle PDF files by extracting text content from each page.
    
    Args:
        pdf_file (UploadedFile): Uploaded PDF file.
    
    Returns:
        list: List of text content extracted from each page.
    """
    document_content = ''
    with pdf_file as file:
        pdf_reader = PdfReader(file)
        page_contents = []
        for page in pdf_reader.pages:
            page_contents.append(page.extract_text())
        document_content += "\n".join(page_contents)
    return document_content

def handle_csv_file(csv_file):
    """
    Handle CSV files by extracting content.
    
    Args:
        csv_file (UploadedFile): Uploaded CSV file.
    
    Returns:
        str: Concatenated content of all pages in the CSV file.
    """
    with csv_file as file:
        uploaded_file = file.read()
        with NamedTemporaryFile(dir='.', suffix='.csv') as f:
            f.write(uploaded_file)
            f.flush()
            loader = CSVLoader(file_path=f.name)
            document_content = "".join([doc.page_content for doc in loader.load()])
    return document_content

st.set_page_config(page_title='AI PDF Chatbot', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("PDF Chatbot")

files = st.file_uploader("Upload PDF files:", accept_multiple_files=True, type=["csv", "pdf"])

openai_key = st.text_input("Enter your OpenAI API key:")
if openai_key:
  os.environ["OPENAI_API_KEY"] = openai_key
  chat = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0)
  embeddings = OpenAIEmbeddings()

query = st.text_input("Enter your query for pdf data:")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

if st.button("Get Answer to Query"):
    if files and openai_key and query:
        document_content = prepare_files(files)
        chunks = text_splitter.split_text(document_content)
        db = FAISS.from_texts(chunks, embeddings)
        chain = load_qa_chain(chat, chain_type="stuff", verbose=True)
        docs = db.similarity_search(query)
        print("docsearch", docs)
        response = chain.run(input_documents=docs, question=query)
        st.write("Query Answer:")
        st.write(response)
    else:
        st.warning("Please upload PDF and CSV files, enter your OpenAI API key and query")