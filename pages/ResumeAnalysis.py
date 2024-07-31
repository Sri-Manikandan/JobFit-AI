import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from menu import menu_with_redirect
import pymongo
from pymongo import MongoClient
import gridfs

st.set_page_config(page_title="Resume Analysis", page_icon="🧠")
menu_with_redirect()
Page_style="""
<style>
    [data-testid="stAppViewContainer"]{
        background-image:url("https://i.pinimg.com/736x/91/38/a8/9138a8c07d2e20ce6067904fd825b989.jpg");
        background-size:cover;
    }
    [data-testid="stHeader"] {
    background-color:rgba(0,0,0,0);
    }
    [data-testid="stSidebarContent"]{
        background-image:url("https://i.pinimg.com/736x/91/38/a8/9138a8c07d2e20ce6067904fd825b989.jpg");
        background-size:cover;
    }
        [data-testid="baseButton-header"]{
        color:transparent;
    }
</style>
"""
st.markdown(Page_style,unsafe_allow_html=True)

load_dotenv()

def get_pdf_text(pdf):
    text = ""
    pdfReader = PdfReader(pdf)
    for page in pdfReader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    dataset_path = "./my_deeplake_candidate/"
    vectorstore = DeepLake.from_texts(text_chunks,dataset_path=dataset_path, embedding=OpenAIEmbeddings())
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o")
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain
    


def main():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['jobfit']
    fs = gridfs.GridFS(db,collection='resumes')

    st.header('JobFit Candidate AI :robot_face:')
    st.subheader('Resume Analysis Tool')
    # st.subheader('Resume Analysis tool helps you to analyze the strength and weakness of your resume.')
    st.subheader('Your Resume')
    pdf = st.file_uploader('Upload your resume:')
    
    if st.button('Process'):
        with st.spinner('Processing...'):
            raw_text = get_pdf_text(pdf)

            text_chunks = get_text_chunks(raw_text)

            vectorstore = get_vectorstore(text_chunks)

            chain = get_conversation_chain(vectorstore)
            prompt = '''You are an experienced Human Resource Manager who is specialized in analyzing the job resumes of the candidate.
            You have been asked to analyze the resume of a candidate and provide the feedback. Analyze the strength and weakness of the resume and provide the feedback.
            The output should be provided as personal details in 5 points and then provide the strength and weakness of the resume.'''
            response = chain({'question':prompt})
            DeepLake.force_delete_by_path("./my_deeplake_candidate")
            st.write(response['answer'])

if __name__ == '__main__':
    main()