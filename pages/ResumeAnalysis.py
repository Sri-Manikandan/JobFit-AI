import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from menu import menu_with_redirect

st.set_page_config(page_title="Resume Analysis", page_icon="🧠")
menu_with_redirect()

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
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain


def main():
    st.header('JobFit Candidate AI :robot_face:')
    st.subheader('Resume Analysis Tool')
    st.subheader('Resume Analysis tool helps you to analyze the strength and weakness of your resume.')
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
            st.write(response['answer'])

if __name__ == '__main__':
    main()