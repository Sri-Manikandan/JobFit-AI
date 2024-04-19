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

st.set_page_config(page_title="Candidate AI", page_icon="🧠")
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
        chunk_size=750,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4")
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain

def handle_userinput(user_question):
    response  = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            usermessage = st.chat_message('user')
            usermessage.write(message.content)
        else:
            aimessage = st.chat_message('ai')
            aimessage.write(message.content)

def handle_defaultinput(user_question):
    response  = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2!=0:
            aimessage = st.chat_message('ai')
            aimessage.write(message.content)

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header('JobFit AI :robot_face:')
    st.subheader('JobFit AI is a tool that helps you analyze your job resume in HR perspective based on your skills and interests.')
    user_question = st.text_input('Ask a question about your resume:')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader('Your Resume')
        pdf = st.file_uploader('Upload your resume:')
        if st.button('Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)
                # handle_defaultinput('Suggest best suited job for this resume by providing the two best job options and expected salary in rupees in about 50 words')
                handle_defaultinput('You are an experienced Human Resource Manager who is specialized in analyzing the job resumes of the candidate. You have been asked to analyze the resume of a candidate and provide the feedback. Analyze the strength and weakness of the resume and provide the feedback.')
                # handle_defaultinput('Suugest the missing keywords and skills in the resume to make it more suitable for the job title Data Scientist in about 50 words')
        
        st.divider()


if __name__ == '__main__':
    main()