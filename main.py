import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

load_dotenv()

def get_pdf_text(pdf):
    text = ""
    pdfReader = PdfReader(pdf)
    for page in pdfReader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
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
    st.set_page_config(page_title='JobFit AI', page_icon=':robot_face:')
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header('JobFit AI :robot_face:')
    st.subheader('JobFit AI is a tool that helps you find the right job for you based on your skills and interests.')
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
                handle_defaultinput('Suggest best suited job for this resume by providing the two best job options and expected salary in rupees in about 50 words')

if __name__ == '__main__':
    main()