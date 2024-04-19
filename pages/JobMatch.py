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

def handle_defaultinput(user_question,vectorstore):
    conversation = get_conversation_chain(vectorstore)
    response  = conversation({'question':user_question})
    aimessage = st.chat_message('ai')
    aimessage.write(response['answer'])

def main():
    st.header('JobFit AI :robot_face:')
    st.subheader('Job Match Tool helps us to match the resume with the job description')
    user_question = st.text_input('Enter the job description:')
    pdf = st.file_uploader('Upload your resume:')
    if st.button('Process'):
        with st.spinner('Processing...'):
            raw_text = get_pdf_text(pdf)

            text_chunks = get_text_chunks(raw_text)

            vectorstore = get_vectorstore(text_chunks)
            prompt = f'''Rate the resume on a scale of 1 to 100 percentage based on the job specifications: "{user_question}" and provide feedback on the same in about 50 words', applicant_name, percentage match, job_description, feedback'''
            handle_defaultinput(prompt,vectorstore)
    st.divider()


if __name__ == '__main__':
    main()