import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from sqlalchemy import Column, Integer, String, create_engine, Table, select, insert, MetaData



st.set_page_config(page_title="Hirer AI", page_icon="🧠")

load_dotenv()

def database():
    metadata_obj = MetaData()

    job = Table(
        'job',
        metadata_obj,
        Column('id', Integer, primary_key=True),
        Column('applicant_name', String),
        Column('rating', String),
    )

    engine = create_engine('sqlite:///./hirer.db', echo=False)

    metadata_obj.create_all(engine)

    observations = [
        ["Sri Manikandan", "90"],
        ["Sai Sidharthan", "80"]
    ]
    conn = engine.connect()
    for obj in observations:
        insert_stmt = insert(job).values(
            applicant_name=obj[0],
            rating=obj[1]
        )
        conn.execute(insert_stmt)

    select_statement = select(job)
    result = conn.execute(select_statement)
    rows = result.fetchall()
    for row in rows:
        st.write(row)

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

def handle_defaultinput(user_question):
    response  = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2!=0:
            st.write(f"AI: {message.content}")

def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header('JobFit AI :robot_face:')
    st.subheader('JobFit AI is a tool that helps you find the right job for you based on your skills and interests.')

    with st.sidebar:
        st.subheader('Job Specifications')
        job_specification = st.text_area("Enter the job specifications:")
        st.divider()
        st.subheader('Your Resumes')
        pdfs = st.file_uploader('Upload resumes:', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('Processing...'):
                for pdf in pdfs:
                    raw_text = get_pdf_text(pdf)

                    text_chunks = get_text_chunks(raw_text)

                    vectorstore = get_vectorstore(text_chunks)

                    st.session_state.conversation = get_conversation_chain(vectorstore)

                    handle_defaultinput(f'Rate the resumes on a scale of 1 to 100 based on the job specifications: "{job_specification}" and provide feedback on the same in about 50 words')

    database()
if __name__ == '__main__':
    main()