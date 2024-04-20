import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from sqlalchemy import Column, Integer, String, create_engine, Table, select, insert, MetaData
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from menu import menu_with_redirect

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.vectorstores import DeepLake

st.set_page_config(page_title="Hirer AI", page_icon="🧠")
menu_with_redirect()

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

    conn = engine.connect()

    select_statement = select(job)
    result = conn.execute(select_statement)
    rows = result.fetchall()
    for row in rows:
        st.sidebar.write(row)

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
    # vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings)
    dataset_path = "./my_deeplake/"
    # vectorstore = DeepLake(dataset_path=dataset_path, embedding=OpenAIEmbeddings(),read_only=True)
    vectorstore = DeepLake.from_texts(text_chunks,dataset_path=dataset_path, embedding=OpenAIEmbeddings())
    return vectorstore

def handle_defaultinput(resp):
    # database()
    db = SQLDatabase.from_uri("sqlite:///./hirer.db")
    llm = OpenAI(temperature=0, verbose=True)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    agent_executor.run(
    f"Insert values into job table with applicant_name as '{resp.name}' and rating as '{resp.rating}' and give the answer with the query alone"
    )
    DeepLake.force_delete_by_path("./my_deeplake")


def get_conversation_chain(vectorstore,user_question):
    st.session_state.answer = ""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    class Job(BaseModel):
        name: str = Field(description="applicant's name in the resume")
        rating: int = Field(description="rating of the candidate's resume")

    parser = PydanticOutputParser(pydantic_object=Job, output_variables=["applicant_name", "rating"])
    prompt = PromptTemplate(
        template="Format the context given to u.\n{format_instructions}\n{answer}\n",
        input_variables=["answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    conversation_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    chain = prompt | llm | parser
    response  = conversation_chain.invoke({'query': user_question})
    resp = chain.invoke({"answer":response['result']})
    st.session_state.answer = resp
    if resp.name != None or resp.rating != None:
        handle_defaultinput(resp)
    else:
        resp.name = "Manish"
        resp.rating = 50
        handle_defaultinput(resp)


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "answer" not in st.session_state:
        st.session_state.answer = None
    st.header('JobFit Recruiter AI :robot_face:')
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

                    get_conversation_chain(vectorstore,f'Retrieve the applicant Name and Rate the resume on a scale of 1 to 100 based on the job specifications: "{job_specification}", if the resume doesnt match the the required skills for the given job description give a score between 10 and 30 and get the applicant name in the resume and rating of the resume as output.')

    aimessage = st.chat_message('ai')
    aimessage.write(st.session_state.answer)
if __name__ == '__main__':
    main()