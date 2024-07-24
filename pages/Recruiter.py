import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PyPDF2 import PdfReader
from sqlalchemy import Column, Integer, String, create_engine, Table, MetaData
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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import pandas as pd


st.set_page_config(page_title="Hirer AI", page_icon="🧠")
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

def create_database():
    metadata_obj = MetaData()

    job = Table(
        'job',
        metadata_obj,
        Column('id', Integer, primary_key=True),
        Column('applicant_name', String, unique=True),
        Column('rating', Integer),
    )

    engine = create_engine('sqlite:///./hirer.db', echo=False)

    metadata_obj.create_all(engine)

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
    dataset_path = "./my_deeplake/"
    vectorstore = DeepLake.from_texts(text_chunks,dataset_path=dataset_path, embedding=OpenAIEmbeddings())
    return vectorstore

def handle_defaultinput(resp):
    db = SQLDatabase.from_uri("sqlite:///./hirer.db")
    llm = OpenAI(temperature=0, verbose=True)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
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
    retriever=vectorstore.as_retriever()
    template = '''
        Use the following pieces of context to answer the user's question. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ----------------
        Context:
        {context}
        ----------------
        Question:
        {query}
        ----------------
        Finally Format the answer in the following format:
        {format_instructions}

        Answer:
        '''
    prompt = PromptTemplate(
        template=template,
        input_variables=["context","query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    conv_chain = RunnableParallel({"context":retriever,"query":RunnablePassthrough()}) | prompt | llm | parser
    resp  = conv_chain.invoke(user_question)
    if resp.name != None or resp.rating != None:
        handle_defaultinput(resp)
    else:
        resp.name = "Manish"
        resp.rating = 50
        handle_defaultinput(resp)

def retrieve_candidates():
    db = SQLDatabase.from_uri("sqlite:///./hirer.db")
    llm = OpenAI(temperature=0)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    response = agent_executor.run(
        "Select applicant_name, rating from job where rating > 60 order by rating limit 2"
    )
    st.session_state.answer = response
    
def barchart():
    engine = create_engine('sqlite:///./hirer.db',echo=False)

    query = '''
    select * from job
    '''
    df = pd.read_sql_query(query, engine)
    st.bar_chart(data=df,x='applicant_name',y='rating')

def main():
    if "answer" not in st.session_state:
        st.session_state.answer = ""
    if "barchart" not in st.session_state:
        st.session_state.barchart = None
    st.header('JobFit Recruiter AI :robot_face:')
    st.subheader('Your Resumes')

    pdfs = st.file_uploader('Upload resumes:', accept_multiple_files=True)
    with st.sidebar:
        st.subheader('Job Specifications')
        job_specification = st.text_area("Enter the job specifications:")
        st.divider()
    if st.button('Process'):
        with st.spinner('Processing...'):
            create_database()
            for pdf in pdfs:
                raw_text = get_pdf_text(pdf)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                get_conversation_chain(vectorstore,f'Retrieve the applicant name and rate the resume on a scale of 1 to 100 based on the job specifications: "{job_specification}" and get the applicant name in the resume and rating of the resume as output.')
            retrieve_candidates()
            aimessage = st.chat_message('ai')
            aimessage.write(st.session_state.answer)
            barchart()
if __name__ == '__main__':
    main()