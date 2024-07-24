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
from pymongo import MongoClient
import gridfs

st.set_page_config(page_title="Candidate AI", page_icon="🧠")
menu_with_redirect()
Page_style="""
<style>
    [data-testid="stApp"]{
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
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain

def handle_defaultinput(user_question,vectorstore):
    conversation = get_conversation_chain(vectorstore)
    response  = conversation({'question':user_question,"chat_history":""})
    aimessage = st.chat_message('ai')
    aimessage.write(response['answer'])
    DeepLake.force_delete_by_path("./my_deeplake_candidate")

def main():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['jobfit']
    fs = gridfs.GridFS(db,collection='resumes')

    st.header('JobFit Candidate AI :robot_face:')
    st.subheader('Job Match Tool')
    # st.subheader('Job Match Tool helps us to check if the resume aligns with the job description')
    user_question = st.text_input('Enter the job description:')
    pdf = st.file_uploader('Upload your resume:')
    if st.button('Process'):
        with st.spinner('Processing...'):
            if pdf:
                file_path = 'E:/New folder/downloads/' + pdf.name
                with open(file_path, 'rb') as file_data:
                    data = file_data.read()
                fs.put(data, filename=pdf.name)

            raw_text = get_pdf_text(pdf)

            text_chunks = get_text_chunks(raw_text)

            vectorstore = get_vectorstore(text_chunks)
            prompt = f'''Rate the resume on a scale of 1 to 100 percentage based on the job specifications: "{user_question}" and provide feedback on the same in about 50 words.'''
            handle_defaultinput(prompt,vectorstore)
    st.divider()


if __name__ == '__main__':
    main()