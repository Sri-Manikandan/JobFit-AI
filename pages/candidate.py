import streamlit as st
from PyPDF2 import PdfReader
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
api_key = st.secrets["openai"]["OPENAI_API_KEY"]

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
    dataset_path = "./my_deeplake_candidate/"
    vectorstore = DeepLake.from_texts(text_chunks,dataset_path=dataset_path, embedding=OpenAIEmbeddings(api_key=api_key))
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
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

def download_file(download_loc, db, fs, file_name):
    data = db['resumes'].files.find_one({"filename": file_name})
    if data:
        fs_id = data['_id']
        out_data = fs.get(fs_id).read()

    with open(download_loc, 'wb') as output:
        output.write(out_data)

    print("Download Completed!")

def main():
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['jobfit']
    fs = gridfs.GridFS(db,collection='resumes')

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main page
    st.header('JobFit Candidate AI :robot_face:')
    st.subheader('Resume Chat')
    # st.subheader('JobFit AI is a tool that helps you analyze your job resume in HR perspective based on your skills and interests.')
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
                DeepLake.force_delete_by_path("./my_deeplake_candidate")
        st.divider()
    
    # if st.button('Download Resume'):
    #     download_file('D:/Projects/JobFit AI/src/downloads/'+pdf.name, db, fs, pdf.name)


if __name__ == '__main__':
    main()