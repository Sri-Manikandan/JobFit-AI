import streamlit as st
from pymongo import MongoClient
from menu import menu_with_redirect
from dotenv import load_dotenv

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

def main():
    st.title("Resume Data")
    st.write("This is a page for viewing resume data.")

    client = MongoClient("mongodb://localhost:27017/")
    db = client["jobfit"]
    
    if st.button('Get Database'):
        pipeline = [
            {
                '$group': {
                    '_id': '$filename', 
                    'document': {
                        '$first': '$$ROOT'
                    }
                }
            }, 
            {
                '$replaceRoot': {
                    'newRoot': '$document'
                }
            }
        ]
        result = db['resumes.files'].aggregate(pipeline)
        for document in result:
            st.write(document)

if __name__ == "__main__":
    main()