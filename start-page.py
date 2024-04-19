import streamlit as st
from menu import menu

st.set_page_config(page_title="JobFit AI", page_icon="🧠", layout="centered")

if 'role' not in st.session_state:
    st.session_state.role = None

def main():
    st.title("Welcome to the JobFit AI!")
    st.subheader("Please choose your role?")
    col1,col2 = st.columns([1,5])
    with col1:
        if st.button("Candidate"):
            st.session_state.role = "candidate"
            st.switch_page("pages/candidate.py")
    with col2:
        if st.button("Recruiter"):
            st.session_state.role = "hirer"
            st.switch_page("pages/hirer.py")
        
menu()

if __name__ == '__main__':
    main()