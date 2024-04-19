import streamlit as st

def authenticated_menu():
    if st.sidebar.button("Switch Accounts"):
        st.session_state.role = None
        st.switch_page("start-page.py")
    if st.session_state.role == "candidate":
        st.sidebar.page_link("pages/candidate.py", label="Candidate AI")
        st.sidebar.divider()
    if st.session_state.role == "hirer":
        st.sidebar.page_link("pages/hirer.py", label="Recruiter AI")
        st.sidebar.divider()

def unauthenticated_menu():
    st.sidebar.page_link("start-page.py", label="Home")
    st.sidebar.divider()

def menu():
    if "role" not in st.session_state or st.session_state.role is None:
        unauthenticated_menu()
        return
    authenticated_menu()

def menu_with_redirect():
    if "role" not in st.session_state or st.session_state.role is None:
        st.switch_page("start-page.py")
    menu()