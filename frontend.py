
import streamlit as st
# Make sure main_chain.py is in the same folder as this file!
from main_chain import Standard_User, Premium_User 

#set page
st.set_page_config(page_title="Video RAG AI", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        font-weight: bold;
    }
    .stChatInput {
        border-color: #4A90E2 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #f7f9fc;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🎬 Smart Video Assistant</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Your personal AI mentor for video content.</p>", unsafe_allow_html=True)
st.divider()


# SIDEBAR

with st.sidebar:
    st.header("👤 User Settings")
    
   
    user_type = st.radio(
        "Select Account Type:",
        ("Standard User", "Premium User")
    )
    
  
    user_name = "Prasuk Jain"
    
    st.info(f"Current Mode: **{user_type}**")
    st.success(f"Logged in as: **{user_name}**") 

 


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about your videos..."):
    
    

    # A. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Initialize Backend
    if user_type == "Standard User":
        active_user = Standard_User(user_name) 
    else:
        active_user = Premium_User(user_name)  

    # C. Get AI Response
    with st.chat_message("assistant"):
        with st.spinner(f"{user_type} AI is thinking..."):
            try:
               
                
                
                response = active_user.ask_query(prompt)
                st.markdown(response)
                
               
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")

