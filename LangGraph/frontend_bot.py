import streamlit as st
from backed_boy import workflow
from langchain_core.messages import HumanMessage

CONFIG = {'configurable':{'thread_id':1}}

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for message in st.session_state['chat_history']:
     with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Ask Anything..')

if user_input:

    st.session_state['chat_history'].append({'role':'user','content':user_input})

    with st.chat_message('user'):
        st.markdown(user_input)

    response = workflow.invoke(  
        {'messages':[HumanMessage(content=user_input)]},
        config=CONFIG
    )

    AI_message = response['messages'][-1].content
    st.session_state['chat_history'].append({'role':'assistant','content':AI_message})

    with st.chat_message('assistant'):
        st.markdown(AI_message)