import streamlit as st
from backend_bot import workflow, model, load_chat_titles, save_chat_title

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
import uuid

template = PromptTemplate(
    template='Summarize this chat into a short title (max 4 words) and just return the title.\n chat: {messages}',
    input_variables=['messages'],
)

# ================= utility functions ==============
def gen_thread_id():
    return str(uuid.uuid4())

def generate_chat_title(messages, llm):
    _prompt = template.invoke({'messages':messages})
    title =  llm.invoke(_prompt)
    return title.content

def gen_new_session():
    if len(st.session_state['chat_history']) == 0:
        return
    thread_id = gen_thread_id()
    st.session_state['thread_id'] = thread_id
    append_thread(thread_id)
    st.session_state['chat_history'] = []

def append_thread(thread_ID, title='Untitled Chat'):
    if thread_ID not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_ID] = title

def load_chat_state(thread_ID):
    state = workflow.get_state({'configurable':{'thread_id': thread_ID}}).values
    if "messages" in state:
        return state['messages']
    else:
        return []
    
def switch_sessions(thread_id):
    st.session_state['thread_id'] = thread_id
    messages = load_chat_state(thread_id)
    structured_messages = [
        {'role': 'user' if isinstance(m, HumanMessage) else 'assistant', 'content': m.content}
        for m in messages
    ]
    st.session_state['chat_history'] = structured_messages
    st.rerun()

# ==================================================
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = gen_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = load_chat_titles()

append_thread(st.session_state['thread_id'])

# side bar
st.sidebar.title('ChateX')
if st.sidebar.button('Start new chat'):
    gen_new_session()

st.sidebar.header('Chat Sessions')
for thread_id, title in st.session_state['chat_threads'].items():
    if st.sidebar.button(title, key=thread_id):
        switch_sessions(thread_id)


# ==================================================
for message in st.session_state['chat_history']:
     with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Ask Anything..')
CONFIG = {'configurable':{'thread_id': st.session_state['thread_id']}}

if user_input:

    st.session_state['chat_history'].append({'role':'user','content':user_input})

    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        AI_message  = st.write_stream(
            message_token.content for message_token, metadata in workflow.stream(
                {'messages':[HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            ))

    
    st.session_state['chat_history'].append({'role':'assistant','content':AI_message})

    thread = st.session_state['thread_id']
    chat_data = st.session_state['chat_threads']

    if chat_data.get(thread, 'Untitled Chat') == 'Untitled Chat':
        title = generate_chat_title(st.session_state['chat_history'], model)
        st.session_state['chat_threads'][thread] = title
        save_chat_title(thread, title)
        st.rerun()