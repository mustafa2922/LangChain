from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

class StateMessage(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:StateMessage) -> StateMessage:
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages':[response]}

graph = StateGraph(StateMessage)
graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

connection_obj = sqlite3.connect(database='chats.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=connection_obj)

workflow = graph.compile(checkpointer=checkpointer)    


def get_all_threads():
    all_threads = set()
    for checkpoints in checkpointer.list(None):
        all_threads.add(checkpoints.config['configurable']['thread_id'])

    return list(all_threads)

def init_title_table():
    with connection_obj:
        connection_obj.execute('''
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        )
        ''')

init_title_table()

def save_chat_title(thread_id,title):
    with connection_obj:
        connection_obj.execute(
        "INSERT OR REPLACE INTO chat_titles (thread_id,title) VALUES(?,?)",
        (thread_id,title)
        )

def load_chat_titles():
    with connection_obj:
        cursor = connection_obj.execute("SELECT thread_id, title FROM chat_titles")
        return dict(cursor.fetchall())