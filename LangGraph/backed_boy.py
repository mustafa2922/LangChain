from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

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

checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)    