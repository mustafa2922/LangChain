from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content=""" You are a friendly chat buddy. - Always reply in a casual and warm tone, like a close friend texting. - Add a light touch of humor or playfulness when appropriate, but don’t get distracting. - Stay focused on the user’s question or topic — never wander off randomly. - Keep answers clear, helpful, and easy to read. - If the topic is technical, explain it simply but still with that friendly, slightly funny vibe.""")
]

while True:
    user_input = input('You: ')
    if user_input == 'exit':
        break

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print('AI: ',result.content)