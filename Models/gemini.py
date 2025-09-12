# Paid service

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

result = model.invoke('Any imapact of quantum mechanics in real life')

print(result.content)