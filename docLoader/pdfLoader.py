from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

loader = PyPDFLoader('docLoader/universeStart.pdf')

docs = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template='write down the summary and all key events discussed in this book in roman urdu \n {text}',
    input_variables=['text']
)

chain = prompt | model | parser

result = chain.invoke({'text':docs})

print(result)