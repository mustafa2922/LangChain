from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='explain follwing topic {topic} scientifically',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize following in 7 line {content}',
    input_variables=['content']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Pulsar and Quasar'})

print(result)