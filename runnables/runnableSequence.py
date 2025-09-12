from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-72B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(
    template='Tell me a Joke about {topic}',
    input_variables=['topic']
)

# this will also works, hahah
# chain = template | model | parser

chain = RunnableSequence(template,model,parser)

result = chain.invoke({'topic':'Machine learning'})

print(result)