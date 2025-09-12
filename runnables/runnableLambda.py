from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-72B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(
    template='write a scientific fact with numbers about {topic}',
    input_variables=['topic']
)

fact_gen_chain = RunnableSequence(template,model,parser)

parallel_chian = RunnableParallel({
    'fact':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x:len(x.split()))
})

chain = RunnableSequence(fact_gen_chain , parallel_chian)

result = chain.invoke({'topic':'neutron stars'})

print(result)