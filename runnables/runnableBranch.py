from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template1 = PromptTemplate(
    template='write a detailed report about {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='summarize this report in less than 300 words \n {report}',
    input_variables=['report']
)

report_gen = RunnableSequence(template1, model, parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split()) > 300, RunnableSequence(template2,model,parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_gen, branch_chain)

result = chain.invoke({'topic':'Neutron Star'})

print(result)
print('====>',len(result.split()))