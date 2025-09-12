from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

llm2 = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.1',
    task='text-generation'
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Create some bulletpoints from this text \n {text}',
    input_variables=['text']
)
 
prompt2 = PromptTemplate(
    template='Generate some multiple choice questions from provided text \n {text}',
    input_variables=['text']
)
 
prompt3 = PromptTemplate(
    template='Merge the follwing points and Mcqs \n points => {points} and Mcqs => {mcqs}',
    input_variables=['points','mcqs']
)

# creating paralell chain
collateral_chains = RunnableParallel({
    'points': prompt1 | model1 | parser,
    'mcqs': prompt2 | model2 | parser
})

# merging outputs from paralell chains
consolidate_chain = prompt3 | model2 | parser

# final chain
chain = collateral_chains | consolidate_chain

with open(r"D:\AAA_structured_repos\@GenAI\chains\doc.txt",'r',encoding='utf-8') as f:
    text = f.read()

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()