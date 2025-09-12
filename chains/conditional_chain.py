from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser , StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Sentiment(BaseModel):
    sentiment: Literal['positive','negative'] =  Field(description='Give the sentiment of the review from customer')

str_Parser = StrOutputParser()
parser = PydanticOutputParser(pydantic_object=Sentiment)

classify_prompt = PromptTemplate(
    template="Classify the sentiment of the user's feedback as either positive or negative \n {feedback} \n{format_insruction}",
    input_variables=['feedback'],
    partial_variables={'format_insruction':parser.get_format_instructions()}
)

pos_resp_gen_prompt = PromptTemplate(
    template='Generate an humble and thanksgiving response for this positive feedback from user in a funny way \n {feedback}',
    input_variables=['feedback']
)
 
neg_resp_gen_prompt = PromptTemplate(
    template='Generate a trolling and mocking response for this negative feedback from user \n {feedback}',
    input_variables=['feedback']
)
 
categorizer_chain = classify_prompt | model | parser

branch_chains = RunnableBranch(
    (lambda x:x.sentiment == 'positive', pos_resp_gen_prompt | model | str_Parser),
    (lambda x:x.sentiment == 'negative', neg_resp_gen_prompt | model | str_Parser),
    RunnableLambda(lambda x:'could not find any sentiment')
)

chain = categorizer_chain | branch_chains

result = chain.invoke({'feedback':'its terrible of all, camera sucks and speakers are like 90s radio, thumbs down shit'})

print(result)

chain.get_graph().print_ascii()