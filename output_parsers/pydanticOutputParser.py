from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation',
    temperature=2
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):

    name: str = Field(description='fullname of the person')
    age: int = Field(description='age of the person', gt=18)
    city: str = Field(description='city of the person') 


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me the name age and city of any hypothetical person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({})

print(result)