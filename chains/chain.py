from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field , EmailStr
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class HypoData(BaseModel):
    name: str = Field(description='name of the person')
    email: EmailStr = Field(description='email of the person')
    age: int = Field(description='age between 18 and 50', gt=17 ,lt=51)
    city: str = Field(description='city of the specified country')
    phone: str = Field(description='phone number of specified country with country code')

parser = PydanticOutputParser(pydantic_object=HypoData)

prompt = PromptTemplate(
    template='Given name, email, age, city and phone number of any {reigon} hypothetical person \n{format_instruction}',
    input_variables=['reigon'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({'reigon':'USA'})

print(result)

chain.get_graph().print_ascii()