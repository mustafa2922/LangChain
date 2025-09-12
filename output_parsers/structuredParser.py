from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen2.5-7B-Instruct',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='1st Mindblowing fact about the topic'),
    ResponseSchema(name='fact_2', description='2nd though provoking fact about the topic'),
    ResponseSchema(name='fact_3', description='3rd real life related fact about the topic'),
    ResponseSchema(name='fact_4', description='4th deep and philosophical fact about the topic'),
    ResponseSchema(name='fact_5', description='5th scientific fact about the topic with little calculations'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 5 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'neutron star'})

print(result)