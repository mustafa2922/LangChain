from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=model_name)

vector_store = Chroma(
    embedding_function= embedding_model,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

retriever = vector_store.as_retriever(search_kwargs={'k':2})

query = 'what is Machine Learning and where it is used'

results = retriever.invoke(query)

for i,doc in enumerate(results):
    print(f'========= Result {i+1} =========')
    print(doc.page_content)