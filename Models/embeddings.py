from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name)

texts = ["Encapsulation in OOP is data hiding.", "Polymorphism allows flexibility."]

vectors = embeddings.embed_documents(texts)

print(vectors)