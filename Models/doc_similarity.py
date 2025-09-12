import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name)

docs = [
"Pakistan is famous for its rich culture and the majestic Karakoram mountains.",
"Pakistan has one of the largest irrigation systems in the world.",
"Japan is known for its advanced technology and efficient bullet trains.",
"Japan has a deep-rooted tradition of tea ceremonies and cherry blossoms.",
"Brazil is famous for football, samba dance, and the Amazon rainforest.",
"Brazil has one of the most biodiverse ecosystems on Earth."
]

query = 'Which country is famous for samba dance?'

doc_embeddings = embeddings.embed_documents(docs)
query_embeddings = embeddings.embed_query(query)

similarities = []


for i in doc_embeddings:

    v1 = np.array(i)
    v2 = np.array(query_embeddings)

    dot = np.dot(v1,v2)

    norm1 = np.sqrt(np.sum(v1 ** 2))
    norm2 = np.sqrt(np.sum(v2 ** 2))

    similarities.append(dot / (norm1 * norm2))

similarities = [float(x) for x in similarities]

index,score = sorted(enumerate(similarities),key=lambda x:x[1])[-1]

print('All similarities => ', [round(x,2) for x in similarities])
print('Similarity Score => ' ,score)
print('Most matched sentence => ' ,docs[index])