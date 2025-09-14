from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

model_name = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=model_name)

from langchain.schema import Document

docs = [
    Document(
        page_content="Artificial intelligence is the branch of computer science that focuses on creating systems capable of performing tasks that normally require human intelligence. These systems can include reasoning, learning, perception, and decision-making. AI is often divided into narrow AI, which performs specific tasks, and general AI, which has the potential to perform any intellectual task. Current progress is driven by machine learning algorithms and vast data availability.",
        metadata={"topic": "Artificial Intelligence"}
    ),
    Document(
        page_content="Machine learning is a subset of AI that allows systems to learn and improve from experience without being explicitly programmed. The learning process often involves training models on large datasets to identify patterns and make predictions. Supervised, unsupervised, and reinforcement learning are the main categories of machine learning. Applications range from fraud detection and recommendation systems to natural language processing.",
        metadata={"topic": "Machine Learning"}
    ),
    Document(
        page_content="Natural language processing (NLP) is a field of AI that enables computers to understand, interpret, and generate human language. NLP combines linguistics with machine learning techniques to process written and spoken text. Some of its applications include chatbots, sentiment analysis, machine translation, and text summarization. Recent advancements in transformer-based models have significantly improved accuracy and fluency.",
        metadata={"topic": "Natural Language Processing"}
    ),
    Document(
        page_content="Cloud computing delivers computing services such as storage, databases, networking, and analytics over the internet. This approach allows businesses to scale resources on demand without maintaining physical infrastructure. Public, private, and hybrid clouds provide flexibility for different use cases. Major providers include AWS, Microsoft Azure, and Google Cloud, which offer solutions for startups and enterprises alike.",
        metadata={"topic": "Cloud Computing"}
    ),
    Document(
        page_content="Cybersecurity involves protecting systems, networks, and data from digital attacks. It includes practices like encryption, access control, and vulnerability management to reduce risks. Cyber threats range from phishing attacks and ransomware to nation-state cyber warfare. Strong cybersecurity strategies involve not only technical solutions but also awareness training for individuals and employees.",
        metadata={"topic": "Cybersecurity"}
    ),
    Document(
        page_content="Blockchain is a decentralized ledger that records transactions across multiple computers in a secure and transparent manner. It eliminates the need for intermediaries by relying on consensus mechanisms to validate transactions. Beyond cryptocurrency, blockchain is used in supply chain management, healthcare, and digital identity systems. Its immutability and transparency make it an attractive solution for trust-based processes.",
        metadata={"topic": "Blockchain Technology"}
    ),
    Document(
        page_content="The Internet of Things refers to a network of connected devices that communicate and share data over the internet. These devices range from smart home appliances and wearable technology to industrial sensors. IoT enables automation, real-time monitoring, and predictive maintenance in various industries. Security and interoperability remain major challenges as billions of devices continue to come online.",
        metadata={"topic": "Internet of Things"}
    ),
    Document(
        page_content="Software engineering applies structured principles and methods to design, develop, and maintain software systems. It emphasizes quality, scalability, and efficiency in building reliable applications. Methodologies such as Agile and DevOps have become popular for iterative development and continuous delivery. The field requires collaboration between developers, testers, project managers, and stakeholders to ensure project success.",
        metadata={"topic": "Software Engineering"}
    ),
    Document(
        page_content="Data science is the interdisciplinary field that extracts insights and knowledge from structured and unstructured data. It combines statistics, machine learning, data visualization, and domain expertise to solve real-world problems. Data scientists often use Python, R, and SQL to analyze and model data. Organizations use data science to make evidence-based decisions in areas like marketing, finance, and healthcare.",
        metadata={"topic": "Data Science"}
    ),
    Document(
        page_content="Quantum computing leverages the principles of quantum mechanics to perform calculations that classical computers cannot handle efficiently. Unlike traditional bits, quantum bits (qubits) can exist in superpositions, enabling massive parallelism. Quantum algorithms, such as Shor’s algorithm for factoring, demonstrate significant speed advantages. Although still in experimental stages, quantum computing holds promise for cryptography, optimization, and material science.",
        metadata={"topic": "Quantum Computing"}
    ),
]

vector_store = Chroma(
    embedding_function= embedding_model,
    persist_directory='my_chroma_db',
    collection_name='sample'
)

# vector_store.add_documents(docs)

# result = vector_store.get(include=['embeddings','documents','metadatas'])
# print(result)

result = vector_store.similarity_search(
    query='what is block a chain technology',
    k=2
)

print(result)