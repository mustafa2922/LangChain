from langchain_community.document_loaders import TextLoader

loader = TextLoader('docLoader/doc.txt')

docs = loader.load()

print(docs[0])