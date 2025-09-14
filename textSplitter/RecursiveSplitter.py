from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('docLoader/doc.txt') as f:
    text = f.read()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=212,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

print(len(chunks))