## importing the main opackages
import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
load_dotenv() ## activating all the secret/api keys
## in order to run my ai model

file_path = "D:\langchain_and_langgarph_revision_revision\documents\lord_of_the_rings.txt"

file = TextLoader(file_path)
documents = file.load() ## this contains all text of the file

## now i need to convert this text into word vector embeddings 
## before doing that i need convert the entire chunk into parts

splitter = CharacterTextSplitter(chunk_size = 1000,chunk_overlap=50)
docs = splitter.split_documents(documents)


## now i need to convert all of these into word vector embeddings
## for that i need openai embedding model

embeddings = OpenAIEmbeddings(
    model = "text-embedding-3-small"
)

vectorstore = Chroma.from_documents(
    docs,embeddings
)
retriever = vectorstore.as_retriever(
    search_type = "similarity_score_threshold", ## type of similarity score i will beusing
    search_kwargs = {"k":4,"score_threshold":0.5} ## getting top four chunks only who has score more than 0.5
)

question = input("Enter your any question regarding the Lords of The Rings movie:\n")
response = retriever.invoke(question)


for i, doc in enumerate(response, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")





