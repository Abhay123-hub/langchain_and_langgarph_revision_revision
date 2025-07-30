print("learning langchain and langgraph")
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

llm = ChatOpenAI(model = "gpt-3.5-turbo")
result = llm.invoke("what is Machine Learning").content
print(result)