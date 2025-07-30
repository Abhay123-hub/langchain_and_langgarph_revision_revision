from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(model = "gpt-3.5-turbo")
messages = [
    SystemMessage("You are an expert in Social Media Strategy"),
    HumanMessage("Give me advice to get more views and followers on instagram")
]
response = llm.invoke(messages)

print("learning langgraph and langchain")
print(response.content)