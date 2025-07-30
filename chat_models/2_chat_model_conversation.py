import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
chat_history = [] ## initializing emopty list where all chat history will be stored in form
## of HumanMessage,AIMessage,SystemMessage

llm = ChatOpenAI(model = "gpt-3.5-turbo")
system_message = SystemMessage("You are a helpfull AI assistant!")
chat_history.append(system_message)

while True:
    query = input("You:")
    if query.lower() == "exit":
        break
    else:
        chat_history.append(HumanMessage(query))
        response = llm.invoke(chat_history).content
        chat_history.append(AIMessage(response))
        print(chat_history)
        print(response)

