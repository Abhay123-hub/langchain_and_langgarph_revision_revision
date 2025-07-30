import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence

load_dotenv()
llm = ChatOpenAI(model = "gpt-3.5-turbo")

messages = [
    ("system","You are an {technology} expert"),
    ("human","what are best applications of this technology in the {field}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

## create individual runnables(inside the chain)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

model = RunnableLambda(lambda x: llm.invoke(x))

parsed_response = RunnableLambda(lambda x:x.content)

## creating the runnable chain using all the components

chain = RunnableSequence(first=format_prompt,middle=[model],last = parsed_response)

## executing the chain (Runnable chain)
## i created three different  components using Runnable Lambda
## then connected these three runnables using RunnableSequence

response = chain.invoke({"technology":"machine learning and Data Sceince","field":"Database Management"})
print(response)