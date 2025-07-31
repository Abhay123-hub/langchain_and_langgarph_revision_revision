import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model = "gpt-3.5-turbo")

messages = [
    ("system","You are expert in the {subject}"),
    ("human","Tell me applications of {subject} in {field}")
]

## converting this simple message into langchain prompt_template

promt_template = ChatPromptTemplate.from_messages(messages)

prepration_for_translation = RunnableLambda(lambda output: {"text":output,"language":"Hindi"})

template = [
    ("system","you convert any one language into another langauge"),
    ("human","convert the given {text} text into {language} language")
]

translator_template = ChatPromptTemplate.from_messages(template)

## now i am ready to build my first sequential chain

chain = promt_template | llm | StrOutputParser() | prepration_for_translation | translator_template | llm | StrOutputParser()

## generating the response using the above sequential chain
response = chain.invoke({'subject':'machine learning','field':'aerospace engineering'})
print(response)
