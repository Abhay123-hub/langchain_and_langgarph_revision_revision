import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatOpenAI(model = "gpt-3.5-turbo")

## here i am making an AI application just using the concepts of langchain
## i will using one of most important components of the langchain that is chains
## specifically i will be using parllel chains of langchain
## so my objective is to get report of plot of story
## and chracter of the story
## i want this thing in output
## and in input only the movie name will be given like "inception","harry potter"

messages = [
    ('system','You have a deep knowledge of movies and you provide brief summary of movies'),
    ('human','give me the summary of {movie}')
]

prompt = ChatPromptTemplate.from_messages(messages)

## building function for analyzing the plot

def analyze_plot(movie_summary):
    template = [
        ("system","You are a movie critic"),
        ("human","based on movie summary {summary}.analyze the plot and what "
        "are its strength and weaknesses")
    ]

    prompt = ChatPromptTemplate.from_messages(template)
    return prompt.format(summary = movie_summary)

def analyze_characters(movie_summary):
    template = [
        ('system','You are a movie critic'),
        ('human','based on the movie summary {summary},analyze all the characters and what'
        'are there main strength and weaknesses. ')
    ]
    prompt = ChatPromptTemplate.from_messages(template)

    return prompt.format(summary = movie_summary)

def combine_verdict(plot,characters):
    return f"The movie plot is \n\n{plot} and the character analysis is\n\n{characters}"

analyze_plot_chain = (
    RunnableLambda(lambda x: analyze_plot(x) ) | llm | StrOutputParser()
)


analyze_chracters_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser()
)


final_chain = (
    prompt | llm | StrOutputParser() 
    | RunnableParallel(branches = {"plot":analyze_plot_chain,"characters":analyze_chracters_chain})
    | RunnableLambda(lambda x: combine_verdict(x["branches"]["plot"],x["branches"]["characters"]))

)

response = final_chain.invoke({"movie":"Harry Potter"})

print(response)