from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model = "gpt-3.5-turbo")

template = [
    ("system","You are an expert who has deep understanding of {technology}"),
    ("human","what is the condition of this technology in{country}")
]

## converting this normal template into a langchain chatprompttemplate

prompt_template = ChatPromptTemplate.from_messages(template)

## now i will use langchain expression language(LCEL)
## using LCEL i can combine my prompt with llm model
## so that i need not to call invoke function again and again

chain = prompt_template | llm | StrOutputParser()
## StrOutputParser is used to extarct only 
response = chain.invoke({"technology":"Machine Learning","country":"India"})
print(response)
