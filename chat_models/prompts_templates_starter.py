from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
llm = ChatOpenAI(model = "gpt-3.5-turbo")

template = "Write a {tone} email to {company} for the {role}.mentioning" \
"{skill} as a key strength.keep it maximum to four lines "
## this is a simple template ,now i need to convert this simple template as langchain prompt template
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"tone":"confident","company":"google","role":"machine learning engineer","skill":"Data Science"})
response = llm.invoke(prompt)

print(prompt)
print(response.content)

## suppose i want my emails in a particular way like very formal or casual or anything
## for that i need to add system message along with human message 
## let us see how actually it works


messages = [
    ("system","what ever is asked to you,you always give answer in {tone} way .It does not matter how the question is?"),
    ("human","Tell me applicatins of {first_topic} and {second_topic}")
]
## converting this messages list containing of two tuples into a ChatPromptTempalte

prompt_template = ChatPromptTemplate.from_messages(messages)
## now converting this prompt_template into a prompt which will be sent to large langauge model(llm)
prompt = prompt_template.invoke({"tone":"funny","first_topic":"machine learning","second_topic":"Data Science"})

response = llm.invoke(prompt)
print(response.content)