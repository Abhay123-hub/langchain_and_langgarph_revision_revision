## here i am going to build a react agent using langchain frmaework
## the react agent works on this principle
## thought->action->observation
## then again the loop will continue based on the requirement

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults,tool
from langchain.agents import Tool,initialize_agent
import datetime

## activating all the secret varaibles here in my python key
load_dotenv()
## first of all test the llm model ,then i will move further

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
llm = ChatOpenAI(model = "gpt-3.5-turbo")
# result = llm.invoke("Tell me facts about machine learning and data science")

# print(result.content)
search_tool = TavilySearchResults(search_depth = 'basic')
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


## now building the react agent
tools = [search_tool,get_system_time]

agent = initialize_agent(tools=tools,llm=llm,agent = 'zero-shot-react-description',verbose = True)

## let us now compare the output of both only llm call vs using react agent with llm call
## by providing the same input we will see the difference in the output
##and will try to understand the internal working of both llm call vs recat agent with llm call

question1 = "tell me today weather in Delhi,tell me in funny way"
question2 = "when was Chandrayan-3 launched and how much days before today"
# response_llm = llm.invoke(question)
# print("The response of plain llm is:\n",response_llm.content)
agent.invoke(question2)

