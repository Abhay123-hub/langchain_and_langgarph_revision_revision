from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel,RunnableBranch
load_dotenv()

llm = ChatOpenAI(model = "gpt-3.5-turbo")



## i am going to make an AI application 
## this AI application will work in such a way that
## user will give any review to this ai application
## ai application first will classify that review
## based on positive,negative and neutral
## based on any of the feedback
## the application will generate the response


template = [
    ("system","You are an intelligent system who classifies the reviews based on positive ,negative or neutra land returns any "
    "one of these three"),
    ("human","given the review {review} find out is it positive,neagtive ,neutral or escalate review")
]

prompt_review = ChatPromptTemplate.from_messages(template)

## till here i have classified the review
## now based on type of review i will generate its response


positive_response = ChatPromptTemplate.from_messages(
[("system","You are a helpfull assistant"),
 ("human","Generate a response on this positive feedback {feedback} by thanking the customer ")]

)

negative_response = ChatPromptTemplate.from_messages(
    [("system","You are a helpfull assistant"),
     ("human","Generate a response on the user negative feedback {feedback}  ,say sorry and say we will not do this mistake agian")]

)

neutral_response = ChatPromptTemplate.from_messages(
    [("system","You are a helpfull assistant"),
     ("human","Given the user neutral feed back {feedback},give response and try to ask user overall experience and problems faced ")]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

## these are the conditional branches working on a particular condition
## only one branch will work at a time
conditional = RunnableBranch(
    (lambda x: "positive" in x,
     positive_response | llm | StrOutputParser()),

     (lambda x: "negative" in x,
      negative_response | llm | StrOutputParser()),
      (lambda x: "neutral" in x,
       neutral_response | llm | StrOutputParser()),
    escalate_feedback_template | llm | StrOutputParser()

     
)


final_chain = prompt_review | llm | StrOutputParser() | conditional

review1 = "It was a great experience , i really enjoyed the product"
review2 = " very disappointing product"
review3 = "Can't say anything"


response = final_chain.invoke({"review":review2})
print(response)
