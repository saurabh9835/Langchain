from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

chat_template = ChatPromptTemplate([
    ("system","You are helpful {domain} assistant"),                        #create tuple
    ("human","Expalin insimple word , What is {topic}")
    #SystemMessage(content="You are helpful {domain} assistant"),
    #HumanMessage(content="Expalin insimple word , What is {topic}")        Both these don't work in Chatprompttemplate as it was working in Prompttemplate
])

result = chat_template.invoke({"domain":"Ai", "topic":"ML"})

print(result)