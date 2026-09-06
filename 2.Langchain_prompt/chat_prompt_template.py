from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

# ChatPromptTemplate ka use hota hai prompt ko structured chat format me banana.
# Yahan system message se model ko instruction milti hai, aur human message se user ka question aata hai.
# `{domain}` aur `{topic}` placeholders hain, jo later runtime par actual values se replace hote hain.
chat_template = ChatPromptTemplate([
    ("system","You are helpful {domain} assistant"),                        # create tuple
    ("human","Expalin insimple word , What is {topic}")
    # SystemMessage(content="You are helpful {domain} assistant"),
    # HumanMessage(content="Expalin insimple word , What is {topic}")
    # In comments me ye note hai ki ChatPromptTemplate me yeh direct object form kaam nahi karta,
    # kyunki yahan tuple-based structure use hoti hai.
])

# `invoke()` method template ko actual values ke saath fill karta hai.
# Isse final formatted prompt ready hota hai jise model ko pass kiya ja sakta hai.
result = chat_template.invoke({"domain":"Ai", "topic":"ML"})

print(result)