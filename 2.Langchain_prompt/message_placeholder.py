from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

#Template

template = ChatPromptTemplate([
    ("system","you are a helpful customer support"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{query}")
])

#load chat history
chat_history =[]
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())


print(chat_history)

prompt = template.invoke({"chat_history":chat_history, "query":"Where is my refund"})

print(prompt)
