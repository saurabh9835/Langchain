from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# ChatPromptTemplate ek prompt structure banata hai jisme system message, previous chat history,
# aur current human query sab ek saath combine hote hain.
# Yeh template ek reusable format deta hai jo model ko kaise respond karna hai, usse define karta hai.

template = ChatPromptTemplate([
    ("system","you are a helpful customer support"),
    # MessagesPlaceholder previous chat messages ko dynamically inject karta hai.
    # Isse model ko earlier conversation ka context milta hai.
    MessagesPlaceholder(variable_name="chat_history"),
    ("human","{query}")
])

# chat_history.txt se previous conversation ko read karte hain.
# Isse model ko earlier messages ka context milta hai, taaki woh continuity maintain kar sake.
chat_history =[]
with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())


print(chat_history)

# `invoke()` method template ko actual values ke saath fill karta hai, aur final prompt banata hai.
prompt = template.invoke({"chat_history":chat_history, "query":"Where is my refund"})

print(prompt)
