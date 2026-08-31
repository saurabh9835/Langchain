from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content= "Tell me about Langchain")
]

model = GoogleGenerativeAI(model="gemini-3.1-flash-lite")

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)

#Langchain has inbuilt messages history system in name of SystemMessages,HumanMessages and AIMessage
#so instead of doing manual list append as chat history we use these fn 