from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(model ="gemini-3.1-flash-lite", max_output_tokens = 50)

chat_history = [
    SystemMessage(content='You are a Helpful Assistant')
]

while True:
    user_input = input("You :")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "Exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot :",result.content)

print(chat_history)    

