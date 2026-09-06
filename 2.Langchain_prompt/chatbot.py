from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# .env file se API key load karte hain taaki Gemini model chal sake.
# Isse sensitive credentials ko code me hardcode nahi karna padta.
load_dotenv()

# ChatGoogleGenerativeAI ek LangChain wrapper hai jo Google Gemini model ko use karne ke liye bana hai.
# Yahan `model` ka matlab actual LLM instance hai jo prompts ko process karta hai.
# `max_output_tokens` ka matlab response me maximum kitne tokens (words/chars) aa sakte hain, usko limit karna.
model = ChatGoogleGenerativeAI(model ="gemini-3.1-flash-lite", max_output_tokens = 50)

# SystemMessage ek instruction message hai jo model ko batata hai ki kaise behave karna hai.
# HumanMessage user input ko represent karta hai, aur AIMessage model ka previous response.
# Chat history maintain karne ke liye in message objects ko list me store karte hain.
chat_history = [
    SystemMessage(content='You are a Helpful Assistant')
]

while True:
    user_input = input("You :")
    # User ka input HumanMessage ke form me append karte hain taaki conversation ka context maintain rahe.
    chat_history.append(HumanMessage(content=user_input))

    if user_input == "Exit":
        break
    # `invoke()` method model ko full chat history de deta hai, isse model previous conversation ko samajh sakta hai.
    result = model.invoke(chat_history)
    # Model ka response AIMessage me store karte hain taaki next turn me context available rahe.
    chat_history.append(AIMessage(content=result.content))
    print("Bot :",result.content)

print(chat_history)    

