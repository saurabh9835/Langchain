from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# .env file se API key load karte hain taaki Gemini model use kar sake.
load_dotenv()

# LangChain me conversation ko `messages` list ke form me represent kiya jata hai.
# `SystemMessage` -> model ko instructions deta hai, jaise "aap assistant ho".
# `HumanMessage` -> user ka message.
# `AIMessage` -> model ka previous response.
messages = [
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content= "Tell me about Langchain")
]

# GoogleGenerativeAI model ko in messages ke saath invoke karte hain.
# Isse model ko full context milta hai ki kis type ka conversation chal raha hai.
model = GoogleGenerativeAI(model="gemini-3.1-flash-lite")

result = model.invoke(messages)

# Model ka answer AIMessage ke form me append karte hain, taaki next interaction me history maintain rahe.
messages.append(AIMessage(content=result.content))
print(messages)

# LangChain me built-in message history system hota hai: SystemMessage, HumanMessage, AIMessage.
# Isliye manual list append ya custom chat history maintain karne ki jagah, yeh inbuilt classes use hoti hain.
