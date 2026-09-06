from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

template1 = PromptTemplate(
    template="write a detailed report on topic {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template = "write a 3 line summary on following text \n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({"topic":"Black Hole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

result1 = model.invoke(prompt2)

print(result1.content)

