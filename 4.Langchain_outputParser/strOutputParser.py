from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"Black Hole"})

print(result)