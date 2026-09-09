from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template = "Generate a detailed report about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = "give 5 important point from the following \n {text}",
    input_variables=['text']
)

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'Unemployment'})

print(result)

chain.get_graph().print_ascii()