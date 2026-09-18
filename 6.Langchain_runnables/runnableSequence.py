from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= "Generate a joke about -> {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Explain the {text}",
    input_variables=['text']
)

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({"topic": "computer"})

print(result)
