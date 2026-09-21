from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite" , )

parser = StrOutputParser()

def word_count(text):
    return len(text.split())


prompt1 = PromptTemplate(
    template="Generate a tweet about -> {topic}",
    input_variables=['topic']
)

joke_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_chain,parallel_chain)
result = final_chain.invoke({'topic':'AI'})

print(result[word_count])
