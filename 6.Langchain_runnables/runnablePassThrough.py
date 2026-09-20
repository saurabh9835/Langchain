from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a joke about -> {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a explaination about -> {topic}",
    input_variables=['topic']
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'Explanation': RunnableSequence(prompt2,model,parser)

})

final_chain = RunnableSequence(joke_gen_chain,parallel_chain)

result = final_chain.invoke({'topic':'football'})

print(result)

final_chain.get_graph().print_ascii()