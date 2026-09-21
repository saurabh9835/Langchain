from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableLambda,RunnablePassthrough,RunnableParallel,RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a detailed topic on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summarize the following text \n {text}",
    input_variables=['text']
)

report_chain = RunnableSequence(prompt1,model,parser)

# RunnableBranch ka basic syntax:
# RunnableBranch(
#     (condition_function, runnable_if_true),
#     default_runnable_or_passthrough
# )
#
# Intuition:
# - "agar condition true hai, to yeh runnable chalna hai"
# - "warna default runnable use hoga"
# - basically if-else logic in runnable form
#
# Example pattern:
# RunnableBranch(
#     (lambda x: condition, some_chain),
#     RunnablePassthrough()   # default branch
# )
#
# Is file mein:
# - agar input text ka length 300+ words hai, to summarize chain run hoga
# - warna input ko as-it-is pass-through kar diya jayega

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain,branch_chain)

print(final_chain.invoke({'topic':'Iran vs Israil'}))
