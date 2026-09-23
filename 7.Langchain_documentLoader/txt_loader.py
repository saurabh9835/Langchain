from pathlib import Path

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import  StrOutputParser
from dotenv import load_dotenv

load_dotenv()

docs = [Document(page_content=Path('poem.txt').read_text(encoding='utf-8'))]

llm = HuggingFaceEndpoint(
      repo_id="google/gemma-4-31B-it",
      task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Create the summary of given \n {text}",
    input_variables=['text']
)

chain = prompt | model | parser

result = chain.invoke({'text':docs[0].page_content})

print(result)