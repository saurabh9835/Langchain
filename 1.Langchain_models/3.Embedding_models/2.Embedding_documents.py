from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", dimension=20)

documents = [
    "What is capital of India",
    "What color crow is"
    "who is first prime minister of India"
]

result = embeddings.embed_documents(documents)
print(result)