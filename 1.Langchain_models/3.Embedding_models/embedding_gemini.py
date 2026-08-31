from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model ="gemini-embedding-2-preview", dimension=20)

result = embedding.embed_query("Delhi is capital of India")

print(str(result))