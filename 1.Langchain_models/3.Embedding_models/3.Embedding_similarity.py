from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", dimension=200)
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

ques = "Who is virat kohli?"

doc_embed = embedding.embed_documents(documents)
ques_embed = embedding.embed_query(ques)

score=cosine_similarity([ques_embed],doc_embed)[0]
index,score = sorted(list(enumerate(score)), key= lambda x:x[1])[-1]

print(ques)
print(documents[index])
print("Similarity_Score :",score)


