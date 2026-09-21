# 1) Importing required libraries
# GoogleGenerativeAIEmbeddings: ek model hai jo text ko vector (numbers) mein convert karta hai.
# Embedding = text ko machine-friendly numerical form mein convert karna.
# Vector = ek array/sequence of numbers jo text ka meaning represent karta hai.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load_dotenv(): .env file se environment variables load karta hai, jaise API key.
from dotenv import load_dotenv

# cosine_similarity: do vectors ke beech ka angle compare karke similarity nikalta hai.
# Agar vectors same direction mein hain, similarity high hoti hai.
from sklearn.metrics.pairwise import cosine_similarity

# numpy: array operations ke liye use hota hai; yaha vector handling ke kaam aata hai.
import numpy as np

# .env file se API keys aur settings load karo.
load_dotenv()

# 2) Create embedding model
# model = Gemini embedding model jo text ko 200-dimensions ke vector mein convert karega.
# dimension = vector ki length; jitna bada, utna detailed meaning ya representation.
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", dimension=200)

# 3) Documents list
# Yeh woh sentences hain jinhe hum compare karna chahte hain.
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# 4) User query
# Query = user ka actual question/keyword jo search ya match karna hai.
ques = "Who is virat kohli?"

# 5) Convert documents to embeddings
# embed_documents(): multiple texts ko ek baar mein vectors mein convert karta hai.
# Har document ek unique vector ban jata hai.
doc_embed = embedding.embed_documents(documents)

# 6) Convert query to embedding
# embed_query(): ek single question ko vector mein convert karta hai.
# Isse hum document vectors ke sath compare kar sakte hain.
ques_embed = embedding.embed_query(ques)

# 7) Compute similarity score
# cosine_similarity([ques_embed], doc_embed)[0] => question vector aur har document vector ke beech score nikalta hai.
# Agar score high hai, matlab question aur document closely related hain.
score = cosine_similarity([ques_embed], doc_embed)[0]

# 8) Pick the best matching document
# enumerate(score): (index, similarity_score) pairs bana deta hai.
# sorted(..., key=lambda x: x[1]) => similarity score ke according ascending order mein sort karta hai.
# [-1] => sabse highest similarity score wala element nikalta hai.
# index = most relevant document ka index
# score = highest similarity score
index, score = sorted(list(enumerate(score)), key=lambda x: x[1])[-1]

# 9) Print result
print(ques)
print(documents[index])
print("Similarity_Score :", score)

# Simple explanation:
# - Embedding = text ko numbers mein convert karna
# - Vector = numbers ka representation
# - cosine_similarity = similarity ka measurement
# - query = user ka question
# - doc_embed = document vectors
# - ques_embed = question vector
# - index = best matched document ka position


