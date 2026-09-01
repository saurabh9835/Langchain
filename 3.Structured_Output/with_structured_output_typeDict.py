from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()
model = ChatGoogleGenerativeAI(model ="gemini-3.1-flash-lite", max_output_tokens = 30)

#Creating data Schema

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("This mobile offers a sleek design, smooth performance, and a vibrant display that enhances everyday use. The battery life easily lasts a full day, and the camera captures clear, detailed photos in most lighting conditions. Overall, it provides excellent value for money and is a reliable choice for daily tasks.")

print(result["sentiment"])
