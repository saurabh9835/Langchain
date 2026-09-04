from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal,Optional

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

class Review(BaseModel):
    key_theme: list[str] = Field(description= "Review me discuss hue sabse important themes ko list format likho.")
    summary: str = Field(description="Review ka short aur clear summary do.")
    sentiment: Literal["pos","neg"] = Field(description="Sentiment ko positive, negative ya neutral me batao.")
    pros: Optional[list[str]] = Field(default=None,description="")
