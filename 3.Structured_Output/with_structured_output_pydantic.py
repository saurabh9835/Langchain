from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal,Optional

# Pydantic aur TypedDict dono ka kaam schema/structure define karna hai,
# lekin unka purpose thoda different hota hai.
#
# 1) Pydantic ka use isliye hota hai kyunki hum LLM se structured output expect karte hain.
#    Model output ko BaseModel ke andar validate, parse, aur type-safe bana diya jata hai.
#    Isse aapko output me missing fields, wrong types, ya invalid values ka issue immediately milta hai.
#    Yeh LangChain me with_structured_output(...) ke saath kaafi useful hota hai.
#
# 2) TypedDict ek static type annotation hai. Yeh runtime validation nahi karta.
#    Sirf code editor aur type checker ko help karta hai ki object kis structure ka hona chahiye.
#    Example: dict ke keys aur value types define kar sakta hai, lekin actual data ko verify nahi karta.
#
# 3) Agar aapko LLM se data parse karna hai aur guarantee chahiye ki output valid schema ke hisaab se aaye,
#    toh Pydantic best choice hai. TypedDict sirf typing ke liye useful hai, validation ke liye nahi.

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite")

class Review(BaseModel):
    key_theme: list[str] = Field(description= "Review me discuss hue sabse important themes ko list format likho.")
    summary: str = Field(description="Review ka short aur clear summary do.")
    sentiment: Literal["pos","neg"] = Field(description="Sentiment ko positive, negative ya neutral me batao.")
    pros: Optional[list[str]] = Field(default=None,description="Write down all pros inside list")
    cons: Optional[list[str]] = Field(default=None,description="Write down all cons inside list")
    name: Optional[str] = Field(default=None,description="Write name of reviewer")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Saurabh""")

    
print(result)
     