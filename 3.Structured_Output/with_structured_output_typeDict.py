#Structured Output Without Validation
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional

# .env file se API key load karte hain
# Isse hum environment variable se Google API key ko access kar sakte hain
load_dotenv()

# Gemini model ko initialize karte hain
# Yeh ek chat model hai jo text ko input leta hai aur output deta hai
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# Structured output ka matlab hai ki hum model se ek fixed format me data expect karte hain
# Example: hum chahte hain ki output me summary, sentiment, aur key_theme ho
# Isse code ko predictably data handle karna easy ho jata hai

# TypedDict ka use hota hai ek custom dictionary structure define karne ke liye
# Matlab: hum bata rahe hain ki dictionary me konsi keys honi chahiye aur unka type kya hoga lekin ye #runtime pe data type ko validate nhi karta h means ham bata rhe h ki output yesa chahiye lekin #jaruri nhi str ya int ya defined data type hi mile llm int ke jagah str bhi de skta h
# Jaise Review me summary ek string hona chahiye, sentiment bhi string, key_theme list of strings hona chahiye
class Review(TypedDict):
    key_theme: Annotated[ list[str],
        "Review me discuss hue sabse important themes ko list format likho."
    ]
    summary: Annotated[
        str,
        "Review ka short aur clear summary do."
    ]
    sentiment: Annotated[
        str,
        "Sentiment ko positive, negative ya neutral me batao."
    ]
    pros: Annotated[
        Optional[list[str]],
        "Write down all pros inside list"
        ]
    cons: Annotated[
        Optional[list[str]],
        "Write down all cons inside list"
        ]
    name: Annotated[
        str,
        "Write name of reviewer"
        ]

# Annotated ka use hota hai field ke saath extra instruction dene ke liye
# Yahan hum model ko clear direction de rahe hain ki har field kya return karna chahiye
# Matlab: summary ko kaisa hona chahiye, sentiment kya values accept karni chahiye, etc.

# with_structured_output() method model ko ye bata deta hai ki output ek specific schema ke according aana chahiye
# Isse JSON-like structured response milta hai
structured_model = model.with_structured_output(Review)

# Input text ko model ke paas bhejte hain
# Hum ek product review de rahe hain, aur model se structured response nikalna chahte hain
result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Saurabh""")

# Final output print karte hain
# Output ek dictionary jaisa structure me aayega, jisme keys summary, sentiment, aur key_theme honge
print(result)
