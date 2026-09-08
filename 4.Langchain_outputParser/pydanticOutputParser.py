# ===== IMPORTS =====
# LangChain se Hugging Face model use karne ke liye imports
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Prompt template aur output parser ke liye imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Pydantic: Data validation ke liye library (structured data define karna)
# BaseModel: Pydantic ka base class - sabhi data models yahan se inherit hote hain
# Field: Har field ko describe karne ke liye (constraints, description, etc.)
from pydantic import BaseModel, Field

# .env file se environment variables load karta hai (API keys, tokens, etc.)
load_dotenv()

# ===== LLM CONFIGURATION =====
# Hugging Face endpoint configure karta hai - Gemma model use karega
# repo_id: Kaunsa model use karna hai (Google ka Gemma 4 31B)
# task: "text-generation" = text generate karna hai
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

# LLM ko chat model mein wrap karta hai taaki ise LangChain pipeline mein use kar saken
# ChatHuggingFace: Conversation ke liye optimized interface
model = ChatHuggingFace(llm=llm)

# ===== PYDANTIC MODEL: OUTPUT STRUCTURE DEFINE KARNA =====
# BaseModel: Pydantic ka base class jo data validation ke liye use hota hai
# Yeh class batati hai ki LLM ka output kaise FORMAT hona chahiye
class Person(BaseModel):
    # Field() - Har property ko describe karta hai (type validation + constraints)
    # description: API ko batata hai ki yeh field kya represent karta hai
    # gt=18: Greater Than constraint - age 18 se zyada hona chahiye
    
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description='Age of the Person (must be > 18)')
    city: str = Field(description='Name of the city person belongs to')

# ===== PYDANTIC OUTPUT PARSER =====
# pydantic_object=Person: Parser ko Person class give karte hain
# Parser yeh bataata hai LLM ko ki output EXACTLY Person class ke hisaab se hona chahiye
# Agar output match nahi karega to automatically parse karega/error dega
parser = PydanticOutputParser(pydantic_object=Person)

# ===== PROMPT TEMPLATE =====
# Template: Model ko kya prompt dena hai (yeh template variable le sakta hai)
# input_variables=['place']: Ye variable dynamically set hoga (indian, american, etc.)
# 
# TECHNICAL TERM: format_instruction
# parser.get_format_instructions() automatically ek detailed instruction generate karta hai
# Example output:
# "Output a JSON object with the following schema:
#  {'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer', 'gt': 18}, ...}}"
# 
# TECHNICAL TERM: partial_variables
# Ye template variables hote hain jo FIXED hote hain (runtime mein change nahi hote)
# partial_variables={'format_instruction': ...} = format instruction FIXED rahega
# sirf 'place' variable change hoga jab template.invoke() call hoga

template = PromptTemplate(
    template="Generate name, age , city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],  # Ye variable user/code se aayega
    
    # partial_variables: Ye variables automatically set hote hain (pre-populated)
    # 'format_instruction' = Parser ka generated instruction (fixed)
    # Iska fayda: har baar format instruction generate nahi karna padta
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# ===== PROMPT GENERATION =====
# template.invoke() - Template ko actual value deta hai
# place='indian' - 'place' variable ko 'indian' set karte hain
# Output: Poora prompt with format instructions
prompt = template.invoke({'place': 'indian'})

# ===== DEBUG: PROMPT DEKH SAKTE HO =====
print("=== GENERATED PROMPT ===")
print(prompt)
print("\n")

# ===== LLM SE OUTPUT LENA =====
# model.invoke(prompt) - LLM ko prompt dete hain, output generate karata hai
# Output: AIMessage object hota hai (contains: content, metadata, etc.)
result = model.invoke(prompt)

# ===== OUTPUT PARSING =====
# parser.parse(result.content) - Parser LLM ka output leta hai
# Phir JSON extract karta hai aur Person class mein convert karta hai
# Validation bhi karata hai (age > 18 check karega, data types verify karega)
final_result = parser.parse(result.content)

# ===== FINAL RESULT =====
# final_result: Python object (Person class ka instance)
# Ab iska use karte hain: final_result.name, final_result.age, etc.
print("=== PARSED RESULT ===")
print(final_result)
