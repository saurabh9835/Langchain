# ============================================================================
# LANGCHAIN CONDITIONAL CHAIN - Feedback Classifier aur Response Generator
# ============================================================================
# Yeh program feedback ko classify karta hai (positive/negative) aur appropriate
# response generate karta hai using LangChain ka RunnableBranch

# IMPORTS - Library ko import karne ke liye
# ============================================================================

from langchain_google_genai import ChatGoogleGenerativeAI  
# ChatGoogleGenerativeAI: Google ka Gemini AI model jo chat-based conversations handle karta hai

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  
# ChatHuggingFace: HuggingFace ke models ko wrap karta hai
# HuggingFaceEndpoint: API endpoint ke through HuggingFace models access karne ke liye

from dotenv import load_dotenv  
# load_dotenv: .env file se environment variables (API keys, secrets) load karta hai

from langchain_core.prompts import PromptTemplate  
# PromptTemplate: AI model ko instructions dene ke liye template banate hain
# Placeholder variables ke saath dynamic prompts create kar sakte hain

from langchain_core.output_parsers import StrOutputParser  
# StrOutputParser: AI ka output ko simple string format mein convert karta hai

from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda  
# RunnableBranch: Conditional logic ke through alag-alag paths choose karta hai (if-else jaisa)
# RunnableLambda: Custom Python function ko chain mein use karne ke liye
# RunnableParallel: Multiple tasks ko parallel mein run karta hai

from langchain_core.output_parsers import PydanticOutputParser
# PydanticOutputParser: AI output ko structured JSON format mein convert karta hai
# Pydantic ke through validation bhi hota hai

from pydantic import BaseModel, Field
# BaseModel: Data structure define karne ke liye (schema banate hain)
# Field: Schema mein fields ko describe karte hain aur validation rules dete hain

from typing import Literal
# Literal: Specific values ko define karte hain (enum jaisa)


# ============================================================================
# CONFIGURATION - Setup aur Model Initialization
# ============================================================================

load_dotenv()  
# .env file se API keys aur secrets load kar rahe hain (jaise GOOGLE_API_KEY)

# MODEL 1: HuggingFace se Gemma model load kar rahe hain
# ============================================================================
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",  # Model ka naam: Google Gemma 4 31B (instruction-tuned)
    task="text-generation"  # Task: text generation (aage text predict karna)
)
# 31B = 31 Billion parameters (model kitna bada hai, jitna bada utna accurate)
# it = instruction-tuned (instructions follow karne ke liye train kiya gaya)

# MODEL 2 & 3: Chat-based wrappers banate hain
# ============================================================================
model1 = ChatHuggingFace(llm=llm)  
# HuggingFace ke Gemma model ko chat format mein use karne ke liye wrap kiya

model2 = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
# Google ka Gemini model - fast aur lightweight version (lite = chota)

# ============================================================================
# DATA STRUCTURE - Feedback ka Schema define kar rahe hain (Pydantic Model)
# ============================================================================
class Feedback(BaseModel):
    """
    Pydantic Model: Feedback ke expected output ka structure define karta hai
    BaseModel: Validation aur type-checking automatic karta hai
    """
    sentiment: Literal['postive','negative'] = Field(
        description='Give the sentiment of the feedback'
        # Literal: sirf ye 2 options possible hain - 'positive' ya 'negative'
        # NOTE: 'postive' mein typo hai - 'positive' hona chahiye!
    )



# ============================================================================
# OUTPUT PARSERS - AI ke output ko parse karne ke liye
# ============================================================================

parser1 = PydanticOutputParser(pydantic_object=Feedback)
# Pydantic Parser: AI ka output ko Feedback schema mein convert karta hai
# Output ko JSON mein validate aur structure karta hai
# Use Case: Sentiment classification ke liye (structured output chahiye)

parser2 = StrOutputParser()
# String Parser: AI ka output ko simple string mein convert karta hai
# Koi validation nahi, bas raw text return karta hai
# Use Case: Response generation ke liye (narrative text chahiye)

# ============================================================================
# STEP 1: CLASSIFIER CHAIN - Sentiment classify karne ke liye
# ============================================================================

prompt1 = PromptTemplate(
    template='classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    # Template: AI ko instructions deta hai kya karna hai
    # {feedback}: placeholder - actual feedback text yahan replace hoga
    # {format_instruction}: parser1 se JSON format instructions
    
    input_variables=['feedback'],
    # input_variables: chain ke through aane wali inputs
    
    partial_variables={
        'format_instruction': parser1.get_format_instructions()
        # Automatically parser1 ke instructions add karta hai template mein
        # Parser1 ko JSON format de karte hain
    }
)

classifier_chain = prompt1 | model2 | parser1
# PIPE (|) operator: LangChain mein chain banane ka tarika
# prompt1: Template prepare karta hai
#   ↓
# model2 (Gemini): Template ko AI process karta hai
#   ↓
# parser1: Output ko Feedback object mein convert karta hai (sentiment milega)



# ============================================================================
# STEP 2: RESPONSE TEMPLATES - Positive aur Negative feedback ke liye
# ============================================================================

prompt2 = PromptTemplate(
    template='Write appropriate response to the positive feedback \n {feedback}',
    # Template: Positive feedback pe reply likha dega
    
    input_variables=['feedback']
    # Input: feedback ka text
)

prompt3 = PromptTemplate(
    template='Write appropriate response to the negative feedback \n {feedback}',
    # Template: Negative feedback pe reply likha dega
    
    input_variables=['feedback']
    # Input: feedback ka text
)

# ============================================================================
# STEP 3: BRANCH CHAIN - Conditional Logic (if-else) 
# ============================================================================
# RunnableBranch: Sentiment ke base par alag-alag path execute karta hai
# Jaisa switch-case statement ho ya if-elif-else

branch_chain = RunnableBranch(
    # First Condition: Agar sentiment 'positive' hai to yeh chain run hogi
    (lambda x: x.sentiment == 'positive', prompt2 | model2 | parser2),
    # lambda x: x.sentiment == 'positive' -> Condition check karta hai
    # prompt2 | model2 | parser2 -> Positive template -> AI se response -> String output
    
    # Second Condition: Agar sentiment 'negative' hai to yeh chain run hogi
    (lambda x: x.sentiment == 'negative', prompt3 | model2 | parser2),
    # lambda x: x.sentiment == 'negative' -> Condition check karta hai
    # prompt3 | model2 | parser2 -> Negative template -> AI se response -> String output
    
    # Default Case: Agar koi condition match nahi hue (fallback)
    RunnableLambda(lambda x: "Could not find answer")
    # RunnableLambda: Custom Python function run karta hai
    # Default message return karega agar sentiment unclear hai
)

# ============================================================================
# FINAL CHAIN - Pura Flow ek saath
# ============================================================================
# classifier_chain: Feedback se sentiment predict karta hai
#     ↓
# branch_chain: Sentiment ke base par appropriate response generate karta hai
#     ↓
# parser2: Output ko string mein convert karta hai

chain = classifier_chain | branch_chain | parser2

# ============================================================================
# EXECUTION - Test karte hain
# ============================================================================
print(chain.invoke({'feedback': 'this phone is terrible'}))
# Input: 'this phone is terrible' (negative feedback)
# Expected Flow:
#   1. Classifier: sentiment = 'negative'
#   2. Branch: Negative condition match, prompt3 run hoga
#   3. Response: AI negative feedback pe professional response likha dega
#   4. Output: String format mein print hoga

