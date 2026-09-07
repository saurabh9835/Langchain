from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# .env file se environment variables, jaise Hugging Face API token, load karta hai.
load_dotenv()

# Text-generation model ko configure karta hai jo character ki details generate karega.
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

# Endpoint ko chat model mein wrap karta hai taaki ise LangChain pipeline mein use kar saken.
model = ChatHuggingFace(llm=llm)

# Model ke JSON response ko Python object, usually dict, mein convert karta hai.
parser = JsonOutputParser()

# Parser ki instructions prompt mein add karta hai taaki model ko expected JSON format pata ho.
template = PromptTemplate(
    template= "Give me name,age and city of a ficional character \n {format_instruction}",
    # Prompt render karte waqt caller ko koi runtime value dene ki zaroorat nahi hai.
    input_variables=[],
    # Aisi value store karta hai jo prompt use hone par automatically insert ho jaati hai.
    # Fixed helper text, jaise JSON formatting rules, ke liye yeh useful hota hai.
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Model ko bhejne se pehle prompt render karta hai. partial_variables
# {format_instruction} placeholder ko fill karta hai, isliye format() ko argument nahi chahiye.
prompt = template.format()

# Pipe operator prompt, model aur parser ko ek runnable chain mein connect karta hai.
chain = template | model | parser
# invoke({}) chain ko start karta hai. Empty dictionary kaafi hai kyunki
# runtime par dene ke liye koi input_variables nahi hain.
final_result = chain.invoke({})

# Parsed value aur uska Python type print karta hai, taaki confirm ho sake ki parser ne
# model ka raw text nahi, balki Python object (normally dict) return kiya hai.
print(final_result)
print(type(final_result))
