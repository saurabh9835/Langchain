from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# .env file se environment variables, jaise Hugging Face API token, load karta hai.
load_dotenv()

# Text-generation model ko configure karta hai jo character ki details generate karega.
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

# Endpoint ko chat model mein wrap karta hai taaki ise LangChain pipeline mein use kar saken.
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = 'fact_1',description='fact_1 about topic'),
    ResponseSchema(name = 'fact_2',description='fact_2 about topic'),
    ResponseSchema(name = 'fact_3',description='fact_3 about topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template =  PromptTemplate(
        template = "Give me 3 fact about {topic}\n {format_instruction}",
        input_variables=['topic'],
        partial_variables={'format_instruction': parser.get_format_instruction()}
    )
prompt = template.invoke({'topic':'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)





