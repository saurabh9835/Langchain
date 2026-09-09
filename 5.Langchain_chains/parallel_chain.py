# Zaruri libraries import kar rahe hain
from langchain_google_genai import ChatGoogleGenerativeAI  # Google ka AI model
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # HuggingFace models
from dotenv import load_dotenv  # Environment variables load karne ke liye
from langchain_core.prompts import PromptTemplate  # Prompt banane ke liye
from langchain_core.output_parsers import StrOutputParser  # Output ko string mein convert karne ke liye
from langchain_core.runnables import RunnableParallel  # Parallel mein execute karne ke liye

load_dotenv()  # Environment variables ko load kar rahe hain

# HuggingFace se Gemma model load kar rahe hain
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation"
)

# Pehla model - HuggingFace ka Gemma
model1 = ChatHuggingFace(llm=llm)
# Doosra model - Google ka Gemini
model2 = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

# Pehla prompt - notes banane ke liye
prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

# Doosra prompt - questions aur answers ke liye
prompt2 = PromptTemplate(
    template = "give 5 question answer from the following text \n {text}",
    input_variables=['text']
)

# Teesra prompt - notes aur quiz ko merge karne ke liye
prompt3 = PromptTemplate(
    template= "merge the following notes and quiz in a single notes \n notes -> {notes} and quiz -> {quiz}",
    input_variables=['notes','text']
)

# Output ko string format mein convert karne ke liye parser
parser = StrOutputParser()

# Pehli chain - prompt1 -> model1 -> parser (notes generate karegi)
notes_chain = prompt1 | model1 | parser

# Doosri chain - prompt2 -> model2 -> parser (questions generate karega)
quiz_chain = prompt2 | model2 | parser

# Dono chains ko PARALLEL mein run kar rahe hain - ek saath execute hongi!
parallel_chain = RunnableParallel({
    'notes': notes_chain,  # Notes chain
    'quiz' : quiz_chain    # Quiz chain
})

# Merge chain - notes aur quiz ko combine karne ke liye
merge_chain = prompt3 | model2 | parser

# Final chain - pehle parallel mein dono chalega, fir merge hoga
chain = parallel_chain | merge_chain

# Input text jo process karna hai - ye text notes aur questions dono banayega
text = '''Linear Regression is a fundamental supervised machine learning and statistical algorithm used to model the relationship between a dependent variable (target) and one or more independent variables (predictors). The core objective is to fit a linear equation to observed data to explain how changes in the predictors impact the target, allowing you to predict continuous numerical outcomes.Simple Linear Regression: Models the relationship using a single independent variable.Multiple Linear Regression: Models the relationship using two or more independent variables.🧮 Mathematical Formulation1. The Regression EquationFor a dataset with n independent features, the true relationship in the population is expressed as:\(y=\beta _{0}+\beta _{1}x_{1}+\beta _{2}x_{2}+\dots +\beta _{n}x_{n}+\epsilon \)When building a model from sample data, we estimate this relationship to compute a predicted value (ŷ):\(\^{y}=\^{\beta }_{0}+\^{\beta }_{1}x_{1}+\^{\beta }_{2}x_{2}+\dots +\^{\beta }_{n}x_{n}\)Where:ŷ (Target / Dependent Variable): The continuous value you want to predict (e.g., house price).\(x_1, x_2, \dots, x_n\) (Predictors / Independent Variables): The input features used for the prediction (e.g., square footage, number of bedrooms).β̂₀ (Intercept): The value of ŷ when all x variables are equal to zero. Geometrically, this is where the regression line crosses the y-axis.\(\hat{\beta}_1, \hat{\beta}_2, \dots, \hat{\beta}_n\) (Coefficients / Slopes): The weights assigned to each feature. A coefficient represents the expected change in ŷ for a one-unit change in its corresponding x variable, assuming all other variables are held constant.ε (Error Term / Residual): The random noise or variation in y that the linear model cannot explain. For a single data point i, the residual is \(e_i = y_i - \hat{y}_i\).'''

# Chain ko execute kar rahe hain - parallel mein notes aur quiz generate honge, fir merge hoga
result = chain.invoke({'text':text})

# Final result ko print kar rahe hain
print(result)

# Chain ka flow diagram print kar rahe hain - kiski order mein execute ho raha hai
chain.get_graph().print_ascii()


