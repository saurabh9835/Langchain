from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

st.header("Research Summarizer")

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation length",["Short (1-2 paragraphs)","Medium (3-5 paragraphs)","Large (detailed explanation)"])

template = load_prompt('package.json')

#Fill the placeholders
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
if st.button("summarize"):
    result = model.invoke(prompt)
    st.write(result.content)