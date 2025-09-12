from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import pycountry
import streamlit as st
from langchain_core.prompts import load_prompt

countries = [country.name for country in pycountry.countries]
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

st.header("Get info about a country")

country = st.selectbox( "Select Country Name", countries )
perspective = st.selectbox( "Choose your Perspective to judge a country", ["Geographical Perspective","Historical Perspective","Political & Governance Perspective","Economic Perspective","Technological & Innovation Perspective","Cultural & Social Perspective","Educational & Scientific Perspective","Military & Strategic Perspective","Environmental & Natural Resource Perspective","Global Relations & Diplomatic Perspective"] )
mode = st.selectbox("Choose the text mode", ["Paragraph Mode","List Mode","Numbered Mode (Steps / Ranking)","Extensive Sentence Mode","Tabular Mode","Diagram / Mindmap Mode"])

template = load_prompt('prompts/template.json')
prompt = template.invoke({
    "country":country,
    "perspective":perspective,
    "mode":mode
})

if st.button('Send'):
    result = model.invoke(prompt)
    st.write(result.content)