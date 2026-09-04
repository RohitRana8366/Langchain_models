from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

st.header("Research Tool")

user_input = st.text_input("Enter your prompt")

if st.button("Submit"):
    result = model.invoke(user_input)
    st.write(result.content)