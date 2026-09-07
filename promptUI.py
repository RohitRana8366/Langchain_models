from langchain_groq import ChatGroq
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

st.header("Research Tool")

paper = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "Generative Adversarial Networks",
        "Deep Residual Learning for Image Recognition",
        "ImageNet Classification with Deep Convolutional Neural Networks"
    ]
)

style = st.selectbox(
    "Select Style Format",
    [
        "IEEE",
        "APA",
        "MLA",
        "Chicago",
        "Harvard"
    ]
)

length = st.selectbox(
    "Select Length",
    [
        "Short (100 words)",
        "Medium (1000 words)",
        "Long (1500 words)",
        "Very Long (2000 words)",
        "Detailed (3000 words)"
    ]
)
template=PromptTemplate(
    template="""You are an expert academic research paper writer.

Your task is to write a well-structured and academically sound research paper based on the following inputs:

**Research Paper Topic/Reference:** {paper}

**Citation & Formatting Style:** {style}

**Required Length:** {length}

### Instructions:

1. Write the research paper according to the selected topic/reference.
2. Follow the **{style}** formatting and citation style consistently throughout the paper.
3. Keep the content within approximately **{length}**.
4. Use a formal, academic, and professional tone.
5. Organize the paper with appropriate headings and subheadings.
6. Include the following sections where appropriate:

   * Title
   * Abstract
   * Keywords
   * Introduction
   * Background / Literature Review
   * Methodology
   * Main Analysis / Discussion
   * Results or Findings
   * Conclusion
   * References
7. Explain technical concepts clearly while maintaining academic depth.
8. Avoid unnecessary repetition, filler content, and unsupported claims.
9. Do not fabricate research findings, statistics, citations, authors, or references.
10. If specific information is unavailable, clearly indicate the limitation rather than inventing information.
11. Ensure that citations and the reference list are consistent with the selected **{style}**.
12. Make the final output ready for academic review or further editing.

### User Inputs:

* Paper: {paper}
* Style: {style}
* Length: {length}

Now generate the complete research paper.
""",
input_variables=['paper','style','length']

)
if st.button("Submit"):
    prompt = template.invoke({
        "paper": paper,
        "style": style,
        "length": length,
    })

    with st.spinner("Generating research paper..."):
        result = model.invoke(prompt)

    st.subheader("Generated Research Paper")
    st.write(result.content)