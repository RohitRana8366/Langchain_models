from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="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"
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the meaning of prem?")
print(response.content)