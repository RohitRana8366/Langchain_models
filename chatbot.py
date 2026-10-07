from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

while True:
    user_input = input("you: ")

    if user_input == "exit":
        break

    result = model.invoke(user_input)

    print("AI:", result.content)