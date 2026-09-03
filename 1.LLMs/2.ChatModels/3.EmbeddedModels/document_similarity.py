from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
embeddings=HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2"
)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

documents = [
    """Artificial Intelligence enables machines to perform tasks that usually require human intelligence.
    Machine learning is a subset of AI that learns from data.
    Deep learning uses neural networks with multiple layers.
    AI is widely used in healthcare, finance, and robotics.
    Many companies invest heavily in AI research and development.""",

    """Python is one of the most popular programming languages.
    It is known for its simple and readable syntax.
    Python is widely used in data science and web development.
    Libraries such as NumPy and Pandas make data analysis easier.
    Beginners often choose Python as their first programming language.""",

    """Cricket is a bat-and-ball sport played between two teams.
    It is especially popular in India, Australia, and England.
    Players score runs by hitting the ball and running between wickets.
    International tournaments attract millions of viewers.
    Cricket requires teamwork, strategy, and skill.""",

    """Space exploration helps scientists understand the universe.
    Satellites are used for communication and navigation.
    Space agencies launch missions to study planets and stars.
    Modern telescopes capture images of distant galaxies.
    Research in space technology continues to advance rapidly.""",

    """Healthy living involves regular exercise and balanced nutrition.
    Fruits and vegetables provide important vitamins and minerals.
    Drinking enough water supports body functions.
    Physical activity improves strength and mental well-being.
    A healthy lifestyle reduces the risk of many diseases.""",

    """Renewable energy comes from naturally replenished sources.
    Solar panels convert sunlight into electricity.
    Wind turbines generate power using moving air.
    Renewable energy reduces dependence on fossil fuels.
    Many countries are investing in sustainable energy solutions.""",

    """Cybersecurity protects systems, networks, and data from attacks.
    Strong passwords help prevent unauthorized access.
    Encryption secures sensitive information during transmission.
    Organizations use firewalls and monitoring tools for protection.
    Cyber threats continue to evolve with technology.""",

    """Online education has become increasingly popular in recent years.
    Students can access courses from anywhere in the world.
    Video lectures and interactive assignments improve learning.
    Digital platforms make education more accessible.
    Technology plays a major role in modern learning environments."""
]

query=input("enter your query=>")

doc_embeddings=embeddings.embed_documents(documents)
query_embeddings=embeddings.embed_query(query)

scores=cosine_similarity([query_embeddings],doc_embeddings)

print("this is cosine similarity vector's over ducument=>\n",scores)
best_doc_index=np.argmax(scores)
print("this is best document index which is related to our query=>",best_doc_index)
print("print that related document by the help of that's query=>\n",documents[best_doc_index])
print("similarity score in percentage:",scores[0][best_doc_index]*100)

prompt = f"""
Answer the question using the context.

Context:
{documents[best_doc_index]}

Question:
{query}
"""

response = model.invoke(prompt)

print(response.content)