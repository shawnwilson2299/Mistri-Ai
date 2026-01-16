import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

# Set same embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load from ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("samsung_manual")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# Create query engine
query_engine = index.as_query_engine()

# Test queries
print("Testing RAG system...\n")

queries = [
    "How do I defrost the Samsung refrigerator?",
    "What temperature should I set for the freezer?",
    "How do I clean the water filter?"
]

for q in queries:
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}\n")
    print("-" * 80 + "\n")
