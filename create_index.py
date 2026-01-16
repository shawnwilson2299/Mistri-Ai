import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

# Use OpenAI's latest embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Read your parsed manual
with open('parsed_manual.md', 'r', encoding='utf-8') as f:
    text = f.read()

documents = [Document(text=text)]

# Create ChromaDB storage
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("samsung_manual")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index (calls OpenAI API to generate embeddings)
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context
)

print("✅ Index created with OpenAI embeddings!")
print(f"✅ Vector database saved to ./chroma_db/")
