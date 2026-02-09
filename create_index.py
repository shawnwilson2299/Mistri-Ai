import chromadb

from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv

load_dotenv()

# 1. Set embedding model (same as before)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 2. Describe each parsed manual (brand, model, etc.)
MANUALS = [
    {
        "parsed_file": "parsed_samsung.md",
        "brand": "Samsung",
        "model_code": "DA68-04823J-00",
        "manual_type": "User manual",
        "source_filename": "samsung_manual.pdf",
    },
    {
        "parsed_file": "parsed_whirlpool.md",
        "brand": "Whirlpool",
        "model_code": "W11468670D",
        "manual_type": "Owner's Manual",
        "source_filename": "whirlpool_manual.pdf",
    },
    {
        "parsed_file": "parsed_lg.md",
        "brand": "LG",
        "model_code": "LPXS30866D",
        "manual_type": "Owner's Manual",
        "source_filename": "lg_manual.pdf",
    },
]

def load_chunks_from_file(parsed_path: str):
    """
    Very simple chunker:
    - We already have '---' as separators from LlamaParse.
    - Split the file by '---' and treat each part as one chunk.
    """
    with open(parsed_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    raw_chunks = full_text.split("\n---\n")
    # Clean and drop empty chunks
    chunks = [c.strip() for c in raw_chunks if c.strip()]
    return chunks

# 3. Build a list of Documents with metadata
all_documents = []

for manual in MANUALS:
    chunks = load_chunks_from_file(manual["parsed_file"])
    print(f"📄 {manual['parsed_file']}: {len(chunks)} chunks")

    for idx, chunk_text in enumerate(chunks):
        metadata = {
            "brand": manual["brand"],
            "model_code": manual["model_code"],
            "manual_type": manual["manual_type"],
            "source_filename": manual["source_filename"],
            # optional: a simple chunk index for debugging
            "chunk_index": idx,
        }
        doc = Document(text=chunk_text, metadata=metadata)
        all_documents.append(doc)

print(f"Total chunks/documents to index: {len(all_documents)}")

# 4. Create ChromaDB storage (we'll use a single collection for all manuals)
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("fridge_manuals")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. Create index (this will embed all chunks using OpenAI)
index = VectorStoreIndex.from_documents(
    all_documents,
    storage_context=storage_context,
)

print("✅ Index created with OpenAI embeddings!")
print("✅ Vector database saved to ./chroma_db/")
