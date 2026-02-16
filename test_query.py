import chromadb


from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import (
    MetadataFilters,
    ExactMatchFilter,
)
from dotenv import load_dotenv


load_dotenv()


Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# Load from ChromaDB
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("fridge_manuals")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)



def run_brand_queries(brand: str, queries):
    print(f"\n==============================")
    print(f" BRAND: {brand}")
    print(f"==============================\n")

    # Build metadata filter: brand must equal this value
    brand_filter = MetadataFilters(
        filters=[ExactMatchFilter(key="brand", value=brand)]
    )

    # System prompt matching app.py
    qa_prompt_str = """You are a technical assistant. Answer using ONLY the context below.

CRITICAL: If the context doesn't contain relevant information, respond EXACTLY: "The manual does not provide information about this topic. This may require contacting customer service or a certified technician."

Context:
{context_str}

Question: {query_str}
Answer:"""

    qa_prompt_tmpl = PromptTemplate(qa_prompt_str)

    # Create a query engine that filters by brand metadata
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_mode="compact",
        filters=brand_filter,
        text_qa_template=qa_prompt_tmpl,
    )

    for q in queries:
        print(f"Q ({brand}): {q}")
        response = query_engine.query(q)
        print(f"A: {response}\n")

        print("SOURCES USED:")
        print("=" * 80)
        for idx, node in enumerate(response.source_nodes, 1):
            print(f"\n[Source {idx}]")
            print(f"Relevance Score: {node.score:.3f}")
            print(f"Text: {node.text[:200]}...")
            if node.metadata:
                print(f"Metadata: {node.metadata}")
        print("\n" + "-" * 80 + "\n")



if __name__ == "__main__":
    print("Testing brand-aware RAG system...\n")

    queries = [
        "How do I defrost the refrigerator?",
        "What temperature should I set for the freezer?",
        "How do I clean the water filter?",
    ]

    # Test for each brand separately
    for b in ["Samsung", "Whirlpool", "LG"]:
        run_brand_queries(b, queries)
