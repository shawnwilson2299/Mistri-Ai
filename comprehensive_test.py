import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Configure embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Load index
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("fridge_manuals")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

print("=" * 100)
print("MISTRI AI - COMPREHENSIVE TEST SUITE")
print("=" * 100)
print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)
print()

# Query function matching app.py
def query_manual(brand: str, question: str):
    """Query the manual with brand filtering"""
    brand_filter = MetadataFilters(
        filters=[ExactMatchFilter(key="brand", value=brand)]
    )
    
    qa_prompt_str = """You are a technical assistant. Answer using ONLY the context below.

CRITICAL: If the context doesn't contain relevant information, respond EXACTLY: "The manual does not provide information about this topic. This may require contacting customer service or a certified technician."

Context:
{context_str}

Question: {query_str}
Answer:"""

    qa_prompt_tmpl = PromptTemplate(qa_prompt_str)
    
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        response_mode="compact",
        filters=brand_filter,
        text_qa_template=qa_prompt_tmpl,
    )
    
    response = query_engine.query(question)
    return response

# Check if response is weak
def is_weak_response(response_text: str) -> bool:
    """Detect if response lacks info"""
    weak_indicators = [
        "does not provide",
        "doesn't provide",
        "not provide information",
        "no information",
        "contact",
        "service center",
        "service centre",
        "certified technician"
    ]
    
    response_lower = response_text.lower()
    
    if len(response_text.strip()) < 30:
        return True
    
    if any(indicator in response_lower for indicator in weak_indicators):
        return True
    
    return False

# Test cases organized by category
TEST_CASES = {
    "Component-Specific Queries": [
        ("Samsung", "How do I clean the water filter?"),
        ("Samsung", "How do I replace the ice maker?"),
        ("Whirlpool", "How do I install a new water filter?"),
        ("Whirlpool", "How often should I replace the air filter?"),
        ("LG", "How to change the deodorizer filter?"),
    ],
    
    "Terminology Variations": [
        ("Samsung", "How do I fix a broken water dispenser?"),
        ("Whirlpool", "My ice maker is not working, how to repair it?"),
        ("LG", "How do I clean the condenser coils?"),
    ],
    
    "Temperature & Settings": [
        ("Samsung", "What is the ideal temperature setting?"),
        ("Whirlpool", "What temperature should I set for the freezer?"),
        ("LG", "How do I adjust the refrigerator temperature?"),
    ],
    
    "Maintenance Procedures": [
        ("Samsung", "How do I defrost the freezer?"),
        ("Whirlpool", "How do I clean the door seal?"),
        ("LG", "How often should I clean the water filter housing?"),
    ],
    
    "Troubleshooting": [
        ("Samsung", "My refrigerator is not cooling properly"),
        ("Whirlpool", "Water dispenser is leaking"),
        ("LG", "Strange noise coming from compressor"),
    ],
    
    "Edge Cases": [
        ("Samsung", "How do I fix everything?"),
        ("Whirlpool", "Something is broken"),
        ("LG", "What's the warranty period?"),
    ],
    
    "Cross-Brand Consistency Test": [
        ("Samsung", "How do I replace the water filter?"),
        ("Whirlpool", "How do I replace the water filter?"),
        ("LG", "How do I replace the water filter?"),
    ],
}

# Run all tests
test_number = 0
total_tests = sum(len(queries) for queries in TEST_CASES.values())
weak_responses_count = 0
successful_responses_count = 0

for category, test_queries in TEST_CASES.items():
    print("\n" + "=" * 100)
    print(f"CATEGORY: {category}")
    print("=" * 100)
    
    for brand, question in test_queries:
        test_number += 1
        print(f"\n[TEST {test_number}/{total_tests}]")
        print(f"Brand: {brand}")
        print(f"Query: {question}")
        print("-" * 100)
        
        try:
            response = query_manual(brand, question)
            response_text = response.response
            
            # Check if weak
            is_weak = is_weak_response(response_text)
            
            print(f"Response: {response_text}")
            print(f"\nResponse Length: {len(response_text)} chars")
            print(f"Weak Response (Should Trigger Tier 3): {'YES ⚠️' if is_weak else 'NO ✅'}")
            
            if is_weak:
                weak_responses_count += 1
            else:
                successful_responses_count += 1
            
            # Show top 3 sources
            print(f"\nTop 3 Sources Retrieved:")
            for idx, node in enumerate(response.source_nodes[:3], 1):
                print(f"  [{idx}] Relevance: {node.score:.3f} | Chunk: {node.metadata.get('chunk_index', 'N/A')}")
                print(f"      Preview: {node.text[:150]}...")
            
            print("-" * 100)
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            print("-" * 100)

# Summary Report
print("\n\n" + "=" * 100)
print("TEST SUMMARY REPORT")
print("=" * 100)
print(f"Total Tests Run: {test_number}")
print(f"Successful Responses (Good Info): {successful_responses_count} ({successful_responses_count/total_tests*100:.1f}%)")
print(f"Weak Responses (Tier 3 Needed): {weak_responses_count} ({weak_responses_count/total_tests*100:.1f}%)")
print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 100)

# Analysis recommendations
print("\n" + "=" * 100)
print("ANALYSIS & RECOMMENDATIONS")
print("=" * 100)

if weak_responses_count > total_tests * 0.5:
    print("⚠️  HIGH WEAK RESPONSE RATE (>50%)")
    print("   → Manuals may lack comprehensive DIY information")
    print("   → Tier 3 cross-brand reference is critical for this use case")
    print("   → Consider adding more examples to prompts or increasing top_k further")
elif weak_responses_count > total_tests * 0.3:
    print("⚠️  MODERATE WEAK RESPONSE RATE (30-50%)")
    print("   → Some queries lack manual coverage")
    print("   → Tier 3 is working as designed for gap-filling")
    print("   → System is appropriately transparent about limitations")
else:
    print("✅ LOW WEAK RESPONSE RATE (<30%)")
    print("   → Manuals provide good coverage for most queries")
    print("   → RAG system is extracting information effectively")
    print("   → Tier 3 available for edge cases")

print("\n" + "=" * 100)
print("TEST SUITE COMPLETE")
print("=" * 100)
