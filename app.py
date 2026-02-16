import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Mistri AI - Appliance Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .brand-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .samsung-badge {
        background-color: #034EA2;
        color: white;
    }
    .whirlpool-badge {
        background-color: #ED1C24;
        color: white;
    }
    .lg-badge {
        background-color: #A50034;
        color: white;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        color: #1a1a1a;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #8bc34a;
        color: #1a1a1a;
    }
    .official-response {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    .reference-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6c757d;
        margin: 0.5rem 0;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_brand" not in st.session_state:
    st.session_state.current_brand = "Samsung"

if "show_cross_brand" not in st.session_state:
    st.session_state.show_cross_brand = False

# Initialize embedding model and load index
@st.cache_resource
def load_index():
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("fridge_manuals")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    return index

# Query function with brand filtering
def query_manual(brand: str, question: str, index):
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

# Cross-brand search for reference
def search_other_brands(question: str, current_brand: str, index):
    """Search other brands for reference information"""
    other_brands = [b for b in ["Samsung", "Whirlpool", "LG"] if b != current_brand]
    results = {}
    
    qa_prompt_str = """Extract instructions from the context below.

EXAMPLE:
- If user asks: "how do I clean the water filter"
- And context shows: "Locate the water filter... Lift open the filter cover door... rotate counterclockwise to remove..."
- You respond: "Water filters should be replaced, not cleaned. To replace: 1) Locate the water filter in the top-right corner. 2) Lift open the filter cover door. 3) Rotate counterclockwise to remove. 4) Install new filter and rotate clockwise."

INSTRUCTIONS:
- If context mentions the component (water filter, ice maker, etc.) and has steps, PROVIDE THOSE STEPS
- For "clean filter" questions: If context has "replace" steps, say "Filters are replaced, not cleaned. Steps: [extract exact steps]"
- DO NOT say "no information" if context clearly has instructions about that component
- Only say "no information" if context is truly empty about that component

Context:
{context_str}

Question: {query_str}
Answer:"""

    qa_prompt_tmpl = PromptTemplate(qa_prompt_str)
    
    for brand in other_brands:
        brand_filter = MetadataFilters(
            filters=[ExactMatchFilter(key="brand", value=brand)]
        )
        
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            filters=brand_filter,
            text_qa_template=qa_prompt_tmpl,
        )
        
        response = query_engine.query(question)
        results[brand] = {
            "response": response.response,
            "sources": response.source_nodes
        }
    
    return results

# Check if response is weak - FIXED TO CATCH MORE CASES
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
    
    # Also check if response is very short
    if len(response_text.strip()) < 30:
        return True
    
    if any(indicator in response_lower for indicator in weak_indicators):
        return True
    
    return False

# App Header
st.markdown('<div class="main-header">🔧 Mistri AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your intelligent appliance manual assistant for professional technicians</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    brand = st.selectbox(
        "Select Appliance Brand",
        ["Samsung", "Whirlpool", "LG"],
        index=["Samsung", "Whirlpool", "LG"].index(st.session_state.current_brand),
        help="Choose the brand of your refrigerator"
    )
    
    if brand != st.session_state.current_brand:
        st.session_state.current_brand = brand
        st.session_state.messages = []
        st.session_state.show_cross_brand = False
        st.rerun()
    
    badge_class = f"{brand.lower()}-badge"
    st.markdown(f'<span class="brand-badge {badge_class}">{brand}</span>', unsafe_allow_html=True)
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_cross_brand = False
        st.rerun()
    
    st.divider()
    
    st.subheader("📋 About")
    st.markdown("""
    Mistri AI helps professional technicians find answers from official appliance manuals.
    
    **Features:**
    - Official manufacturer documentation
    - Verified source citations
    - Cross-brand reference (when needed)
    - Transparent about limitations
    """)
    
    st.divider()
    
    st.caption(f"**Selected Brand:** {brand}")
    model_codes = {
        "Samsung": "DA68-04823J-00",
        "Whirlpool": "W11468670D",
        "LG": "LPXS30866D"
    }
    st.caption(f"**Model:** {model_codes[brand]}")
    st.caption(f"**Questions asked:** {len([m for m in st.session_state.messages if m['role'] == 'user'])}")

# Display chat history
if st.session_state.messages:
    st.markdown("### 💬 Conversation History")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Mistri AI:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    st.divider()

# Input form
st.markdown("### 🔍 Ask a Question")

with st.form(key="query_form", clear_on_submit=True):
    question = st.text_input(
        "Your question:",
        placeholder="e.g., How do I replace the water filter?",
        help="Type your question and press Enter",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.form_submit_button("🔍 Ask", type="primary", use_container_width=True)

# Process query
if ask_button and question:
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.spinner(f"🔎 Searching {brand} manual..."):
        try:
            index = load_index()
            response = query_manual(brand, question, index)
            
            st.session_state.messages.append({"role": "assistant", "content": response.response})
            st.session_state.last_sources = response.source_nodes
            st.session_state.last_response = response.response
            st.session_state.last_brand = brand
            
            is_weak = is_weak_response(response.response)
            st.session_state.is_weak_response = is_weak
            
            if is_weak:
                with st.spinner("Checking other brands for reference information..."):
                    cross_brand_results = search_other_brands(question, brand, index)
                    st.session_state.cross_brand_results = cross_brand_results
            else:
                st.session_state.cross_brand_results = None
            
            st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Please check your setup and try again.")

elif ask_button and not question:
    st.warning("⚠️ Please enter a question first.")

# Display current response with Tier 2 + 3 logic
if st.session_state.messages and "last_response" in st.session_state:
    st.markdown("---")
    st.markdown("### 📋 Latest Response Details")
    
    st.markdown(f"#### ✅ Official {st.session_state.last_brand} Manual Response")
    st.markdown(f'<div class="official-response">{st.session_state.last_response}</div>', unsafe_allow_html=True)
    
    with st.expander("📚 View Sources from Official Manual", expanded=False):
        if st.session_state.last_sources:
            for idx, node in enumerate(st.session_state.last_sources, 1):
                st.markdown(f"**Source {idx}** - Relevance: {node.score:.3f}")
                st.markdown('<div class="source-card">', unsafe_allow_html=True)
                
                if node.metadata:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(f"**Brand:** {node.metadata.get('brand', 'N/A')}")
                    with col_b:
                        st.markdown(f"**Model:** {node.metadata.get('model_code', 'N/A')}")
                    with col_c:
                        st.markdown(f"**Chunk:** {node.metadata.get('chunk_index', 'N/A')}")
                
                st.divider()
                st.text(node.text[:400] + "..." if len(node.text) > 400 else node.text)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("")
        else:
            st.info("No sources available.")
    
    if st.session_state.get("is_weak_response") and st.session_state.get("cross_brand_results"):
        st.markdown("---")
        st.markdown('<div class="warning-box">⚠️ <strong>Additional Reference Available</strong><br>The official manual has limited DIY information. Cross-brand reference data is available below.</div>', unsafe_allow_html=True)
        
        if st.button("🔍 View Cross-Brand Reference Information", use_container_width=True):
            st.session_state.show_cross_brand = not st.session_state.show_cross_brand
        
        if st.session_state.show_cross_brand:
            st.markdown("#### 🛠️ Reference Information from Other Brands")
            st.warning("⚠️ **IMPORTANT DISCLAIMER**: The information below is from different appliance models. Do not directly apply these steps to your selected brand without verifying compatibility and safety.")
            
            for other_brand, data in st.session_state.cross_brand_results.items():
                with st.expander(f"📖 {other_brand} Manual Reference", expanded=False):
                    st.markdown(f'<div class="reference-box"><strong>{other_brand} says:</strong><br>{data["response"]}</div>', unsafe_allow_html=True)
                    
                    st.markdown("**Sources:**")
                    for idx, node in enumerate(data["sources"][:3], 1):
                        st.caption(f"Source {idx} - Relevance: {node.score:.3f}")
                        st.text(node.text[:200] + "..." if len(node.text) > 200 else node.text)
                        st.markdown("")

# Footer
st.divider()
st.caption("Powered by LlamaIndex, ChromaDB, and OpenAI | Built with Streamlit")
st.caption("🔧 Mistri AI - Professional appliance manual assistant | Grounded in official documentation")
