import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Mistri AI - Appliance Assistant",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme with high contrast
st.markdown("""
    <style>
    /* Dark theme */
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    [data-testid="stHeader"] {
        background-color: #0f172a;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #cbd5e1;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Message styling */
    .user-message {
        background-color: #1e3a8a;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #3b82f6;
        color: #ffffff;
    }
    
    .assistant-message {
        background-color: #065f46;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #10b981;
        color: #ffffff;
    }
    
    .official-response {
        background-color: #1e40af;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #3b82f6;
        margin: 1rem 0;
        color: #ffffff;
        line-height: 1.7;
    }
    
    .warning-box {
        background-color: #92400e;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        color: #fef3c7;
    }
    
    .reference-box {
        background-color: #334155;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 4px solid #94a3b8;
        margin: 0.75rem 0;
        color: #f1f5f9;
    }
    
    .source-card {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        margin: 0.5rem 0;
        color: #e2e8f0;
    }
    
    /* Force all text to be light colored */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    p, span, div, label {
        color: #e2e8f0 !important;
    }
    
    /* Input styling */
    .stTextInput input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 2px solid #475569 !important;
        border-radius: 8px;
        font-size: 1rem;
    }
    
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #2563eb !important;
    }
    
    /* Selectbox styling */
    .stSelectbox select {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 2px solid #475569 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }
    
    /* Caption/footer text */
    .stCaption, .footer-caption {
        color: #94a3b8 !important;
    }
    
    /* Divider */
    hr {
        border-color: #334155 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Brand logo file paths
BRAND_LOGOS = {
    "Samsung": "logos/samsung.png",
    "Whirlpool": "logos/whirlpool.png",
    "LG": "logos/lg.png"
}

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

# Check if response is weak
def is_weak_response(response_text: str) -> bool:
    """Detect if response lacks info"""
    weak_indicators = [
        "does not provide",
        "doesn't provide",
        "not provide information",
        "no information"
    ]
    
    response_lower = response_text.lower()
    
    if len(response_text.strip()) < 100:
        return True
    
    first_50_chars = response_text[:50].lower()
    if any(indicator in first_50_chars for indicator in weak_indicators):
        return True
    
    has_instructions = any(word in response_lower for word in [
        "step", "1.", "2.", "3.", "follow", "locate", "remove", "install", 
        "press", "turn", "open", "close", "replace", "align"
    ])
    
    has_contact = any(word in response_lower for word in [
        "contact", "call", "service center", "certified technician"
    ])
    
    if has_instructions and has_contact:
        return False
    
    if has_contact and not has_instructions:
        return True
    
    if not any(indicator in response_lower for indicator in weak_indicators):
        return False
    
    return True

# App Header
st.markdown('<div class="main-header">🔧 Mistri AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your intelligent appliance manual assistant for professional technicians</div>', unsafe_allow_html=True)

# Sidebar - CLEAN VERSION WITH FIXED IMAGE PARAMETER
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Brand selection
    brand = st.selectbox(
        "Select Appliance Brand",
        ["Samsung", "Whirlpool", "LG"],
        index=["Samsung", "Whirlpool", "LG"].index(st.session_state.current_brand),
        help="Choose the brand of your refrigerator"
    )
    
    # Handle brand change
    if brand != st.session_state.current_brand:
        st.session_state.current_brand = brand
        st.session_state.messages = []
        st.session_state.show_cross_brand = False
        st.rerun()
    
    st.markdown("---")
    
    # Brand display with logo - FIXED PARAMETER
    model_codes = {
        "Samsung": "DA68-04823J-00",
        "Whirlpool": "W11468670D",
        "LG": "LPXS30866D"
    }
    
    # Display logo centered
    logo_path = BRAND_LOGOS.get(brand)
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    
    # Display brand info
    st.markdown(f"### {brand}")
    st.caption(f"Model: {model_codes[brand]}")
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.show_cross_brand = False
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📋 About")
    st.markdown("""
    **Mistri AI** helps professional technicians find answers from official appliance manuals.
    
    **Features:**
    - 📖 Official manufacturer docs
    - ✅ Verified source citations
    - 🔄 Cross-brand reference
    - 🔒 Transparent limitations
    """)
    
    st.divider()
    
    st.markdown("### 📊 Stats")
    st.metric("Questions Asked", len([m for m in st.session_state.messages if m['role'] == 'user']))

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
        label_visibility="collapsed"
    )
    
    ask_button = st.form_submit_button("🔍 Ask", type="primary")

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
                with st.spinner("Checking other brands..."):
                    cross_brand_results = search_other_brands(question, brand, index)
                    st.session_state.cross_brand_results = cross_brand_results
            else:
                st.session_state.cross_brand_results = None
            
            st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif ask_button and not question:
    st.warning("⚠️ Please enter a question first.")

# Display current response
if st.session_state.messages and "last_response" in st.session_state:
    st.markdown("---")
    st.markdown("### 📋 Response")
    
    st.markdown(f"#### ✅ {st.session_state.last_brand} Manual")
    st.markdown(f'<div class="official-response">{st.session_state.last_response}</div>', unsafe_allow_html=True)
    
    with st.expander("📚 View Sources", expanded=False):
        if st.session_state.last_sources:
            for idx, node in enumerate(st.session_state.last_sources[:5], 1):
                st.markdown(f"**Source {idx}** - Relevance: {node.score:.3f}")
                st.markdown('<div class="source-card">', unsafe_allow_html=True)
                
                if node.metadata:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Brand:** {node.metadata.get('brand', 'N/A')}")
                    with col_b:
                        st.markdown(f"**Chunk:** {node.metadata.get('chunk_index', 'N/A')}")
                
                st.text(node.text[:300] + "..." if len(node.text) > 300 else node.text)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No sources available.")
    
    if st.session_state.get("is_weak_response") and st.session_state.get("cross_brand_results"):
        st.markdown("---")
        st.markdown('<div class="warning-box">⚠️ <strong>Limited information found.</strong> Cross-brand references available below.</div>', unsafe_allow_html=True)
        
        if st.button("🔍 View Cross-Brand References"):
            st.session_state.show_cross_brand = not st.session_state.show_cross_brand
        
        if st.session_state.show_cross_brand:
            st.markdown("#### 🛠️ Other Brand References")
            st.warning("⚠️ **DISCLAIMER**: Information from different models. Verify compatibility before applying.")
            
            for other_brand, data in st.session_state.cross_brand_results.items():
                with st.expander(f"📖 {other_brand} Reference"):
                    st.markdown(f'<div class="reference-box">{data["response"]}</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown('<p class="footer-caption">Powered by LlamaIndex, ChromaDB, and OpenAI | Built with Streamlit</p>', unsafe_allow_html=True)
st.markdown('<p class="footer-caption">🔧 Mistri AI - Professional appliance manual assistant</p>', unsafe_allow_html=True)
