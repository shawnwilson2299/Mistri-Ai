# Mistri AI – AI Copilot for Field Technicians

Converts refrigerator repair manuals into instant, voice‑friendly answers for field technicians in India.

---

## Problem

Indian field technicians (AC, fridge, electrical repair) often work on‑site, on calls, with limited time and patchy connectivity.  
Most manufacturer manuals are 300–500+ pages in English; searching them on a small screen is slow and impractical.  
The typical workflow is to call a senior technician, scroll PDFs blindly, or guess – all of which cost time and hurt first‑time fix rates.

---

## Solution

Mistri AI is a RAG‑powered assistant that answers questions like _“Why is this Samsung fridge not cooling?”_ or _“How do I clean the water filter?”_ in a few seconds, using the **official manuals only**.

- Brand‑aware retrieval across Samsung, Whirlpool, and LG refrigerator manuals
- Strict metadata filtering so answers only come from the selected brand’s manual
- Weak‑answer detection to avoid hallucinations and say “I don’t know” when needed
- Streamlit UI designed for quick use by technicians

---

## Tech Stack

- **LlamaParse** – AI PDF parsing with table/diagram preservation  
- **OpenAI Embeddings** – `text-embedding-3-small` for semantic search  
- **ChromaDB** – local vector database with brand + section metadata  
- **LlamaIndex** – RAG orchestration, query engine, and retrieval tuning  
- **Streamlit** – dark‑theme web UI with brand switching and conversation history  
- **Python** – end‑to‑end glue: ingestion, indexing, retrieval, evaluation

---

## Architecture

1. **Ingestion & Parsing**
   - Parse refrigerator manuals for Samsung, Whirlpool, and LG using LlamaParse
   - Chunk documents with structured metadata: brand, section, page number

2. **Indexing**
   - Create brand‑specific indexes in ChromaDB
   - Store embeddings via OpenAI, tagged with brand + page metadata

3. **Query Flow**
   - User selects a brand and asks a question in natural language
   - Query routed to that brand’s index with strict metadata filters
   - Top‑k chunks retrieved and passed to LlamaIndex for answer generation
   - Weak‑answer heuristics decide whether to answer, fall back, or say “I don’t know”

4. **UI**
   - Streamlit app with:
     - Brand selector (Samsung / Whirlpool / LG)
     - Chat‑style interface with conversation history
     - Expandable source citations with page numbers and relevance scores
     - Dark theme and brand logo display for each manual

---

## Development Log

### Week 1 – Core RAG Pipeline

Goals: get end‑to‑end RAG working for at least one manual with acceptable latency.

- Parsed a 50‑page Samsung refrigerator manual using **LlamaParse**
- Generated embeddings with **OpenAI** and stored them in **ChromaDB**
- Built a basic **LlamaIndex** query engine with `top_k = 5`
- Achieved ~2–3 second response times on typical queries

**Example queries**

> Q: “What temperature should I set for the freezer?”  
> A: Set the freezer temperature to −19 °C.

> Q: “How do I clean the water filter?”  
> A: Hold the top and bottom sides of the filter case, unlock it to reveal the deodorizer filter, replace the filter, and then reinsert the case.

---

### Week 2 – Multi‑brand support & traceability

Goals: support all three brands and make answers traceable for real‑world use.

- Designed the index from day one for **Samsung, Whirlpool, and LG** manuals  
  (brand stored as metadata and enforced via `ExactMatchFilter`)
- Added citation system with:
  - Source page numbers
  - Relevance scores for each retrieved chunk
  - Text snippets so technicians can quickly verify answers
- Implemented traceability so every answer can be traced back to specific manual pages

---

### Week 3 – Reliability & Answer Quality

Goals: move from “sometimes right” to consistently useful.

- Built a small **technician‑style test set** of representative queries across all three brands
- Ran a baseline evaluation and saw correct answers on **~1 out of 5** queries (~17%)  
- Iterated on:
  - Retrieval depth (`top_k` increased from 10 to 15 in some flows)
  - Prompt structure – clearer instructions on citing manuals and handling edge cases
  - **Weak‑answer detection** using simple heuristics (short answers, hedging phrases, low relevance scores)
- Added **cross‑brand fallback logic**:
  - If the selected brand’s manual has no good match, the system can optionally search the other brands
  - Fallback answers are clearly labeled as cross‑brand references with safety disclaimers

After these changes, a focused spot‑check on 5 representative queries across all 3 brands improved from **1/5 correct to 4/5 correct** (headline: moved from ~17% to ~58% success on that check).

---

### Week 4 – Streamlit UI & UX

Goals: make the tool actually usable by a technician between calls.

- Built a full **Streamlit** UI:
  - Dark theme optimized for low‑light environments
  - Brand logo switching (Samsung / Whirlpool / LG)
  - Sticky sidebar with brand selector and controls
  - Conversation history for multi‑turn troubleshooting
- Fixed multiple CSS and deprecation issues around Streamlit theming
- Added expandable sections for **source citations** to keep the main UI clean while still being auditable

---

## Current Features

- Brand‑aware RAG across Samsung, Whirlpool, and LG manuals
- Answers grounded strictly in the selected brand’s documentation
- Source citations with page numbers, relevance, and snippets
- Weak‑answer detection and safe “I don’t know” behavior
- Streamlit UI with dark theme, brand switching, and chat history
- Latency typically in the 2–3 second range per query

---

## Roadmap

- Add **Hinglish voice input** using Whisper (speech‑to‑text tuned for Indian accents)  
- Expand to additional appliance categories beyond refrigerators  
- Build a more formal evaluation set and automate regression testing

---

## Setup

```bash
# Clone the repo
git clone https://github.com/shawnwilson2299/Mistri-Ai.git
cd "Mistri AI"

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# or, if you're installing manually:
# pip install chromadb llama-index-vector-stores-chroma \
#     llama-index-embeddings-openai openai llama-parse python-dotenv streamlit

# 1) Build the index
python create_index.py

# 2) Run quick CLI test
python test_query.py

# 3) Launch Streamlit app
streamlit run app.py
