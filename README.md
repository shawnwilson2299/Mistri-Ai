# Mistri AI – AI Copilot for Field Technicians

Converts refrigerator repair manuals into instant, voice-friendly answers for field technicians in India.

---

## Problem

Indian field technicians (AC, fridge, electrical repair) often work on-site, on calls, with limited time and patchy connectivity.
Most manufacturer manuals are 300–500+ pages in English; searching them on a small screen is slow and impractical.
The typical workflow is to call a senior technician, scroll PDFs blindly, or guess – all of which cost time and hurt first-time fix rates.

---

## Solution

Mistri AI is a RAG-powered assistant that answers questions like _"Why is this Samsung fridge not cooling?"_ or _"How do I clean the water filter?"_ in a few seconds, using the **official manuals only**.

- Brand-aware retrieval across Samsung, Whirlpool, and LG refrigerator manuals
- Strict metadata filtering so answers only come from the selected brand's manual
- Weak-answer detection to avoid hallucinations and say "I don't know" when needed
- Hinglish voice input via Whisper so technicians can speak queries instead of typing
- Streamlit UI designed for quick use by technicians on the job

---

## Tech Stack

- **LlamaParse** – AI PDF parsing with table/diagram preservation
- **OpenAI Embeddings** – `text-embedding-3-small` for semantic search
- **OpenAI Whisper** – `whisper-1` speech-to-text supporting English and Hinglish
- **ChromaDB** – local vector database with brand + section metadata
- **LlamaIndex** – RAG orchestration, query engine, and retrieval tuning
- **Streamlit** – dark-theme web UI with brand switching and conversation history
- **Python** – end-to-end glue: ingestion, indexing, retrieval, evaluation

---

## Architecture

1. **Ingestion & Parsing**
   - Parse refrigerator manuals for Samsung, Whirlpool, and LG using LlamaParse
   - Chunk documents with structured metadata: brand, section, page number

2. **Indexing**
   - Create brand-specific indexes in ChromaDB
   - Store embeddings via OpenAI, tagged with brand + page metadata

3. **Query Flow**
   - User selects a brand and speaks or types a question
   - Voice queries transcribed via Whisper before entering the RAG pipeline
   - Query routed to that brand's index with strict metadata filters
   - Top-k chunks retrieved and passed to LlamaIndex for answer generation
   - Weak-answer heuristics decide whether to answer, fall back, or say "I don't know"

4. **UI**
   - Streamlit app with:
     - Brand selector (Samsung / Whirlpool / LG)
     - Voice input with transcript preview and re-record option
     - Latest response shown immediately at top after each query
     - Cross-brand references shown directly below response when triggered
     - Conversation history in a collapsed expander
     - Expandable source citations with relevance scores
     - Dark theme and brand logo display

---

## Development Log

### Week 1 – Core RAG Pipeline

Goals: get end-to-end RAG working for at least one manual with acceptable latency.

- Parsed a 50-page Samsung refrigerator manual using LlamaParse
- Generated embeddings with OpenAI and stored them in ChromaDB
- Built a basic LlamaIndex query engine with `top_k = 5`
- Achieved ~2–3 second response times on typical queries

**Example queries**

> Q: "What temperature should I set for the freezer?"
> A: Set the freezer temperature to −19 °C.

> Q: "How do I clean the water filter?"
> A: Hold the top and bottom sides of the filter case, unlock it to reveal the deodorizer filter, replace the filter, and then reinsert the case.

---

### Week 2 – Multi-brand Support & Traceability

Goals: support all three brands and make answers traceable for real-world use.

- Designed the index for Samsung, Whirlpool, and LG manuals (brand stored as metadata, enforced via `ExactMatchFilter`)
- Added citation system with source page numbers, relevance scores, and text snippets
- Implemented traceability so every answer can be traced back to specific manual pages

---

### Week 3 – Reliability & Answer Quality

Goals: move from "sometimes right" to consistently useful.

- Built a technician-style test set of representative queries across all three brands
- Ran a baseline evaluation: correct answers on ~1 out of 5 queries (~17%)
- Iterated on retrieval depth (`top_k` increased), prompt structure, and weak-answer detection
- Added cross-brand fallback logic: if the selected brand's manual has no good match, the system searches other brands and surfaces results under a clearly labeled reference section with safety disclaimers
- After changes: improved from ~17% to ~58% on the same test set (4/5 queries answered correctly)

---

### Week 4 – Streamlit UI & UX

Goals: make the tool actually usable by a technician between calls.

- Built full Streamlit UI with dark theme, brand logo switching, sticky sidebar, and conversation history
- Added expandable source citations to keep the main view clean while staying auditable
- Fixed multiple CSS and deprecation issues around Streamlit theming

---

### Week 5 – Voice Input & UI Overhaul

Goals: add Hinglish voice input and fix UI flow for production quality.

**Voice Input (Whisper API)**
- Integrated OpenAI Whisper (`whisper-1`) for speech-to-text supporting English and Hinglish
- Field technicians can now speak queries instead of typing — directly addresses the core user persona
- Transcription appears immediately after recording with "Use this question" and "Re-record" buttons so users can verify before submitting
- Cost: ~$0.001 per query (negligible for demo use)

**Mic Stability Bug (Resolved)**
- `streamlit-mic-recorder` component was disappearing on brand switch and after every query
- Root cause: `st.rerun()` was destroying and remounting all components including the mic
- Fix: introduced `mic_key` counter in session state that only increments on brand switch — mic survives all other reruns

**UI Flow Redesign**
- Redesigned screen order to match user mental model:
  1. Latest Response — shown immediately at top after any query
  2. Cross-brand References — directly below the response, only when triggered
  3. Conversation History — collapsed expander, available but not cluttering the view
  4. Ask a Question — always anchored at the bottom
- Fixed white expander background on open state with CSS overrides
- Extracted shared query logic into `run_query()` to eliminate duplication between voice and typed paths

**Known Limitation**
- Cross-brand fallback occasionally surfaces ambient operating temperature ranges instead of internal fridge temperature settings — semantic overlap in manual chunks causes this
- Planned fix in v2: query expansion to rewrite queries into multiple semantic variations before retrieval

---

## Current Features

- Brand-aware RAG across Samsung, Whirlpool, and LG manuals
- Answers grounded strictly in the selected brand's documentation
- Hinglish voice input via Whisper with transcript preview and re-record
- Source citations with page numbers, relevance scores, and snippets
- Weak-answer detection and safe "I don't know" behavior
- Cross-brand fallback with safety disclaimers
- Streamlit UI with dark theme, brand switching, and chat history
- Latency typically 2–3 seconds per query

---

## Roadmap

- Query expansion: rewrite queries into multiple semantic variations before retrieval (v2)
- Hinglish TTS output: speak answers back to technicians in Hinglish using a Hindi-capable TTS model — ideal for hands-free use on a job site (v2)
- Expand to additional appliance categories beyond refrigerators
- Build a formal evaluation set and automate regression testing


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
# or manually:
# pip install chromadb llama-index-vector-stores-chroma \
#     llama-index-embeddings-openai openai llama-parse \
#     python-dotenv streamlit streamlit-mic-recorder

# 1) Build the index
python create_index.py

# 2) Run quick CLI test
python test_query.py

# 3) Launch Streamlit app
streamlit run app.py
