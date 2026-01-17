# Mistri AI - AI Copilot for Field Technicians

Converts repair manuals into instant, voice-powered answers for field technicians in India.

## Problem
Indian field technicians (AC, fridge, electrical repair) struggle with 500+ page English manuals during customer calls. Current solution: call senior techs or guess.

## Solution
RAG-powered AI system that answers technical questions in 2-3 seconds with relevant context.

## Tech Stack
- **LlamaParse** - AI-powered PDF parsing (preserves tables/diagrams)
- **OpenAI Embeddings** - text-embedding-3-small for semantic search ($0.02/1M tokens)
- **ChromaDB** - Local vector database
- **LlamaIndex** - RAG orchestration framework

## Progress (Week 1 Complete ✅)
- [x] Parse 50-page Samsung refrigerator manual
- [x] Generate embeddings with OpenAI API
- [x] Store vectors in ChromaDB
- [x] Query engine with improved retrieval (top_k=5)
- [x] 2-3 second response time

## Example Queries

**Q:** "What temperature should I set for the freezer?"  
**A:** Set the freezer temperature to -19 °C.

**Q:** "How do I clean the water filter?"  
**A:** To clean the water filter, hold the top and bottom sides of the filter case, unlock the filter case to reveal the deodorizer filter, replace the filter, and then reinsert the case.

## Next Steps (Week 2)
- Add citation system (page numbers)
- Improve UI with Gradio
- Add Hinglish voice input
- Deploy for field testing

## Setup
```bash
cd "Mistri AI"
python -m venv venv
source venv/bin/activate
pip install chromadb llama-index-vector-stores-chroma llama-index-embeddings-openai openai llama-parse python-dotenv
python create_index.py
python test_query.py
