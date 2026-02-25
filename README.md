# 📺 YouTube-Chatbot: Advanced Agentic RAG System

> [!CAUTION]
> ### ⚠️ PROJECT UNDER ACTIVE DEVELOPMENT ⚠️
> **This repository is currently a work-in-progress. I am constantly experimenting with new LLM techniques, Agentic workflows, and retrieval strategies. The code is updated frequently and will be properly structured and finalized once the research phase is complete.**

---

## 🚀 Project Overview
This is an advanced Retrieval-Augmented Generation (RAG) system designed to transform long-form YouTube content into an interactive, searchable knowledge base. Unlike basic RAG systems, this project utilizes **Agentic Workflows** and **Hybrid Vector Search** to provide deep, context-aware insights from video transcripts.

## ✨ Technical "Inside the Box"

### 1. Advanced Data Ingestion & Metadata Pipeline
* **Multi-Format Transcript Parsing:** Robust logic to handle mixed timestamp formats (MM:SS and HH:MM:SS) to ensure zero data loss during ingestion.
* **YouTube Data API v3 Integration:** Automatically extracts rich metadata including channel statistics, categories, tags, and Wikipedia-linked topics for enhanced filtering.
* **Semantic Chunking:** Moves beyond fixed-character splitting. Uses `sentence-transformers` and cosine similarity thresholds to split text where the topic actually changes.

### 2. Two-Level Intelligence Architecture
* **Hybrid Summarization:** A recursive pipeline that generates "Section Summaries" ,"final_summary" and "timestamped_chunks" This preserves granular details that are usually lost in single-pass summaries.
* **Hybrid Search (Pinecone):** Combines Dense embeddings (`all-MiniLM-L6-v2`) with Sparse vectors (`TfidfVectorizer`) using the **Pinecone GRPC** client for high-precision retrieval.

### 3. Agentic Workflow (LangGraph)
The system is built as a state machine using **LangGraph**, allowing for complex decision-making:
* **User Tiering:** Logic for `Standard_User` (basic RAG) and `Premium_User` (Advanced Routing).
* **Intent-Based Routing:** The system doesn't just "search." It analyzes the user query to decide if it can answer using a high-level summary or if it needs to perform a "Deep Dive" into specific transcript sections.
* **Conditional Logic:** Dynamically switches between `SummaryRequest` and `SpecificQuestion` actions based on the query's complexity.

## 🛠️ Tech Stack
* **LLM:** Google Gemini (via `langchain-google-genai`)
* **Orchestration:** LangChain & LangGraph
* **Vector Database:** Pinecone (Hybrid Index)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Metadata & Processing:** Google API Client, `isodate`

---

## 🗂️ Current Repository Structure (Draft)
```text
YouTube-Chatbot/
├── frontend.py             # Streamlit UI (In Progress)
├── main_chain.py          # Core LangGraph & Agent logic
├── prepro_section_final.py  # Transcript parsing & Pinecone Upsert scripts only section and final summary
├── prepro.py                # Transcript parsing & Pinecone Upsert scripts only timestamped 
├── requirements.txt       # Project dependencies
├── .env                   # Local API Keys (Hidden)
