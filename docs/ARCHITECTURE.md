# Architecture

This document is the single reference for system structure, component design, and architectural decisions.

For pipeline flows and step-by-step processes, see [WORKFLOWS.md](./WORKFLOWS.md).

> **Learning context:** Read this document after completing [AI_LEARNING_GUIDE.md](./AI_LEARNING_GUIDE.md) Parts 1–6. The system layers diagram and design patterns will click once you have seen each component work in practice.
>
> **If you are brand new:** Start with [LEARNING_PATH.md](./LEARNING_PATH.md) instead of this document.
>
> **After this document:** Read [PORTFOLIO_NARRATIVE.md](./PORTFOLIO_NARRATIVE.md) to understand the *why* behind each design decision.

---

## Overview

This RAG (Retrieval-Augmented Generation) system answers questions by:

1. **Storing** your documents as searchable chunks in a vector database
2. **Retrieving** the most relevant chunks when you ask a question (hybrid semantic + keyword search)
3. **Generating** an answer by providing those chunks as context to an LLM
4. **Evaluating** the quality of retrieval and generation with metrics
5. **Delivering** the answer with source citations and quality scores

The architecture below shows how these components are organized.

---

## System Layers

```
+------------------------------------------------------------------+
|                         CLI Layer                                |
|    src/cli/__init__.py  (InteractiveRAG)                        |
|    load / query / expand / multihop / toggles / inspection       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     Orchestration Layer                          |
|    src/core/rag_system.py  (RAGSystem)                          |
|    Coordinates all components; owns configuration and state      |
+------------------------------------------------------------------+
         |          |            |             |           |
         v          v            v             v           v
+----------+  +----------+  +--------+  +-----------+  +--------+
| Retrieval|  | Reasoning|  | Genera-|  | Evaluation|  | Persis-|
|          |  |          |  | tion   |  |           |  | tence  |
| Hybrid   |  | Query    |  |        |  | RAGAS     |  |        |
| Search   |  | Expander |  | LLM    |  | Evaluator |  | JSON   |
|          |  |          |  | Answer |  |           |  | Storage|
| Loader   |  | MultiHop |  | Genera-|  | Fact      |  |        |
|          |  | Reasoner |  | tor    |  | Checker   |  +--------+
| Chunker  |  |          |  |        |  |           |
|          |  | Self     |  +--------+  | Hallucin. |
| LRU      |  | Query    |             | Detector  |
| Cache    |  | Decomp.  |             +-----------+
|          |  |          |
| Domain   |  | Adversar.|
| Guard    |  | Suite    |
|          |  |          |
| Reranker |  |          |
|          |  |          |
| Passage  |  |          |
| Highlight|  |          |
+----------+  +----------+
         |
         v
+------------------------------------------------------------------+
|                       Storage Layer                              |
|  ChromaDB (vectors)  |  BM25 index (in-memory)  |  JSON files   |
+------------------------------------------------------------------+
```

---

## Storage Design

### ChromaDB (vector store)
- Persistent client stored in `./chroma_db`
- One collection per loaded source (auto-named)
- Stores chunk text + metadata (source, type, timestamp)
- Used for semantic search via cosine similarity

### BM25 (in-memory keyword index)
- Rebuilt each time a source is loaded
- Tokenized with NLTK `punkt`
- Scores via BM25Okapi from `rank-bm25`
- Discarded at session end (not persisted)

### JSON (persistence)
- `json_data/conversation_history.json` — all turns
- `json_data/evaluation_metrics.json` — RAGAS scores per query
- Written by `JSONStorage` after every query

---

## Design Patterns

| Pattern      | Where applied                                                            |
| ------------ | ------------------------------------------------------------------------ |
| Orchestrator | `RAGSystem` coordinates all components through a single interface        |
| Strategy     | `HybridSearchEngine` combines two interchangeable search strategies      |
| Factory      | `MultiSourceDataLoader` selects the right loader by source type          |
| Decorator    | Feature toggles in CLI wrap the base query pipeline without modifying it |
| Cache-Aside  | `EmbeddingCache` wraps ChromaDB embedding calls with LRU cache           |
| Command      | CLI maps each text command to a dedicated handler method                 |

---

## Performance Characteristics

| Operation               | Typical latency | Notes                           |
| ----------------------- | --------------- | ------------------------------- |
| Load Wikipedia page     | 2–4 s           | Network + chunking + indexing   |
| Embed query (cold)      | 100–200 ms      | ChromaDB default embeddings     |
| Embed query (cached)    | < 1 ms          | LRU cache hit                   |
| Semantic search         | 50–100 ms       | ChromaDB vector query           |
| BM25 keyword search     | 10–30 ms        | In-memory                       |
| Cross-encoder rerank    | 150–250 ms      | Per query-doc pair (5 docs)     |
| MMR diversity filter    | 5–10 ms         | In-memory Jaccard similarity    |
| Passage extraction      | 20–50 ms        | NLTK sentence tokenization      |
| LLM generation          | 1–3 s           | Model and prompt size dependent |
| RAGAS evaluation        | 500 ms – 1 s    | 3 LLM-judged sub-metrics        |
| Fact-checking           | 300–500 ms      | Per-claim context matching      |
| Hallucination detection | 200–400 ms      | Claim grounding similarity      |
| Full query (warm cache) | 2–5 s           | Generation dominates            |
| Full query (cold)       | 4–8 s           | Includes embedding computation  |
| Full query (reranking)  | 4–6 s           | +150-250ms for cross-encoder    |
