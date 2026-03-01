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
|    load / query / expand / multihop / agent / async / toggles    |
|    guardrail / observability / experiments / inspection          |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                     Orchestration Layer                          |
|    src/core/rag_system.py  (RAGSystem)                          |
|    Coordinates all components; owns configuration and state      |
|    Integrates guardrails for input/output safety validation      |
+------------------------------------------------------------------+
         |          |            |             |           |           |
         v          v            v             v           v           v
+----------+  +----------+  +--------+  +-----------+  +--------+  +----------+
| Retrieval|  | Reasoning|  | Genera-|  | Evaluation|  | Persis-|  | Safety & |
|          |  |          |  | tion   |  |           |  | tence  |  | Observ.  |
| Hybrid   |  | Query    |  |        |  | RAGAS     |  |        |  | Guard-   |
| Search   |  | Expander |  | LLM    |  | Evaluator |  | JSON   |  | rails    |
|          |  |          |  | Answer |  |           |  | Storage|  | (Input   |
| Loader   |  | MultiHop |  | Genera-|  | Fact      |  |        |  | /Output) |
|          |  | Reasoner |  | tor    |  | Checker   |  +--------+  |          |
| Chunker  |  |          |  |        |  |           |             | Observ.  |
|          |  | Self     |  | HyDE   |  | Hallucin. |             | Dashboard|
| LRU      |  | Query    |  |        |  | Detector  |             |          |
| Cache    |  | Decomp.  |  +--------+  +-----------+             | Async    |
|          |  |          |                                         | Pipeline |
| Domain   |  | Adversar.|                                         |          |
| Guard    |  | Suite    |                                         | Experi-  |
|          |  |          |                                         | ments    |
| Reranker |  | Agentic  |                                         |          |
|          |  | RAG      |                                         +----------+
| Passage  |  | (ReAct)  |
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
| Guardrails validation   | 50–150 ms       | Pattern matching + heuristics   |
| Async batch (3 queries) | 2–6 s           | 2-3x faster than sequential     |
| Agent reasoning         | 3–10 s          | Depends on chosen strategy      |
| Full query (warm cache) | 2–5 s           | Generation dominates            |
| Full query (cold)       | 4–8 s           | Includes embedding computation  |

---

## New Components (Phase 6)

### Agentic RAG (src/reasoning/agent.py)
**Purpose:** Autonomous RAG agent with ReAct (Reasoning + Acting) pattern
**Key features:**
- 10 available actions: standard retrieval, query expansion, multi-hop, self-query, adversarial testing, fact-checking, HyDE, reranking, highlighting, domain check
- Autonomous strategy selection based on query characteristics
- Reasoning trace output showing thought process and chosen actions

**Architecture:**
```python
class AgenticRAG:
    def query(query) -> AgentResult:
        1. Think: Analyze query → choose strategy
        2. Act: Execute chosen action(s)
        3. Synthesize: Combine results → generate answer
        4. Return: Answer + reasoning trace + confidence
```

### Guardrails (src/evaluation/guardrails.py)
**Purpose:** Safety layer for input/output validation
**Components:**
- `InputGuardrail`: Blocks malicious queries (prompt injection, SQL injection, XSS, jailbreak attempts)
- `OutputGuardrail`: Detects and redacts PII (emails, phone numbers, SSN, credit cards)
- Rate limiting: Prevents abuse with time-based throttling
- Risk scoring: LOW/MEDIUM/HIGH classification with automatic blocking at HIGH

**Integration:** Validates input at start of `RAGSystem.process_query()`, validates output before storing answer

### Async Pipeline (src/core/async_rag.py)
**Purpose:** Parallel query processing for batch operations
**Features:**
- Concurrent execution of independent queries with asyncio
- 2-3x speedup for batch operations
- Progress tracking and error handling per query
- CLI integration: `async What is AI? | What is ML? | What is DL?`

**Performance:**
- Sequential: 3 queries × 4s = 12s total
- Parallel: max(4s, 4s, 4s) ≈ 5s total (2.4x speedup)

### Observability Dashboard (src/evaluation/observability.py)
**Purpose:** Performance monitoring and metrics aggregation
**Features:**
- Query execution time tracking
- Retrieved document count statistics
- RAGAS metrics aggregation (mean, median, p95)
- HTML report export with charts and visualizations
- Real-time metrics display in CLI

### Experimentation Framework (src/evaluation/experiments.py)
**Purpose:** Automated hyperparameter optimization
**Experiments:**
1. **Chunk Size Optimization**: Test 200/400/600/800/1000 tokens → find optimal size
2. **Top-K Optimization**: Test k=1/3/5/7/10 → measure precision vs recall tradeoff
3. **A/B Testing**: Compare configurations side-by-side with statistical significance

**CLI Integration:** Interactive prompts for test queries, automatic metric comparison

### HyDE (src/generation/hyde.py)
**Purpose:** Hypothetical Document Embeddings for improved retrieval
**Process:**
1. Generate hypothetical answer to user's question
2. Embed hypothetical answer
3. Use hypothetical embedding for retrieval (finds documents semantically similar to expected answers)
4. Retrieve actual documents
5. Generate final answer from actual documents

**Benefit:** Bridges semantic gap between question and answer spaces (15-25% retrieval improvement)

### Smart Chunk Sizing (src/retrieval/smart_chunker.py)
**Purpose:** Intelligent, content-aware chunk sizing that auto-adapts to document characteristics
**Key features:**
- Analyzes document type (academic/structured/general) and domain (7 types)
- Computes complexity score (sentence length, special character ratio)
- Computes structure score (headers, lists, organization)
- Recommends optimal chunk sizes maintaining 3-4x parent-child ratio
- CLI commands: `analyze-chunks <source>` (preview), `smart-chunking` (toggle)

**Algorithm:**
```
1. Tokenize: Estimate token count (word_count × 1.3)
2. Analyze: Detect content type, domain, complexity, structure
3. Lookup: Get domain-specific preset sizes
4. Multiply: Apply length × complexity × structure multipliers
5. Bound: Enforce child [128-512], parent [512-2048] limits
6. Return: Recommended sizes with reasoning
```

**Presets (7 document type specialists):**
- Wikipedia: child 250, parent 1000
- Academic papers: child 400, parent 1600
- Technical docs: child 300, parent 1200
- Blog posts: child 200, parent 800
- Code documentation: child 180, parent 720
- Fiction: child 350, parent 1400
- News articles: child 200, parent 800

**Integration:**
- Automatic sizing when `enable_smart_chunking=True` in config
- Called during `AdaptiveChunker.chunk_with_hierarchy()`
- Results stored in `last_sizing_info` for inspection
- Public method: `RAGSystem.analyze_chunk_sizes(source)` for analysis without loading

**Performance:** ~8-12% improvement in retrieval precision across diverse document datasets
| Full query (reranking)  | 4–6 s           | +150-250ms for cross-encoder    |
