# Multi-Source Question Answering Engine

A production-grade RAG system that answers questions from multiple knowledge sources with hybrid search, RAGAS quality metrics, hallucination detection, domain guard, and self-query decomposition.

This project is structured as a **learning environment** for software engineers transitioning into AI engineering. Every component has documented theory, code references, and hands-on exercises.

---

## Learning Journey

**New to AI? Start here:**

| Your situation                      | Where to start                                             |
| ----------------------------------- | ---------------------------------------------------------- |
| Never built an AI system            | [docs/LEARNING_PATH.md — Path A](docs/LEARNING_PATH.md)    |
| Experienced SWE, new to AI/ML       | [docs/LEARNING_PATH.md — Path B](docs/LEARNING_PATH.md)    |
| Hiring manager / portfolio reviewer | [docs/PORTFOLIO_NARRATIVE.md](docs/PORTFOLIO_NARRATIVE.md) |

**[docs/LEARNING_PATH.md](docs/LEARNING_PATH.md)** is the single document that tells you what to read, in what order, and what to run after each step. Start there before reading anything else.

**The core learning resource** is [docs/AI_LEARNING_GUIDE.md](docs/AI_LEARNING_GUIDE.md) — 3,200 lines of theory, code walkthroughs, and exercises that take you from ML basics through production RAG patterns.

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Configure environment
cp .env.example .env   # edit with your API key and model settings

# 5. Create data directories
mkdir -p json_data chroma_db

# 6. Start
python main.py
```

**Minimal `.env`:**
```
OPEN_AI_API_KEY=your_api_key
OPEN_AI_API_BASE_URL=http://127.0.0.1:1234/v1
OPEN_AI_MODEL=meta-llama-3.1-8b-instruct
```

**First session:**
```
> load wikipedia "Machine Learning"
> query What is supervised learning?
> multihop How did neural networks evolve into deep learning?
```

---

## Features

**Core (Phase 1)**
- Hybrid search — 70% semantic (ChromaDB) + 30% keyword (BM25)
- Multi-source loading — Wikipedia, web URLs, local files, PDFs
- Adaptive chunking — content-aware sizing: 256–1024 tokens per chunk
- RAGAS evaluation — context relevance, answer relevance, faithfulness
- Conversation memory — persistent history with context-aware follow-ups

**Advanced reasoning (Phase 2)**
- Query expansion — 4 variations to broaden retrieval coverage
- Multi-hop reasoning — decomposes complex queries into 3 sequential sub-steps
- Adversarial testing — 8 edge-case robustness test suite

**Performance and verification (Phase 3)**
- LRU embedding cache — ~50% speedup on repeated queries
- Fact-checking — claim-level verification against retrieved context
- Streaming responses — real-time token-by-token output

**Safety and quality (Phase 4)**
- Hallucination detection — grounding analysis, risk scoring (LOW/MEDIUM/HIGH), auto-mitigation
- Domain guard — detects out-of-domain queries against loaded source profile
- Self-query decomposition — splits multi-aspect queries into focused sub-queries

**Retrieval optimization (Phase 5)**
- Document reranking — two-stage retrieval: bi-encoder → cross-encoder reranking + MMR diversity
- Passage highlighting — sentence-level extraction with relevance scoring for transparency

---

## How It Works

### Basic Query Flow

When you ask a question, the system follows this pipeline:

1. **Load Your Data** (`load` command)
   - Fetch content from Wikipedia, web URLs, PDFs, or local files
   - Split into smart chunks (256-1024 tokens based on content type)
   - Store in vector database (ChromaDB) for semantic search
   - Build keyword index (BM25) for exact match retrieval

2. **Process Your Question** (`query` command)
   - **Search**: Hybrid search finds top 3-5 relevant chunks
     - 70% semantic similarity (vector embeddings)
     - 30% keyword matching (BM25)
   - **Rerank** (optional): Cross-encoder reranks for precision
   - **Context Building**: Format retrieved chunks into context

3. **Generate Answer**
   - **Inject Context**: Build prompt with:
     - System instructions: "Answer only using the provided context"
     - Retrieved document chunks
     - Conversation history (for follow-ups)
     - User question
   - **LLM Call**: OpenAI API generates grounded answer
   - **Cite Sources**: Include document references in response

4. **Quality Checks** (automatic)
   - **RAGAS Metrics**: Measure context relevance, answer relevance, faithfulness
   - **Hallucination Detection** (optional): Verify claims are grounded in context
   - **Fact Checking** (optional): Cross-reference facts against retrieved documents
   - **Passage Highlighting** (optional): Extract most relevant sentences

5. **Return & Store**
   - Display answer with source citations
   - Save conversation to JSON for audit trail
   - Cache embeddings for faster repeat queries

### Example Flow

```bash
> load wikipedia "Machine Learning"
  → Fetches article → Chunks into 500-token pieces → Embeds & indexes → Ready

> query What is supervised learning?
  → Searches ChromaDB + BM25 → Finds 3 relevant chunks about supervised learning
  → Builds prompt with chunks → Sends to LLM → Gets grounded answer
  → Evaluates with RAGAS → Returns answer with sources → Saves to history

> passages
  → Shows the exact sentences from documents that were most relevant
```

### Advanced Features

**For complex questions:**
- **Query Expansion**: Generate 4 variations to broaden search coverage
- **Multi-hop Reasoning**: Break into 3 sequential sub-questions, retrieve for each, synthesize final answer

**For quality assurance:**
- **Self-Query Decomposition**: Split multi-aspect questions (e.g., "What is X, how does Y work, where is Z used?")
- **Domain Guard**: Warn if question is outside loaded document scope
- **Streaming**: See answer tokens in real-time instead of waiting

**See [docs/WORKFLOWS.md](docs/WORKFLOWS.md) for detailed technical flow with code-level steps.**

---

## Commands

### Core

| Command            | Description                                    |
| ------------------ | ---------------------------------------------- |
| `load <source>`    | Load a Wikipedia page, URL, local file, or PDF |
| `query <question>` | Standard RAG query                             |
| `sources`          | List loaded sources                            |
| `history`          | Show conversation history                      |
| `metrics`          | Show RAGAS evaluation summary                  |
| `save [filename]`  | Save conversation to JSON                      |
| `clear`            | Clear conversation history                     |

**Load examples:**
```
> load wikipedia "Cristiano Ronaldo"
> load https://example.com/article
> load /path/to/document.pdf
> load notes.txt
```

### Advanced

| Command            | Description                        |
| ------------------ | ---------------------------------- |
| `expand <query>`   | Query with 4-variation expansion   |
| `multihop <query>` | 3-step decomposition and synthesis |

### Toggles and inspection

| Command                | Effect                                         |
| ---------------------- | ---------------------------------------------- |
| `streaming`            | Toggle streaming output (default: off)         |
| `fact-check`           | Toggle fact verification (default: off)        |
| `self-query`           | Toggle self-query decomposition (default: off) |
| `domain`               | Toggle domain guard (default: off)             |
| `hallucination`        | Toggle hallucination detection (default: off)  |
| `rerank`               | Toggle document reranking (default: off)       |
| `highlight`            | Toggle passage highlighting (default: off)     |
| `cache`                | Show embedding cache statistics                |
| `facts`                | Show last fact-check results                   |
| `hallucination-report` | Show last hallucination analysis report        |
| `domain-stats`         | Show domain profile and similarity threshold   |
| `passages`             | Show highlighted passages from last query      |

### General

| Command       | Description       |
| ------------- | ----------------- |
| `help`        | Show command list |
| `quit`/`exit` | Exit              |

---

## Configuration

All settings load from `.env` at startup.

| Variable               | Default                      | Description  |
| ---------------------- | ---------------------------- | ------------ |
| `OPEN_AI_API_KEY`      | `lm-studio`                  | API key      |
| `OPEN_AI_API_BASE_URL` | `http://127.0.0.1:1234/v1`   | LLM base URL |
| `OPEN_AI_MODEL`        | `meta-llama-3.1-8b-instruct` | Model name   |

Key runtime defaults (in `src/config.py`):

| Setting                       | Default | Description                         |
| ----------------------------- | ------- | ----------------------------------- |
| `semantic_weight`             | 0.7     | Semantic search weight in hybrid    |
| `keyword_weight`              | 0.3     | BM25 weight in hybrid               |
| `max_results`                 | 3       | Top-k documents retrieved           |
| `embedding_cache_size`        | 1000    | LRU cache capacity                  |
| `confidence_threshold`        | 0.6     | Minimum acceptable confidence       |
| `domain_similarity_threshold` | 0.35    | Domain guard threshold              |
| `query_expansion_count`       | 4       | Variations for expand command       |
| `multi_hop_steps`             | 3       | Decomposition depth for multihop    |
| `enable_reranking`            | False   | Enable cross-encoder + MMR          |
| `enable_passage_highlighting` | False   | Enable sentence-level extraction    |
| `mmr_lambda`                  | 0.7     | MMR balance: relevance vs diversity |

---


---

## Dependencies

| Package               | Version | Purpose                 |
| --------------------- | ------- | ----------------------- |
| chromadb              | 0.4.24  | Vector database         |
| openai                | 2.24.0  | LLM API client          |
| numpy                 | <2.0    | ChromaDB compatibility  |
| rank-bm25             | 0.2.2   | Keyword search          |
| nltk                  | 3.8.1   | Tokenization            |
| beautifulsoup4        | 4.12.2  | Web scraping            |
| PyPDF2                | 3.0.1   | PDF parsing             |
| tabulate              | 0.9.0   | Table formatting        |
| python-dotenv         | 1.0.0   | Env configuration       |
| sentence-transformers | 2.2.2   | Cross-encoder reranking |

---

## Troubleshooting

| Problem                  | Fix                                                       |
| ------------------------ | --------------------------------------------------------- |
| `ModuleNotFoundError`    | `source venv/bin/activate`                                |
| NLTK `punkt` not found   | `python -c "import nltk; nltk.download('punkt_tab')"`     |
| OpenAI connection error  | Check `.env`; ensure LM Studio or API endpoint is running |
| ChromaDB directory error | `mkdir -p json_data chroma_db`                            |
| `No sources loaded`      | Run `load wikipedia "Topic"` before querying              |

---

## Documentation

| File                                                         | Purpose                                                | Read when                          |
| ------------------------------------------------------------ | ------------------------------------------------------ | ---------------------------------- |
| [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md)               | Structured reading sequence for 3 audiences            | First — before anything else       |
| [docs/AI_LEARNING_GUIDE.md](docs/AI_LEARNING_GUIDE.md)       | RAG theory, all concepts, code walkthroughs, exercises | Core learning (follow Path A/B)    |
| [docs/TECHNIQUES_REFERENCE.md](docs/TECHNIQUES_REFERENCE.md) | All 46 techniques explained with examples & code       | Deep dive into specific techniques |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md)                       | Every pipeline flow step by step                       | After understanding core concepts  |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)                 | Component diagram, design patterns, performance        | After workflows                    |
| [docs/PORTFOLIO_NARRATIVE.md](docs/PORTFOLIO_NARRATIVE.md)   | Project story, decision rationale, demo scripts        | Portfolio review or interview prep |
