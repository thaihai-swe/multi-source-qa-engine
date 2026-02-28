# 🎯 Complete Techniques & Concepts Reference Guide
##  All 46 Techniques Explained for New AI Engineers

**Purpose:** Comprehensive reference for every single technique and concept used in our RAG system.
**Audience:** New AI engineers who want to understand EXACTLY what each component does.
**Use:** Read linearly for learning, or jump to specific techniques as needed.

---

## How to Use This Guide

Each technique includes:
- ✅ **What it is** - Clear definition
- ✅ **Why it matters** - Production value
- ✅ **How we use it** - Implementation details
- ✅ **Where to find it** - File locations
- ✅ **Example** - Practical demonstration
- ✅ **Further reading** - Deep dive resources

---

# PART 1: CORE ML/AI FOUNDATIONS (11 Techniques)

## 1. RAG (Retrieval-Augmented Generation)

**What it is:**
An architectural pattern where you retrieve relevant documents from a knowledge base, then provide them as context to an LLM to generate grounded answers.

**Why it matters:**
- Solves "LLM doesn't know your data" problem
- Prevents hallucination by grounding in real documents
- Allows updating knowledge without retraining models
- Provides source citations for verification

**How we use it:**
```python
# src/core/rag_system.py - Main orchestrator
def process_query(query: str) -> RAGResponse:
    # 1. RETRIEVE - Find relevant docs
    docs = retriever.retrieve(query)

    # 2. AUGMENT - Build prompt with context
    prompt = build_prompt(query, docs)

    # 3. GENERATE - LLM answers from context
    answer = llm.generate(prompt)

    return RAGResponse(answer, sources=docs)
```

**Where to find it:**
- `src/core/rag_system.py` - Complete RAG pipeline
- `src/retrieval/hybrid_search.py` - Retrieval stage
- `src/generation/llm_answer_generator.py` - Generation stage

**Example:**
```
User: "What is machine learning?"
System:
  → Searches ChromaDB + BM25 for "machine learning"
  → Finds 3 relevant chunks from Wikipedia
  → Builds prompt: "Based on these documents: {chunks}, answer: {question}"
  → LLM generates grounded answer with citations
```

**Further reading:**
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

## 2. Large Language Models (LLMs)

**What it is:**
Neural networks with billions of parameters trained on massive text datasets to predict next tokens and generate human-like text.

**Why it matters:**
- Core generation engine for answers
- Understands context and nuance
- Can follow instructions and format output
- Enables natural language interaction

**How we use it:**
```python
# src/generation/llm_answer_generator.py
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",  # LM Studio local
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[
        {"role": "system", "content": "Answer based only on context..."},
        {"role": "user", "content": f"Context: {docs}\\n\\nQuestion: {query}"}
    ],
    temperature=0.2,  # Low = factual
    max_tokens=1000
)
```

**Where to find it:**
- `src/generation/llm_answer_generator.py` - Main LLM client
- `src/config.py` - LLM configuration (model, temp, max_tokens)
- `src/reasoning/query_expander.py` - LLM for query variations

**Example:**
```
Prompt: "Based on this context about Einstein's work on relativity,
         answer: What was Einstein's major contribution?"

LLM Output: "Einstein's major contribution was the theory of
            relativity, which fundamentally changed our understanding
            of space, time, and gravity."
```

**Configuration parameters:**
- `temperature`: 0.0 (deterministic) → 2.0 (creative)
- `max_tokens`: Response length limit
- `top_p`: Nucleus sampling parameter
- `stream`: Enable token-by-token output

---

## 3. Vector Embeddings

**What it is:**
Dense numerical representations (vectors) of text where semantically similar texts have similar vectors.

**Why it matters:**
- Enables semantic search (meaning-based, not keyword)
- Captures context and relationships
- Machine-readable representation of meaning
- Foundation for similarity computation

**How we use it:**
```python
# ChromaDB automatically generates embeddings
# src/retrieval/loader.py
collection.add(
    documents=["Machine learning is a subset of AI"],
    ids=["chunk_1"]
)
# ChromaDB converts text → 768-dimensional vector automatically

# Query with embeddings
results = collection.query(
    query_texts=["What is machine learning?"],
    n_results=3
)
# ChromaDB embeds query → finds most similar vectors → returns docs
```

**Mathematical representation:**
```python
text = "machine learning"
embedding = [0.023, -0.145, 0.892, ..., 0.134]  # 768 numbers
# Similar texts have similar vectors:
cosine_similarity("machine learning", "artificial intelligence") = 0.85
cosine_similarity("machine learning", "cooking recipes") = 0.12
```

**Where to find it:**
- ChromaDB handles embedding generation automatically
- `src/retrieval/cache.py` - Caches embeddings for performance
- All semantic search operations use embeddings

**Example:**
```
Input texts:
  "The cat sat on the mat"     → [0.1, 0.3, 0.5, ..., 0.2]
  "A kitten rested on carpet"  → [0.1, 0.3, 0.5, ..., 0.2]  # Similar!
  "Python programming syntax"  → [0.8, -0.2, 0.1, ..., 0.9]  # Different!
```

---

## 4. Cosine Similarity

**What it is:**
Measures the cosine of the angle between two vectors. Ranges from -1 (opposite) to 1 (identical), with 0 meaning orthogonal (unrelated).

**Why it matters:**
- Standard similarity metric for text embeddings
- Scale-invariant (works for vectors of different magnitudes)
- Fast to compute
- Intuitive interpretation (1 = same meaning, 0 = unrelated)

**Mathematical formula:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
  A · B = dot product
  ||A|| = magnitude (length) of vector A
```

**How we use it:**
```python
# ChromaDB uses cosine similarity internally
# src/retrieval/hybrid_search.py
results = collection.query(
    query_texts=[query],
    n_results=5
)
# Returns documents with highest cosine similarity to query embedding

# Manual calculation (for understanding):
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)
```

**Where to find it:**
- ChromaDB distance calculations (implicit)
- `src/evaluation/ragas_evaluator.py` - Answer relevance metric
- `src/evaluation/hallucination_detector.py` - Grounding scores

**Example:**
```python
query_vec = [0.5, 0.5, 0.0]
doc1_vec  = [0.6, 0.4, 0.0]  # Similar topic
doc2_vec  = [0.0, 0.0, 1.0]  # Different topic

similarity(query, doc1) = 0.98  # Very similar!
similarity(query, doc2) = 0.00  # Unrelated
```

**Interpretation:**
- 1.0 = Identical meaning
- 0.7-1.0 = Highly related
- 0.3-0.7 = Somewhat related
- 0.0-0.3 = Barely related
- < 0.0 = Opposite meanings (rare with text)

---

## 5. Semantic Search

**What it is:**
Search by meaning/concept rather than exact keyword matching. "Car" matches "automobile" even though words differ.

**Why it matters:**
- Finds conceptually related content
- Handles synonyms naturally
- Works across languages/paraphrasing
- More human-like search experience

**How we use it:**
```python
# src/retrieval/hybrid_search.py - Semantic component
def _semantic_search(self, query: str, n_results: int):
    # Get ChromaDB collection
    collection = self.collections[source]

    # ChromaDB handles: query → embedding → similarity search
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # Returns most similar documents by cosine similarity
    return self._parse_results(results)
```

**Where to find it:**
- `src/retrieval/hybrid_search.py` - 70% of hybrid search weight
- ChromaDB collections for each loaded source

**Example comparison:**
```
KEYWORD SEARCH:
Query: "automobile safety features"
Matches: Documents containing exact words "automobile", "safety", "features"
Misses: Documents about "car protection systems" (different words!)

SEMANTIC SEARCH:
Query: "automobile safety features"
Matches:
  - "Car safety systems"         ← "car" = "automobile"
  - "Vehicle protection features" ← conceptually similar
  - "Auto security mechanisms"    ← same meaning, different words
```

**Practical demonstration:**
```bash
> load wikipedia "Machine Learning"
> query "What is AI?"
# Semantic search finds docs about "artificial intelligence",
# "machine learning", "deep learning" even though query just says "AI"
```

---

## 6. Prompt Engineering

**What it is:**
Crafting instructions, context, and formatting for LLMs to achieve desired output quality and behavior.

**Why it matters:**
- Dramatically affects answer quality
- Controls response format and style
- Enforces grounding in context
- Prevents common LLM pitfalls

**How we use it:**
```python
# src/generation/llm_answer_generator.py
def _build_prompt(self, query: str, context: List[RetrievedDocument]) -> List[Dict]:
    # SYSTEM PROMPT - Sets behavior
    system_prompt = """You are a helpful AI assistant. Answer questions based
    ONLY on the provided context. If the context doesn't contain enough
    information, say "I don't have enough information to answer that."

    Always cite your sources using [Source: document_name].
    Be concise and accurate."""

    # CONTEXT INJECTION - Provide retrieved documents
    context_text = "\\n\\n".join([
        f"Document {i+1} ({doc.source}):\\n{doc.text}"
        for i, doc in enumerate(context)
    ])

    # USER PROMPT - The actual question
    user_prompt = f"""Context:\\n{context_text}\\n\\nQuestion: {query}\\n\\nAnswer:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
```

**Where to find it:**
- `src/generation/llm_answer_generator.py` - Main answer generation
- `src/reasoning/query_expander.py` - Query variation prompts
- `src/reasoning/multi_hop_reasoner.py` - Step-by-step prompts

**Example prompt structure:**
```
System: You are a helpful assistant. Answer using only the context below.

Context:
Machine learning is a subset of artificial intelligence that enables
computers to learn from data without explicit programming. [Source: Wikipedia]

Question: What is machine learning?

Answer: Machine learning is a subset of artificial intelligence that enables
computers to learn from data without explicit programming. [Source: Wikipedia]
```

**Best practices we follow:**
1. **Clear instructions** - "Answer using ONLY the context"
2. **Context injection** - Provide retrieved documents
3. **Format specification** - "Cite sources using [Source: X]"
4. **Fallback handling** - "Say 'I don't know' if uncertain"
5. **Conversation history** - Include previous turns for follow-ups

---

## 7. Temperature Control

**What it is:**
LLM parameter (0.0-2.0) controlling randomness in token selection. Lower = deterministic, higher = creative.

**Why it matters:**
- Controls factual accuracy vs. creativity
- Prevents hallucination (low temp)
- Enables diverse outputs when needed (high temp)
- Critical for production reliability

**How we use it:**
```python
# src/generation/llm_answer_generator.py
# FACTUAL ANSWERS - Low temperature
answer = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    temperature=0.2,  # ← Low for factual grounding
    messages=prompt
)

# src/reasoning/query_expander.py
# CREATIVE VARIATIONS - Higher temperature
variations = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    temperature=0.7,  # ← Higher for diverse paraphrases
    messages=expansion_prompt
)
```

**Temperature effects:**
```
Prompt: "What is the capital of France?"

Temperature 0.0:
"The capital of France is Paris."
"The capital of France is Paris."  ← Same every time
"The capital of France is Paris."

Temperature 0.8:
"The capital of France is Paris."
"France's capital city is Paris."  ← Varied phrasing
"Paris serves as the capital of France."

Temperature 1.5:
"The capital of France is Paris, a beautiful city..."  ← More creative
"Paris! The stunning capital of France with..."
"Well, France has Paris as its capital, which..."  ← Sometimes tangents
```

**Where to find it:**
- `src/config.py` - Default temperature settings
- All LLM API calls specify temperature

**Guidelines we follow:**
- **0.0-0.3**: Factual Q&A, information extraction
- **0.4-0.7**: Creative writing, query expansion
- **0.8-1.5**: Brainstorming, story generation
- **Never use > 1.5 in production** - Too unpredictable

---

## 8. Context Window

**What it is:**
Maximum amount of text (in tokens) an LLM can process in one request. Includes prompt + response.

**Why it matters:**
- Hard limit on retrieved context size
- Affects chunking strategy
- Determines conversation history length
- Impacts cost (pay per token)

**How we use it:**
```python
# Models have different context windows:
# - Llama-3.1-8B: 8,192 tokens
# - GPT-4: 8,192 or 128,000 tokens
# - Claude-3: 200,000 tokens

# We must fit within budget:
# src/retrieval/chunker.py
class AdaptiveChunker:
    def chunk(self, text: str) -> List[str]:
        # Chunks sized 256-1024 tokens
        # Retrieve 3-5 chunks = 768-5120 tokens
        # + System prompt (~200 tokens)
        # + Query (~50 tokens)
        # + Response (max 1000 tokens)
        # Total: ~2000-6500 tokens ← Well under 8192 limit
        pass
```

**Calculation example:**
```
Available context window: 8192 tokens

Budget allocation:
- System prompt: 200 tokens
- Retrieved context: 3 chunks × 500 tokens = 1500 tokens
- Conversation history: 300 tokens
- User query: 50 tokens
- Response budget: 1000 tokens
--------------------------------------
Total: 3050 tokens ← Safe! (38% of window)
```

**Where to find it:**
- `src/config.py` - Model context window settings
- `src/retrieval/chunker.py` - Chunk size constraints
- `src/generation/llm_answer_generator.py` - max_tokens parameter

**Trade-offs:**
- **More context** = Better answers but slower/costlier
- **Less context** = Faster/cheaper but might miss info
- **Our choice**: 3-5 chunks (~1500-2500 tokens) balances quality + speed

---

## 9. Tokens & Tokenization

**What it is:**
Breaking text into units (tokens) that LLMs process. 1 token ≈ 0.75 words in English, or 4 characters.

**Why it matters:**
- LLMs process tokens, not characters
- Pricing based on token count
- Context window measured in tokens
- Affects chunk sizing

**How we use it:**
```python
# NLTK tokenization (word level)
# src/retrieval/hybrid_search.py - BM25 keyword search
from nltk.tokenize import word_tokenize

text = "Machine learning is amazing!"
tokens = word_tokenize(text)
# Output: ['Machine', 'learning', 'is', 'amazing', '!']

# Sentence tokenization
# src/retrieval/passage_highlighter.py - Extract sentences
from nltk.tokenize import sent_tokenize

paragraph = "ML is great. It solves problems. Very powerful."
sentences = sent_tokenize(paragraph)
# Output: ['ML is great.', 'It solves problems.', 'Very powerful.']
```

**Token counting:**
```python
# Rough estimate (actual may vary by model)
def estimate_tokens(text: str) -> int:
    return len(text) / 4  # 1 token ≈ 4 chars

text = "Hello world"  # 11 chars
tokens = estimate_tokens(text)  # ~2.75 tokens

# More accurate (using tiktoken for OpenAI models):
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = len(encoding.encode("Hello world"))  # Exact count
```

**Where to find it:**
- `src/retrieval/chunker.py` - Token-based chunking
- BM25 uses NLTK word tokenization
- Passage highlighting uses sentence tokenization

**Examples:**
```
Text: "The cat sat on the mat."
Word tokens: ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']  # 7 tokens

Text: "Machine learning rocks!"
Subword tokens (LLM): ['Machine', ' learning', ' rocks', '!']  # 4 tokens
```

---

## 10. Grounding & Source Attribution

**What it is:**
Linking generated claims back to specific source documents. High grounding = low hallucination risk.

**Why it matters:**
- Transparency - users verify claims
- Trust - know where info comes from
- Legal compliance - auditable sources
- Debugging - trace wrong answers

**How we use it:**
```python
# src/core/rag_system.py - Every response includes sources
class RAGResponse:
    answer: str
    sources: List[RetrievedDocument]  # ← Source attribution
    confidence_score: float
    ragas_metrics: RAGASMetrics

# Display with citations
def _display_response(response: RAGResponse):
    print(f"Answer: {response.answer}")
    print("\\nSources:")
    for doc in response.sources:
        print(f"  - [{doc.source}] distance: {doc.distance:.3f}")
```

**Grounding verification:**
```python
# src/evaluation/hallucination_detector.py
def _compute_grounding_scores(self, claims: List[str], context: List[str]):
    """Check each claim against source documents"""
    for claim in claims:
        best_similarity = max([
            cosine_similarity(embed(claim), embed(doc))
            for doc in context
        ])

        if best_similarity >= 0.5:
            status = "GROUNDED"  # ✅ Found in source
        else:
            status = "HALLUCINATION"  # ❌ Not in source!
```

**Where to find it:**
- `src/models/data_models.py` - RAGResponse with sources
- `src/evaluation/hallucination_detector.py` - Grounding analysis
- CLI displays sources after every answer

**Example:**
```
Query: "What is machine learning?"

Answer: "Machine learning is a subset of AI that enables computers
         to learn from data. It includes supervised and unsupervised
         learning approaches."

Sources:
  ✅ [Wikipedia: Machine Learning] (distance: 0.12)
     → "Machine learning is a subset of artificial intelligence..."
  ✅ [Wikipedia: Supervised Learning] (distance: 0.28)
     → "Supervised learning uses labeled training data..."

Grounding: HIGH (all claims found in sources)
```

---

## 11. Hallucination in LLMs

**What it is:**
When LLMs generate plausible-sounding but factually incorrect or unsupported information.

**Why it matters:**
- #1 risk in production LLM systems
- Can spread misinformation
- Damages user trust
- Legal liability in critical domains

**How we detect it:**
```python
# src/evaluation/hallucination_detector.py
class HallucinationDetector:
    def analyze(self, answer: str, context: List[Document]) -> HallucinationReport:
        # 1. Extract claims from answer
        claims = self._extract_claims(answer)

        # 2. Check each claim against context
        groundings = []
        for claim in claims:
            best_match = max([
                cosine_similarity(embed(claim), embed(doc.text))
                for doc in context
            ])
            groundings.append(ClaimGrounding(
                claim=claim,
                grounding_score=best_match,
                is_grounded=(best_match >= self.threshold)
            ))

        # 3. Calculate risk level
        grounded_pct = sum(g.is_grounded for g in groundings) / len(groundings)

        if grounded_pct >= 0.80:
            risk = "LOW"
        elif grounded_pct >= 0.60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"  # ← Auto-regenerate with stronger prompt!

        return HallucinationReport(
            claims=groundings,
            risk_level=risk,
            grounding_score=grounded_pct
        )
```

**Where to find it:**
- `src/evaluation/hallucination_detector.py` - Detection logic
- Enabled via `hallucination` CLI toggle
- View results with `hallucination-report` command

**Example detection:**
```
Answer: "Einstein developed quantum mechanics in 1925 and won 3 Nobel Prizes."

Claims extracted:
1. "Einstein developed quantum mechanics"       ← Checking...
2. "Einstein developed this in 1925"           ← Checking...
3. "Einstein won 3 Nobel Prizes"               ← Checking...

Grounding analysis:
1. Grounding score: 0.35 ❌ HALLUCINATION (he contributed but didn't develop it)
2. Grounding score: 0.42 ❌ HALLUCINATION (wrong year, vague claim)
3. Grounding score: 0.28 ❌ HALLUCINATION (won 1 Nobel Prize, not 3)

Risk Level: HIGH (0% grounded)
Action: Regenerate answer with stronger grounding prompt
```

**Our mitigation strategies:**
1. **Low temperature** (0.2) - Reduces creativity
2. **Strong system prompts** - "Answer using ONLY context"
3. **Hallucination detection** - Auto-catches violations
4. **Auto-mitigation** - Regenerates with stricter prompt
5. **RAGAS faithfulness** - Continuous monitoring

---

# PART 2: SEARCH & RETRIEVAL TECHNIQUES (11 Techniques)

## 12. Hybrid Search

**What it is:**
Combines semantic search (70%) and keyword search (30%) for comprehensive document retrieval.

**Why it matters:**
- Semantic finds conceptually similar docs
- Keyword finds exact phrase matches
- Together: best of both worlds

**How we use it:**
```python
# src/retrieval/hybrid_search.py
class HybridSearchEngine:
    def retrieve(self, query: str, n_results: int = 3):
        # 1. Semantic search (ChromaDB)
        semantic_results = self._semantic_search(query, n_results=10)

        # 2. Keyword search (BM25)
        keyword_results = self._keyword_search(query, n_results=10)

        # 3. Fusion: Combine + weight scores
        combined = self._fuse_results(
            semantic_results,
            keyword_results,
            semantic_weight=0.7,  # ← 70%
            keyword_weight=0.3    # ← 30%
        )

        # 4. Sort by combined score, return top K
        return sorted(combined, key=lambda x: x.score, reverse=True)[:n_results]
```

**Where to find it:**
- `src/retrieval/hybrid_search.py` - Main implementation
- Core of every query's retrieval stage

**Example:**
```
Query: "machine learning algorithms"

Semantic search finds (by meaning):
  1. "ML training methods" (score: 0.92)
  2. "Neural networks overview" (score: 0.88)
  3. "Supervised learning" (score: 0.85)

BM25 finds (by keywords):
  1. "Machine learning algorithms: SVM, Random Forest..." (score: 12.3)
  2. "Algorithm complexity in ML" (score: 9.1)
  3. "Neural networks" (score: 5.2)

Hybrid combines:
  1. "Machine learning algorithms: SVM..." (0.7×0.85 + 0.3×12.3 = 9.54) ✅
  2. "ML training methods" (0.7×0.92 + 0.3×0 = 0.64)
  3. "Neural networks" (0.7×0.88 + 0.3×5.2 = 2.18)

Result: Exact keyword match ranks #1, even though semantic was lower!
```

---

## 13. BM25 Keyword Search

**What it is:**
Statistical ranking function (Okapi BM25) that scores documents by term frequency, inverse document frequency, and document length normalization.

**Why it matters:**
- Finds exact phrase/keyword matches
- Handles acronyms and proper nouns well
- Complements semantic search
- Fast and lightweight

**Formula:**
```
BM25(D, Q) = Σ(IDF(qi) × (f(qi,D) × (k1 + 1)) / (f(qi,D) + k1 × (1 - b + b × |D|/avgdl)))

Where:
  D = document
  Q = query
  qi = query term i
  f(qi,D) = term frequency in document
  |D| = document length
  avgdl = average document length
  k1, b = tuning parameters (typically k1=1.5, b=0.75)
  IDF = inverse document frequency
```

**How we use it:**
```python
# src/retrieval/hybrid_search.py
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Build BM25 index
def _build_bm25_index(documents: List[str]):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

# Search
def _keyword_search(self, query: str, n_results: int):
    query_tokens = word_tokenize(query.lower())
    scores = self.bm25_index.get_scores(query_tokens)

    # Get top K document indices
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [documents[i] for i in top_indices]
```

**Where to find it:**
- `src/retrieval/hybrid_search.py` - BM25 component
- Rebuilt on each `load` command
- 30% weight in hybrid search

**Example:**
```
Documents:
  Doc1: "Machine learning uses algorithms to learn from data."
  Doc2: "Python is a programming language for ML and data science."
  Doc3: "Neural networks are a type of ML algorithm."

Query: "ML algorithms"

BM25 scoring:
  Doc1: "ML" appears 0 times, "algorithms" appears 1 time → Score: 5.2
  Doc2: "ML" appears 1 time, "algorithms" appears 0 times → Score: 3.1
  Doc3: "ML" appears 1 time, "algorithms" appears 1 time → Score: 8.9 ✅

Doc3 ranks highest because BOTH query terms appear!
```

**Strengths:**
- Exact phrase matching
- Acronym handling (ML, AI, NLP)
- Proper nouns (Einstein, Paris)
- Fast (no neural network inference)

**Limitations:**
- Doesn't understand synonyms ("car" ≠ "automobile")
- Ignores word order ("New York" vs "York New")
- No semantic understanding

---

## 14. Two-Stage Retrieval

**What it is:**
Stage 1: Fast bi-encoder retrieves 50 candidates. Stage 2: Slow but accurate cross-encoder reranks to top 5.

**Why it matters:**
- **Speed + Accuracy** - Best of both worlds
- Bi-encoder: Fast but 70-80% precision
- Cross-encoder: Slow but 90-95% precision
- Industry standard for production RAG

**How we use it:**
```python
# Stage 1: Fast retrieval
# src/retrieval/hybrid_search.py
documents = hybrid_search.retrieve(query, n_results=50)  # ← Cast wide net

# Stage 2: Accurate reranking
# src/retrieval/reranker.py
if config.enable_reranking:
    documents = reranker.rerank(query, documents, top_k=5)  # ← Precision filter
```

**Where to find it:**
- `src/core/rag_system.py` - Orchestrates both stages
- `src/retrieval/reranker.py` - Stage 2 implementation

**Performance comparison:**
```
Retrieval only (Stage 1):
  Time: 50-100ms
  Precision: 75%
  Retrieves: 50 candidates

Two-stage (Stage 1 + Stage 2):
  Time: 200-350ms (+150-250ms)
  Precision: 92% (+17%!)
  Returns: Top 5 results

Trade-off: +250ms latency for +17% accuracy
```

**Example:**
```
Query: "How does backpropagation work in neural networks?"

STAGE 1 (Bi-encoder): Fast retrieval of 50 docs
  Results include:
    ✅ "Backpropagation algorithm explained"
    ✅ "Training neural networks with gradient descent"
    ❌ "Introduction to neural networks" (mentions both keywords but not relevant)
    ❌ "History of AI" (tangentially related)
    ... 46 more docs

STAGE 2 (Cross-encoder): Rerank to top 5
  Cross-encoder jointly encodes query + each doc
  Scores:
    1. "Backpropagation algorithm explained" (score: 9.2) ✅
    2. "Training neural networks with gradient descent" (score: 8.7) ✅
    3. "Computing gradients in deep learning" (score: 7.9) ✅
    4. "Neural network optimization" (score: 7.1) ✅
    5. "Activation functions in NNs" (score: 6.8) ✅

  Filtered out:
    50. "History of AI" (score: 2.1) ❌ - Not really relevant!
```

---

## 15. Cross-Encoder Reranking

**What it is:**
Neural model that jointly encodes query + document pairs for accurate relevance scoring. Slow but precise.

**Why it matters:**
- **90-95% precision** vs bi-encoder's 75%
- Captures query-document interactions
- Understands nuanced relevance
- Production standard for high-quality retrieval

**How we use it:**
```python
# src/retrieval/reranker.py
from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self):
        # MS MARCO model trained on 400M query-document pairs
        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-12-v2"
        )

    def _cross_encoder_rerank(self, query: str, documents: List[Document]):
        # Create query-document pairs
        pairs = [(query, doc.text) for doc in documents]

        # Score all pairs (this is the slow part!)
        scores = self.cross_encoder.predict(pairs)

        # Convert scores to distances (higher score = lower distance)
        for doc, score in zip(documents, scores):
            doc.distance = 1.0 - (score / 10.0)  # Score typically -10 to +10

        # Sort by distance (lower = better)
        return sorted(documents, key=lambda d: d.distance)
```

**Where to find it:**
- `src/retrieval/reranker.py` - Cross-encoder implementation
- `requirements.txt` - `sentence-transformers==2.2.2`
- Enabled via `rerank` CLI toggle

**Architecture comparison:**
```
BI-ENCODER (Stage 1):
  Query → Encoder → Vector_Q
  Doc   → Encoder → Vector_D
  Score = cosine_similarity(Vector_Q, Vector_D)

  Fast: Encode once, compare many
  Lower accuracy: No interaction between query and doc

CROSS-ENCODER (Stage 2):
  [Query + Doc] → Encoder → Relevance_Score

  Slow: Must encode every query-doc pair
  Higher accuracy: Joint encoding captures interactions
```

**Example:**
```
Query: "How to train a neural network?"

Document 1: "Neural networks require training data and backpropagation."

Bi-encoder scoring:
  Query embedding: [0.5, 0.3, 0.1, ..., 0.8]
  Doc embedding:   [0.5, 0.3, 0.1, ..., 0.7]
  Cosine sim: 0.92 ← Good, but doesn't understand "how to"

Cross-encoder scoring:
  Input: "[CLS] How to train a neural network? [SEP] Neural networks require training data and backpropagation."
  Output: 8.5/10 ← Understands this doc EXPLAINS how to train, not just mentions it!

Document 2: "Neural networks are powerful ML models."

Bi-encoder: 0.88 (seems relevant)
Cross-encoder: 3.2/10 (recognizes it doesn't explain HOW TO)
```

**Performance:**
- **Latency**: ~5-10ms per document pair
- **Batch size**: 32 pairs → ~200ms for 50 docs
- **Accuracy gain**: +15-20% precision vs bi-encoder

---

## 16. Bi-Encoder (Sentence-BERT)

**What it is:**
Neural model that encodes text into fixed-size vectors independently. Fast but less accurate than cross-encoders.

**Why it matters:**
- **Fast**: Embed once, compare millions of docs
- Enables real-time semantic search
- Foundation for Stage 1 retrieval
- Industry standard for large-scale search

**How we use it:**
```python
# ChromaDB handles bi-encoder embeddings automatically
# Default model: all-MiniLM-L6-v2 (384 dimensions)

# src/retrieval/loader.py
collection.add(
    documents=chunks,      # Each chunk embedded independently
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

# Query also embedded independently
results = collection.query(
    query_texts=[query],   # Query embedded same way
    n_results=50
)
# ChromaDB compares query embedding to all doc embeddings via cosine similarity
```

**Where to find it:**
- ChromaDB (implicit) - Default bi-encoder
- Fast retrieval for all semantic search
- Stage 1 of two-stage retrieval

**Architecture:**
```
Text → BERT/Transformers → Mean Pooling → L2 Normalization → 384-dim Vector

Example:
  "Machine learning" → [0.023, -0.145, 0.892, ..., 0.134]
  "AI algorithms"    → [0.021, -0.142, 0.888, ..., 0.138]  ← Similar!
  "Cooking pasta"    → [0.512, 0.883, -0.221, ..., -0.432] ← Different!
```

**Comparison:**
```
BI-ENCODER vs CROSS-ENCODER

Speed:
  Bi: Embed 1M docs once → Fast search anytime
  Cross: Must process every query-doc pair → Slow

Accuracy:
  Bi: 75-80% precision (no query-doc interaction)
  Cross: 90-95% precision (joint encoding)

Use case:
  Bi: Stage 1 retrieval (cast wide net)
  Cross: Stage 2 reranking (filter to best)
```

---

## 17. MMR (Maximal Marginal Relevance)

**What it is:**
Algorithm that balances relevance and diversity when selecting documents. Formula: `MMR = λ × Relevance - (1-λ) × Similarity`

**Why it matters:**
- Prevents redundant results (same info repeated)
- Ensures diverse perspectives
- Improves answer comprehensiveness
- Production best practice

**Mathematical formula:**
```
MMR = λ × Rel(doc, query) - (1-λ) × max Sim(doc, selected_docs)

Where:
  λ = balance parameter (0 to 1)
  Rel = relevance score (from cross-encoder or distance)
  Sim = similarity to already selected docs (Jaccard)

λ = 1.0: Pure relevance (might get duplicates)
λ = 0.0: Pure diversity (might sacrifice relevance)
λ = 0.7: Balanced (our default)
```

**How we use it:**
```python
# src/retrieval/reranker.py
def _apply_mmr(self, query: str, documents: List[Document], top_k: int):
    selected = []
    candidates = documents.copy()

    while len(selected) < top_k and candidates:
        # Calculate MMR score for each candidate
        mmr_scores = []
        for candidate in candidates:
            # Relevance: Lower distance = higher relevance
            relevance = 1.0 - candidate.distance

            # Diversity: Maximum similarity to already selected docs
            if selected:
                similarity = max([
                    self._compute_similarity(candidate, doc)
                    for doc in selected
                ])
            else:
                similarity = 0.0

            # MMR formula
            mmr_score = (self.config.mmr_lambda * relevance -
                        (1 - self.config.mmr_lambda) * similarity)
            mmr_scores.append((candidate, mmr_score))

        # Select candidate with highest MMR score
        best = max(mmr_scores, key=lambda x: x[1])
        selected.append(best[0])
        candidates.remove(best[0])

    return selected
```

**Where to find it:**
- `src/retrieval/reranker.py` - MMR implementation
- Applied after cross-encoder reranking
- Configured via `mmr_lambda` (default: 0.7)

**Example:**
```
Query: "machine learning applications"

After cross-encoder reranking:
  1. "ML in healthcare: diagnosis, treatment planning" (distance: 0.08)
  2. "ML in medical imaging: X-rays, MRIs" (distance: 0.12)  ← Very similar to #1!
  3. "ML in finance: fraud detection, trading" (distance: 0.15)
  4. "ML in autonomous vehicles" (distance: 0.18)
  5. "ML in recommendation systems" (distance: 0.20)

WITHOUT MMR (pure relevance):
  Returns: #1, #2, #3, #4, #5
  Problem: #1 and #2 both about healthcare → Redundant!

WITH MMR (λ=0.7):
  Iteration 1: Select #1 (highest relevance)
  Iteration 2:
    #2 MMR = 0.7×(1-0.12) - 0.3×similarity(#2,#1)
          = 0.7×0.88 - 0.3×0.9  ← High similarity penalty!
          = 0.616 - 0.27 = 0.346
    #3 MMR = 0.7×(1-0.15) - 0.3×similarity(#3,#1)
          = 0.7×0.85 - 0.3×0.2  ← Low similarity, different topic
          = 0.595 - 0.06 = 0.535 ✅ Winner!

  Final selection: #1, #3, #4, #5, ...
  Result: Healthcare, Finance, Vehicles, Recommendations → Diverse!
```

**Tuning λ:**
- **λ = 0.0**: Maximum diversity (might miss relevant docs)
- **λ = 0.5**: Equal weight to relevance and diversity
- **λ = 0.7**: Our default (favor relevance slightly)
- **λ = 1.0**: Pure relevance (might get redundancy)

---

## 18. Jaccard Similarity

**What it is:**
Measures similarity between two sets as: `|A ∩ B| / |A ∪ B|` (intersection over union).

**Why it matters:**
- Fast diversity calculation for MMR
- Works on word level (no embeddings needed)
- Intuitive interpretation (0 = no overlap, 1 = identical)
- Lightweight alternative to cosine similarity

**Formula:**
```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

Example:
  Set A: {machine, learning, algorithms}
  Set B: {machine, learning, models}

  A ∩ B = {machine, learning}  ← 2 elements in common
  A ∪ B = {machine, learning, algorithms, models}  ← 4 total unique elements

  Jaccard = 2/4 = 0.5
```

**How we use it:**
```python
# src/retrieval/reranker.py
def _compute_similarity(self, doc1: Document, doc2: Document) -> float:
    """Compute Jaccard similarity between two documents"""
    # Tokenize into word sets
    words1 = set(doc1.text.lower().split())
    words2 = set(doc2.text.lower().split())

    # Intersection over union
    intersection = words1 & words2
    union = words1 | words2

    if not union:
        return 0.0

    return len(intersection) / len(union)
```

**Where to find it:**
- `src/retrieval/reranker.py` - MMR diversity calculation
- Fast alternative to embedding-based similarity

**Example:**
```
Doc1: "machine learning uses algorithms to learn from data"
Doc2: "deep learning is a type of machine learning with neural networks"

Words1 = {machine, learning, uses, algorithms, to, learn, from, data}
Words2 = {deep, learning, is, a, type, of, machine, learning, with, neural, networks}

Common words: {machine, learning}
Total unique: {machine, learning, uses, algorithms, to, learn, from, data,
               deep, is, a, type, of, with, neural, networks}

Jaccard = 2 / 16 = 0.125

Interpretation: Low similarity → Good for MMR diversity!
              If Jaccard was 0.8, docs are too similar → MMR would penalize
```

**When to use:**
- **MMR diversity**: Fast word-level similarity
- **Deduplication**: Find near-duplicate documents
- **Preprocessing**: Quick similarity check before expensive embedding

**Limitations:**
- Ignores word order ("New York" vs "York New")
- No semantic understanding ("car" vs "automobile")
- Sensitive to document length

---

## 19. Passage Highlighting

**What it is:**
Extract most relevant sentences from retrieved documents for display. Shows users exactly which text supports the answer.

**Why it matters:**
- **Transparency**: Users see evidence
- **Trust**: Verify claims quickly
- **Debugging**: Find incorrect retrievals
- **Learning**: Understand system reasoning

**How we use it:**
```python
# src/retrieval/passage_highlighter.py
class PassageHighlighter:
    def extract_relevant_passages(
        self,
        query: str,
        answer: str,
        documents: List[Document]
    ) -> List[HighlightedPassage]:
        # 1. Extract keywords from query and answer
        query_keywords = self._extract_keywords(query)
        answer_keywords = self._extract_keywords(answer)

        all_passages = []
        for doc in documents:
            # 2. Split document into sentences
            sentences = sent_tokenize(doc.text)

            # 3. Score each sentence
            for idx, sentence in enumerate(sentences):
                score = self._calculate_sentence_relevance(
                    sentence, query_keywords, answer_keywords, doc
                )

                if score > 0.3:  # Threshold
                    all_passages.append(HighlightedPassage(
                        text=sentence,
                        source=doc.source,
                        relevance_score=score,
                        sentence_index=idx
                    ))

        # 4. Return top N passages
        return sorted(all_passages, key=lambda p: p.relevance_score, reverse=True)[:10]
```

**Relevance scoring:**
```python
def _calculate_sentence_relevance(
    self, sentence: str,
    query_kw: Set[str],
    answer_kw: Set[str],
    doc: Document
) -> float:
    sentence_kw = self._extract_keywords(sentence)

    # How much of query appears in sentence?
    query_overlap = len(sentence_kw & query_kw) / len(query_kw) if query_kw else 0

    # How much of answer appears in sentence?
    answer_overlap = len(sentence_kw & answer_kw) / len(answer_kw) if answer_kw else 0

    # Boost for top-ranked documents
    doc_boost = 0.3 if doc.distance < 0.2 else 0.0

    # Exact phrase match bonus
    phrase_bonus = 0.1 if any(phrase in sentence.lower() for phrase in query_kw) else 0.0

    # Combined score
    score = (query_overlap * 0.6 +     # 60% weight on query match
             answer_overlap * 0.4 +    # 40% weight on answer match
             doc_boost +               # Bonus for top docs
             phrase_bonus)             # Bonus for exact phrases

    return min(score, 1.0)  # Cap at 1.0
```

**Where to find it:**
- `src/retrieval/passage_highlighter.py` - Implementation
- `src/cli/__init__.py` - `passages` command
- Enabled via `highlight` toggle

**Example:**
```
Query: "How does supervised learning work?"

Retrieved documents (3 chunks from Wikipedia)

Answer: "Supervised learning uses labeled training data to learn patterns
         and make predictions on new data."

PASSAGE HIGHLIGHTING:

[1] Source: Wikipedia: Machine Learning | Score: 0.87
    "Supervised learning uses labeled examples where the correct output
     is known during training."
    ← High score: Contains "supervised learning", "labeled", "training"

[2] Source: Wikipedia: Training Data | Score: 0.74
    "Training data consists of input-output pairs used to teach models."
    ← Medium score: Has "training data" but lacks "supervised"

[3] Source: Wikipedia: Classification | Score: 0.65
    "Classification algorithms predict discrete categories from features."
    ← Lower score: Related but doesn't directly answer "how it works"
```

**Use cases:**
- **Research**: Find exact evidence for claims
- **Fact-checking**: Verify answer accuracy
- **Debugging**: See if wrong chunks were retrieved
- **Learning**: Understand what the system used

---

## 20. Batch Processing

**What it is:**
Process multiple items simultaneously instead of sequentially for efficiency.

**Why it matters:**
- **Parallelization**: Utilize multiple cores/GPUs
- **Reduced I/O**: Fewer API calls
- **Faster throughput**: 4× faster for 4 queries
- **Cost efficient**: Batch pricing tiers

**How we use it:**
```python
# src/reasoning/query_expander.py
def expand(query: str, num_variations: int = 4) -> List[str]:
    # Generate 4 variations in ONE LLM call instead of 4 calls!
    prompt = f"Generate {num_variations} variations of: {query}"
    response = llm.generate(prompt)
    variations = response.split('\\n')
    return variations

# Alternative (inefficient):
# for i in range(4):
#     variation = llm.generate(f"Generate variation {i}")  ← 4 slow API calls!

# src/retrieval/reranker.py
# Cross-encoder batch processing
def _cross_encoder_rerank(self, query: str, documents: List[Document]):
    # Score ALL 50 documents in batches of 32
    pairs = [(query, doc.text) for doc in documents]
    scores = self.cross_encoder.predict(pairs, batch_size=32)  # ← Batch!

    # vs sequential (slow):
    # scores = [self.cross_encoder.predict([query, doc.text])[0]
    #           for doc in documents]  ← 50 separate calls!
```

**Where to find it:**
- Query expansion (1 call for 4 variations)
- Cross-encoder reranking (batch_size=32)
- RAGAS evaluation (batch metrics)

**Performance comparison:**
```
SEQUENTIAL (one at a time):
  Query 1: 100ms
  Query 2: 100ms
  Query 3: 100ms
  Query 4: 100ms
  Total: 400ms

BATCHED (all together):
  Queries 1-4: 120ms  ← 3.3× faster!
  (Overhead of 20ms for batching)
```

---

## 21. LRU Caching

**What it is:**
Least Recently Used cache that stores computed values (embeddings) and evicts oldest items when full.

**Why it matters:**
- **~50% speedup** on repeated queries
- Saves API costs (don't recompute embeddings)
- Better user experience (instant responses)
- Production standard

**How we use it:**
```python
# src/retrieval/cache.py
from collections import OrderedDict

class EmbeddingCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, text: str) -> Optional[List[float]]:
        if text in self.cache:
            # Move to end (mark as recently used)
            self.cache.move_to_end(text)
            return self.cache[text]
        return None

    def set(self, text: str, embedding: List[float]):
        if text in self.cache:
            self.cache.move_to_end(text)
        else:
            self.cache[text] = embedding

            # Evict oldest if over capacity
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # Remove first (oldest) item
```

**Where to find it:**
- `src/retrieval/cache.py` - Implementation
- Wraps ChromaDB embedding calls
- Check stats with `cache` command

**Example behavior:**
```
Cache capacity: 3

Query 1: "machine learning"
  Cache miss → Compute embedding (100ms) → Store in cache
  Cache: ["machine learning"]

Query 2: "deep learning"
  Cache miss → Compute embedding (100ms) → Store
  Cache: ["machine learning", "deep learning"]

Query 3: "neural networks"
  Cache miss → Compute (100ms) → Store
  Cache: ["machine learning", "deep learning", "neural networks"]

Query 4: "machine learning"  ← Repeat!
  Cache HIT → Return stored embedding (<1ms) ✅ 100× faster!
  Cache: ["deep learning", "neural networks", "machine learning"] ← Moved to end

Query 5: "supervised learning"
  Cache MISS → Compute (100ms) → Store
  Cache full! Evict "deep learning" (least recently used)
  Cache: ["neural networks", "machine learning", "supervised learning"]
```

**Performance impact:**
```
WITHOUT CACHE:
  Query: 100ms (embedding) + 50ms (search) = 150ms

WITH CACHE (hit rate: 50%):
  Cache hit: <1ms + 50ms = 51ms  ← 3× faster!
  Cache miss: 100ms + 50ms = 150ms
  Average: (51 + 150) / 2 = 100ms  ← 33% faster overall
```

**View cache stats:**
```bash
> cache
Cache Statistics:
  Size: 247 / 1000
  Hit rate: 52.3%
  Recent queries:
    - "machine learning" (cached)
    - "deep learning" (cached)
    - "transformer architecture" (cached)
```

---

## 22. Graceful Degradation

**What it is:**
System continues functioning when optional dependencies are missing, falling back to simpler methods.

**Why it matters:**
- **Reliability**: Works everywhere
- **Flexibility**: Optional optimizations
- **Development**: Install only what's needed
- **Production**: Handle missing dependencies

**How we use it:**
```python
# src/retrieval/reranker.py
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("⚠️ sentence-transformers not installed. Cross-encoder disabled.")

class DocumentReranker:
    def __init__(self):
        if CROSS_ENCODER_AVAILABLE and config.use_cross_encoder:
            self.cross_encoder = CrossEncoder("ms-marco-MiniLM-L-12-v2")
        else:
            self.cross_encoder = None
            logger.info("Using MMR-only reranking")

    def rerank(self, query: str, documents: List[Document]):
        # Stage 1: Cross-encoder (if available)
        if self.cross_encoder:
            documents = self._cross_encoder_rerank(query, documents)
        else:
            logger.info("Cross-encoder unavailable, skipping Stage 1")

        # Stage 2: MMR (always available - lightweight)
        if config.use_mmr:
            documents = self._apply_mmr(query, documents, top_k=5)

        return documents
```

**Where to find it:**
- `src/retrieval/reranker.py` - Optional cross-encoder
- All imports wrapped in try/except
- Warnings logged, not errors

**Degradation levels:**
```
FULL SYSTEM (all dependencies):
  ✅ Hybrid search (semantic + BM25)
  ✅ Cross-encoder reranking
  ✅ MMR diversity
  ✅ Passage highlighting
  Performance: 100%

WITHOUT sentence-transformers:
  ✅ Hybrid search (semantic + BM25)
  ❌ Cross-encoder reranking → Skip
  ✅ MMR diversity (still works - word-based)
  ✅ Passage highlighting
  Performance: 85% (loses Stage 2 precision)

WITHOUT nltk data:
  ✅ Hybrid search
  ✅ Reranking
  ❌ Passage highlighting → Skip (needs sent_tokenize)
  ❌ BM25 → Fallback to semantic-only
  Performance: 70%

MINIMAL (only ChromaDB + OpenAI):
  ✅ Semantic search only
  ✅ LLM generation
  ✅ Basic RAGAS evaluation
  Performance: 60% (still functional!)
```

**Example:**
```bash
# Without sentence-transformers
> rerank
⚠️ Cross-encoder reranking unavailable (sentence-transformers not installed)
✅ MMR diversity still enabled
System continues to work with hybrid search + MMR

# vs crash (bad design):
> rerank
❌ ERROR: ModuleNotFoundError: No module named 'sentence_transformers'
❌ System unusable!
```

---

# PART 3: REASONING PATTERNS (3 Techniques)

## 23. Query Expansion

**What it is:**
Generate multiple variations of a user query (paraphrases, synonyms, reformulations) to broaden retrieval coverage.

**Why it matters:**
- **+12-15% coverage improvement**
- Finds docs that use different terminology
- Handles ambiguous queries better
- Reduces sensitivity to phrasing

**How we use it:**
```python
# src/reasoning/query_expander.py
class QueryExpander:
    @staticmethod
    def expand(query: str, num_variations: int = 4) -> List[str]:
        prompt = f"""Generate {num_variations} alternative phrasings for this query.
        Each should ask the same thing from different angles or with different wording.

        Original: {query}

        Return ONLY the variations, one per line."""

        response = llm.generate(prompt, temperature=0.7)  # Higher temp for diversity
        variations = [v.strip() for v in response.split('\\n')]

        # Always include original
        return [query] + variations[:num_variations-1]
```

**Where to find it:**
- `src/reasoning/query_expander.py`
- Used by `expand` CLI command
- Temperature: 0.7 (creative variations)

**Example:**
```
Original query: "How does machine learning work?"

Generated variations:
1. "How does machine learning work?"  ← Original
2. "What is the process of machine learning?"  ← Rephrased
3. "Explain the mechanics behind ML algorithms"  ← Synonyms
4. "How do machines learn from data?"  ← Simplified

Retrieval results:

Query 1 finds:
  - "Machine learning process overview"
  - " ML algorithm workflow"

Query 2 finds:
  - "The machine learning process involves..."  ← New match!
  - "Process of training ML models"  ← New match!

Query 3 finds:
  - "Mechanics of algorithm training"  ← New match!
  - "How ML algorithms function"

Query 4 finds:
  - "Machines learn by identifying patterns in data"  ← New match!
  - "Data-driven learning in computers"

Combined: 8 unique docs vs 2 from original query alone!
        Coverage: +300% ✅
```

**CLI usage:**
```bash
> expand What is supervised learning?

Generating 4 query variations...

Variations:
  1. What is supervised learning?
  2. Define supervised machine learning
  3. How does supervised learning differ from unsupervised?
  4. Explain the supervised learning paradigm

Retrieving with all variations...

Retrieved 12 unique documents (vs 3 with original query)
Coverage improvement: +300%
```

---

## 24. Multi-Hop Reasoning

**What it is:**
Break complex questions into 3 sequential sub-steps where each step's answer informs the next step's retrieval.

**Why it matters:**
- Handles questions requiring synthesis
- Retrieves different context for each step
- More accurate for complex reasoning
- Mimics human problem-solving

**How we use it:**
```python
# src/reasoning/multi_hop_reasoner.py
class MultiHopReasoner:
    def reason(self, query: str, num_steps: int = 3) -> MultiHopResult:
        steps = self._decompose_query(query, num_steps)

        intermediate_answers = []
        all_sources = []

        for step_query in steps:
            # Retrieve FOR THIS STEP
            docs = retriever.retrieve(step_query)

            # Generate INTERMEDIATE ANSWER
            answer = llm.generate(
                f"Based on: {docs}\\nAnswer: {step_query}"
            )

            intermediate_answers.append(answer)
            all_sources.extend(docs)

        # FINAL SYNTHESIS
        final_answer = llm.generate(
            f"""Original question: {query}

            Step-by-step findings:
            {chr(10).join(intermediate_answers)}

            Synthesize a complete answer:"""
        )

        return MultiHopResult(
            query=query,
            steps=steps,
            intermediate_answers=intermediate_answers,
            final_answer=final_answer,
            all_sources=all_sources
        )
```

**Where to find it:**
- `src/reasoning/multi_hop_reasoner.py`
- Used by `multihop` CLI command

**Example:**
```
Query: "How did Einstein's work lead to nuclear energy?"

DECOMPOSITION (LLM breaks into steps):
  Step 1: "What were Einstein's major contributions to physics?"
  Step 2: "How does mass-energy equivalence (E=mc²) relate to nuclear reactions?"
  Step 3: "How is nuclear fission used to generate energy?"

EXECUTION:

Step 1:
  Retrieve: Docs about Einstein's theories
  Answer: "Einstein developed special relativity and the equation E=mc²,
           showing mass and energy are interchangeable."

Step 2:
  Retrieve: Docs about E=mc² and nuclear physics
  Answer: "E=mc² explains that small amounts of mass can convert to huge
           energy. In nuclear fission, splitting atoms releases this energy."

Step 3:
  Retrieve: Docs about nuclear power plants
  Answer: "Nuclear reactors use controlled fission to heat water, create
           steam, and drive turbines generating electricity."

SYNTHESIS:
  "Einstein's equation E=mc² revealed that mass could convert to energy.
   This principle underlies nuclear fission, where splitting uranium atoms
   releases enormous energy. Nuclear power plants harness this through
   controlled fission reactions to generate electricity."

Sources: [Einstein biography, Nuclear physics textbook, NPP operations manual]
```

**When to use:**
- Questions requiring multiple facts
- "How did X lead to Y?" (causal chains)
- "Compare A and B" (need info on both)
- "What's the relationship between X and Y?"

---

## 25. Self-Query Decomposition

**What it is:**
Split multi-aspect questions into parallel focused sub-queries that can be answered independently.

**Why it matters:**
- Handles "What is X, how does Y work, and where is Z used?" style questions
- Each sub-query gets targeted retrieval
- More comprehensive answers
- Prevents LLM from ignoring sub-questions

**Difference from multi-hop:**
```
MULTI-HOP (Sequential):
  Step 1 → Answer 1 → Step 2 → Answer 2 → Step 3 → Answer 3 → Synthesis
  (Each step depends on previous answers)

SELF-QUERY (Parallel):
  Query 1 ──┐
  Query 2 ──┼→ Retrieve ALL → Generate ONE comprehensive answer
  Query 3 ──┘
  (All sub-queries independent)
```

**How we use it:**
```python
# src/reasoning/self_query_decomposer.py
class SelfQueryDecomposer:
    def decompose(self, query: str) -> DecomposedQuery:
        # 1. Check if query has multiple aspects
        prompt = f"""Does this query ask multiple distinct questions?

        Query: {query}

        If yes, split into focused sub-queries.
        If no, return "SINGLE"."""

        response = llm.generate(prompt, temperature=0.3)

        if "SINGLE" in response:
            return DecomposedQuery(
                original=query,
                sub_queries=[query],
                is_multi_aspect=False
            )

        # 2. Extract sub-queries
        sub_queries = [q.strip() for q in response.split('\\n') if q.strip()]

        return DecomposedQuery(
            original=query,
            sub_queries=sub_queries,
            is_multi_aspect=True
        )

    def retrieve_and_merge(self, decomposed: DecomposedQuery):
        # Retrieve for ALL sub-queries
        all_docs = []
        for sub_q in decomposed.sub_queries:
            docs = retriever.retrieve(sub_q, n_results=3)
            all_docs.extend(docs)

        # Deduplicate by content
        unique_docs = self._deduplicate(all_docs)

        # Generate ONE answer addressing ALL aspects
        answer = llm.generate(
            f"""Original question: {decomposed.original}

            Sub-aspects:
            {chr(10).join(decomposed.sub_queries)}

            Context: {unique_docs}

            Provide a comprehensive answer covering all aspects:"""
        )

        return answer, unique_docs
```

**Where to find it:**
- `src/reasoning/self_query_decomposer.py`
- Enabled via `self-query` toggle
- Applied automatically when detected

**Example:**
```
Query: "What is machine learning, how does it work, and where is it used?"

DECOMPOSITION:
  Detected: Multi-aspect query (3 parts)

  Sub-query 1: "What is machine learning?"
  Sub-query 2: "How does machine learning work?"
  Sub-query 3: "Where is machine learning used?"

RETRIEVAL:
  Sub 1 retrieves:
    - "Definition of ML"
    - "ML as subset of AI"
    - "Types of ML"

  Sub 2 retrieves:
    - "ML algorithms and training"
    - "Learning from data process"
    - "Model optimization"

  Sub 3 retrieves:
    - "ML applications in healthcare"
    - "ML in finance"
    - "ML in autonomous vehicles"

MERGED CONTEXT: 9 documents (after deduplication: 8 unique)

COMPREHENSIVE ANSWER:
  "Machine learning (ML) is a subset of artificial intelligence where
   systems learn from data without explicit programming. It works by...
   [explains process]... ML is used across industries including healthcare
   for diagnosis, finance for fraud detection, and autonomous vehicles
   for navigation."

✅ All three aspects addressed in ONE coherent answer!
```

**CLI usage:**
```bash
> self-query
Self-query decomposition: ON

> query What is supervised learning, how does it differ from unsupervised, and what are examples?

🔍 Multi-aspect query detected
Sub-queries:
  1. What is supervised learning?
  2. How does supervised differ from unsupervised learning?
  3. What are examples of supervised and unsupervised learning?

Retrieving for all sub-queries... (9 documents)

[Comprehensive answer addressing all aspects]
```

---

# PART 4: EVALUATION & QUALITY METRICS (7 Techniques)

## 26. RAGAS (Retrieval-Augmented Generation Assessment)

**What it is:**
Framework for evaluating RAG systems with four key metrics: context relevance, answer relevance, faithfulness, and context precision.

**Why it matters:**
- **Objective quality measurement**
- Track improvements over time
- Identify weak components
- Industry standard for RAG evaluation

**Metrics explained:**
```
1. CONTEXT RELEVANCE (0-1): Are retrieved docs relevant to query?
   High = All docs useful
   Low = Many irrelevant docs retrieved

2. ANSWER RELEVANCE (0-1): Does answer address the question?
   High = Answer on-topic
   Low = Answer tangential/off-topic

3. FAITHFULNESS (0-1): Is answer grounded in context?
   High = All claims supported by sources
   Low = Hallucination detected

4. CONTEXT PRECISION (0-1): Are top results most relevant?
   High = Best docs at top
   Low = Relevant docs buried
```

**How we use it:**
```python
# src/evaluation/ragas_evaluator.py
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    answer_relevancy,
    faithfulness,
    context_precision
)

class RAGASEvaluator:
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> RAGASMetrics:
        # Build evaluation dataset
        dataset = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth] if ground_truth else None
        }

        # Compute all metrics
        scores = evaluate(
            dataset,
            metrics=[
                context_relevancy,
                answer_relevancy,
                faithfulness,
                context_precision
            ]
        )

        return RAGASMetrics(
            context_relevance=scores["context_relevancy"],
            answer_relevance=scores["answer_relevancy"],
            faithfulness=scores["faithfulness"],
            context_precision=scores["context_precision"]
        )
```

**Where to find it:**
- `src/evaluation/ragas_evaluator.py`
- Enabled via `ragas` CLI toggle
- Displayed after every answer

**Example:**
```
Query: "What is machine learning?"
Answer: "Machine learning is a subset of AI that enables computers to learn from data."
Contexts: ["ML is a subset of AI...", "ML uses algorithms...", "Deep learning is a type of ML..."]

RAGAS Evaluation:
  Context Relevance: 0.92 ✅ (all docs about ML)
  Answer Relevance:  0.98 ✅ (directly answers question)
  Faithfulness:      0.95 ✅ (answer grounded in context)
  Context Precision: 0.88 ✅ (best docs at top)

Overall: EXCELLENT (all metrics > 0.85)
```

---

## 27. Context Relevance

**What it is:**
RAGAS metric measuring what percentage of retrieved documents are actually relevant to the query.

**Why it matters:**
- Measures retrieval quality
- High score = Good document selection
- Low score = Noisy retrieval hurts LLM

**Formula:**
```
Context Relevance = (Relevant Docs) / (Total Retrieved Docs)

Example:
  Retrieved 5 docs:
    ✅ Doc 1: About ML (relevant)
    ✅ Doc 2: About training (relevant)
    ❌ Doc 3: About cooking (irrelevant!)
    ✅ Doc 4: About algorithms (relevant)
    ❌ Doc 5: About history (tangential)

  Context Relevance = 3/5 = 0.60 (MEDIUM)
```

**What affects it:**
- Hybrid search quality
- Query phrasing
- Chunk size/overlap
- Collection relevance (loading wrong data)

**Where to find it:**
- Part of RAGAS evaluation
- View with `ragas` toggle enabled

---

## 28. Answer Relevance

**What it is:**
RAGAS metric measuring how well the generated answer addresses the user's question.

**Why it matters:**
- Catches off-topic answers
- Ensures LLM stays focused
- Measures prompt quality

**How it's computed:**
```
1. Generate variations of the answer (reverse query generation)
2. Compute similarity between original query and generated variations
3. High similarity = Answer on-topic
```

**Example:**
```
Query: "How does supervised learning work?"

Good answer (relevance: 0.95):
"Supervised learning trains models on labeled data where inputs are paired
 with correct outputs. The model learns patterns and makes predictions."

Generated variations:
  - "How does supervised learning function?" (sim: 0.98)
  - "What is the supervised learning process?" (sim: 0.95)
  - "Explain supervised ML training" (sim: 0.92)
Average similarity to original: 0.95 ✅

Bad answer (relevance: 0.42):
"Machine learning has many applications in various industries like healthcare..."

Generated variations:
  - "What are ML applications?" (sim: 0.35)
  - "Where is ML used?" (sim: 0.48)
  - "ML use cases in industry" (sim: 0.43)
Average similarity: 0.42 ❌ (Answer went off-topic!)
```

---

## 29. Faithfulness

**What it is:**
RAGAS metric measuring what percentage of claims in the answer are supported by retrieved context.

**Why it matters:**
- **#1 hallucination detector**
- Trust indicator
- Legal compliance (auditable claims)

**Formula:**
```
Faithfulness = (Supported Claims) / (Total Claims)

Example answer: "Einstein developed relativity in 1905 and won a Nobel Prize."
Claims:
  1. "Einstein developed relativity" ← Check context...
  2. "This happened in 1905" ← Check context...
  3. "Einstein won a Nobel Prize" ← Check context...

Context contains:
  ✅ "Einstein's theory of relativity..."
  ✅ "Published in 1905..."
  ✅ "Nobel Prize in Physics 1921..."

Faithfulness = 3/3 = 1.0 ✅ Perfect grounding!
```

**Where to find it:**
- Part of RAGAS metrics
- Complemented by `HallucinationDetector`

---

## 30. Hallucination Detection

**What it is:**
Specialized evaluation checking if LLM invented facts not present in source documents.

**Why it matters:**
- **Auto-remediation** - Regenerates bad answers
- Risk assessment (LOW/MEDIUM/HIGH)
- Production safety net

**How we use it:**
```python
# src/evaluation/hallucination_detector.py
class HallucinationDetector:
    def analyze(self, answer: str, context: List[Document]):
        # 1. Extract claims
        claims = self._extract_claims(answer)

        # 2. Check grounding for each claim
        groundings = []
        for claim in claims:
            # Embed claim and find best matching context
            claim_emb = embed(claim)
            best_score = max([
                cosine_similarity(claim_emb, embed(doc.text))
                for doc in context
            ])

            groundings.append(ClaimGrounding(
                claim=claim,
                grounding_score=best_score,
                is_grounded=(best_score >= 0.5)
            ))

        # 3. Calculate risk
        grounded_pct = sum(g.is_grounded for g in groundings) / len(groundings)

        if grounded_pct >= 0.80:
            risk = "LOW"
        elif grounded_pct >= 0.60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
            # AUTO-REMEDIATION
            self._trigger_regeneration(answer, context, risk="HIGH")

        return HallucinationReport(
            claims=groundings,
            risk_level=risk,
            grounding_score=grounded_pct
        )
```

**Where to find it:**
- `src/evaluation/hallucination_detector.py`
- Enabled via `hallucination` toggle
- View detailed report with `hallucination-report`

**Example:**
```
Answer: "Einstein developed quantum mechanics in 1925 and won 3 Nobel Prizes."

Hallucination Analysis:

Claim 1: "Einstein developed quantum mechanics"
  Best context match: "Einstein contributed to quantum theory..." (score: 0.35)
  Status: ❌ HALLUCINATION (threshold: 0.50)
  Note: He contributed but didn't develop alone

Claim 2: "This happened in 1925"
  Best context match: "His work on relativity was in 1905" (score: 0.28)
  Status: ❌ HALLUCINATION
  Note: Wrong year/event

Claim 3: "Einstein won 3 Nobel Prizes"
  Best context match: "Nobel Prize in Physics 1921" (score: 0.62)
  Status: ❌ HALLUCINATION (says 1 prize, not 3)

Grounding: 0% (0/3 claims supported)
Risk: HIGH ⚠️

Action: Auto-regenerating with stronger grounding prompt...
```

---

## 31. Fact Checking

**What it is:**
Cross-validates generated answers against trusted sources and detects factual errors.

**Why it matters:**
- Additional safety layer
- Catches subtle errors
- Domain-specific validation

**How we use it:**
```python
# src/evaluation/fact_checker.py
class FactChecker:
    def check(self, answer: str, sources: List[Document]) -> FactCheckResult:
        # 1. Extract factual claims
        facts = self._extract_facts(answer)

        # 2. Verify each fact
        checks = []
        for fact in facts:
            # Search for supporting evidence
            evidence = self._find_evidence(fact, sources)

            if evidence:
                status = "VERIFIED"
                confidence = evidence.similarity_score
            else:
                status = "UNVERIFIED"
                confidence = 0.0

            checks.append(FactCheck(
                fact=fact,
                status=status,
                evidence=evidence,
                confidence=confidence
            ))

        # 3. Overall verdict
        verified_pct = sum(c.status == "VERIFIED" for c in checks) / len(checks)

        return FactCheckResult(
            checks=checks,
            verified_percentage=verified_pct,
            verdict="PASS" if verified_pct >= 0.90 else "REVIEW"
        )
```

**Where to find it:**
- `src/evaluation/fact_checker.py`
- Used in conjunction with hallucination detector

**Example:**
```
Answer: "The capital of France is Paris, with a population of 2.1 million."

Fact Check:

Fact 1: "The capital of France is Paris"
  Evidence: "Paris is the capital and largest city of France" [Wikipedia]
  Status: ✅ VERIFIED (confidence: 0.98)

Fact 2: "Paris has a population of 2.1 million"
  Evidence: "The city of Paris has 2.1 million residents" [Wikipedia]
  Status: ✅ VERIFIED (confidence: 0.95)

Overall: PASS (100% verified)
```

---

## 32. Confidence Scoring

**What it is:**
Aggregate metric combining RAGAS scores, grounding analysis, and retrieval quality into single confidence score (0-100).

**Why it matters:**
- Single quality indicator for users
- Threshold for auto-responses
- A/B testing metric

**How we compute it:**
```python
# src/models/data_models.py
def calculate_confidence(
    ragas_metrics: RAGASMetrics,
    hallucination_risk: str,
    retrieval_quality: float
) -> float:
    # Weight different components
    ragas_score = (
        ragas_metrics.context_relevance * 0.25 +
        ragas_metrics.answer_relevance * 0.25 +
        ragas_metrics.faithfulness * 0.40 +      # ← Most important!
        ragas_metrics.context_precision * 0.10
    )

    # Penalty for hallucination risk
    hallucination_penalty = {
        "LOW": 0.0,
        "MEDIUM": 0.15,
        "HIGH": 0.40
    }[hallucination_risk]

    # Final score
    confidence = (ragas_score * 0.7 + retrieval_quality * 0.3) - hallucination_penalty

    return max(0.0, min(1.0, confidence)) * 100  # Scale to 0-100
```

**Where to find it:**
- Displayed with every answer
- `src/models/data_models.py` - Calculation logic

**Interpretation:**
```
90-100: EXCELLENT - Highly reliable
80-89:  GOOD - Generally trustworthy
70-79:  FAIR - Review recommended
60-69:  POOR - Use with caution
0-59:   UNRELIABLE - Do not use
```

---

# PART 5: DATA PROCESSING & STORAGE (5 Techniques)

## 33. ChromaDB Vector Database

**What it is:**
Embedded vector database for storing text embeddings and performing similarity search.

**Why it matters:**
- **No server required** - Embedded database
- $Fast semantic search
- Built-in embedding generation
- Persistent storage

**How we use it:**
```python
# src/retrieval/loader.py
import chromadb

class DocumentLoader:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")

    def load_source(self, source_name: str, documents: List[str]):
        # Create collection (vector store)
        collection = self.client.get_or_create_collection(
            name=source_name,
            metadata={"description": f"Docs from {source_name}"}
        )

        # Add documents (automatically embeds)
        collection.add(
            documents=documents,
            ids=[f"{source_name}_{i}" for i in range(len(documents))],
            metadatas=[{"source": source_name, "chunk_id": i}
                      for i in range(len(documents))]
        )

        return collection

    def query(self, collection_name: str, query: str, n_results: int = 5):
        collection = self.client.get_collection(collection_name)

        # Semantic search (embeds query, finds similar docs)
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results
```

**Where to find it:**
- Database stored in `chroma_db/` folder
- `src/retrieval/loader.py` - Loading documents
- `src/retrieval/hybrid_search.py` - Querying

**Structure:**
```
chroma_db/
├── chroma.sqlite3              ← Metadata database
├── 8f99bd6d-.../               ← Collection 1 (e.g., "wikipedia")
│   ├── data_level0.bin         ← Vector embeddings
│   └── header.bin              ← Collection metadata
├── 928e632f-.../               ← Collection 2 (e.g., "arxiv")
└── ...
```

**Example:**
```bash
> load wikipedia "Machine Learning"

Loading documents... ✓
Chunking into 512-token chunks... ✓
Generating embeddings (768-dim)... ✓
Storing in ChromaDB collection 'wikipedia'... ✓

Loaded: 42 chunks

> query What is ML?

Searching ChromaDB 'wikipedia' collection...
Found 3 relevant chunks (cosine similarity):
  1. "Machine learning is..." (distance: 0.08)
  2. "ML uses algorithms..." (distance: 0.15)
  3. "Types of ML include..." (distance: 0.23)
```

---

## 34. Adaptive Chunking

**What it is:**
Intelligent text splitting that creates 256-1024 token chunks based on document structure (sentences, paragraphs) with overlap.

**Why it matters:**
- **Preserves semantic units** (doesn't split mid-sentence)
- Context window optimization
- Better retrieval (complete thoughts)
- Overlap ensures no information loss

**How we use it:**
```python
# src/retrieval/chunker.py
class AdaptiveChunker:
    def __init__(
        self,
        min_chunk_size: int = 256,
        max_chunk_size: int = 1024,
        overlap: int = 128
    ):
        self.min_size = min_chunk_size
        self.max_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[ChunkMetadata]:
        # 1. Split into sentences
        sentences = sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Would adding this sentence exceed max?
            if current_tokens + sentence_tokens > self.max_size:
                # Save current chunk if above minimum
                if current_tokens >= self.min_size:
                    chunks.append(ChunkMetadata(
                        text=" ".join(current_chunk),
                        token_count=current_tokens,
                        chunk_id=len(chunks)
                    ))

                    # Start new chunk with overlap
                    overlap_sentences = self._get_overlap(current_chunk)
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = sum(self._count_tokens(s)
                                        for s in current_chunk)
                else:
                    # Add sentence even if it exceeds max
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        # Add final chunk
        if current_chunk and current_tokens >= self.min_size:
            chunks.append(ChunkMetadata(
                text=" ".join(current_chunk),
                token_count=current_tokens,
                chunk_id=len(chunks)
            ))

        return chunks
```

**Where to find it:**
- `src/retrieval/chunker.py`
- Called during `load` command
- Configured in `src/config.py`

**Visualization:**
```
Original text (2000 tokens):
[Sentence 1][Sentence 2][Sentence 3]...[Sentence 50]

Adaptive chunking (256-1024 tokens, 128 overlap):

Chunk 1: [S1][S2][S3][S4][S5]  (512 tokens)
                    └─ overlap ─┐
Chunk 2:                [S4][S5][S6][S7][S8]  (480 tokens)
                                    └─ overlap ─┐
Chunk 3:                             [S7][S8][S9][S10]  (520 tokens)

Benefits:
✅ No mid-sentence splits
✅ Overlap prevents information loss
✅ Size within optimal range
```

**Example:**
```
Input document (5000 tokens):

"Machine learning is a subset of AI. It enables computers to learn from data
 without explicit programming. There are three main types of machine learning..."

Chunking output:

Chunk 0 (512 tokens):
"Machine learning is a subset of AI. It enables computers to learn from data..."

Chunk 1 (487 tokens):  ← Overlaps with Chunk 0 last 128 tokens
"...learn from data without explicit programming. There are three main types..."

Chunk 2 (502 tokens):  ← Overlaps with Chunk 1 last 128 tokens
"...three main types of machine learning: supervised, unsupervised, and..."

Total: 10 chunks with smooth transitions
```

---

## 35. NLTK Tokenization

**What it is:**
Natural Language Toolkit library for splitting text into words and sentences.

**Why it matters:**
- **Foundation for BM25** (word tokenization)
- Sentence extraction (passage highlighting)
- Language-aware splitting
- Production-ready and tested

**How we use it:**
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required data (one-time setup)
nltk.download('punkt')
nltk.download('stopwords')

# Word tokenization (for BM25)
text = "Machine learning uses algorithms to learn from data."
words = word_tokenize(text.lower())
# Output: ['machine', 'learning', 'uses', 'algorithms', 'to', 'learn', 'from', 'data', '.']

# Sentence tokenization (for passage highlighting)
paragraph = "ML is powerful. It solves problems. It's widely used."
sentences = sent_tokenize(paragraph)
# Output: ['ML is powerful.', 'It solves problems.', "It's widely used."]
```

**Where to find it:**
- `src/retrieval/hybrid_search.py` - BM25 word tokenization
- `src/retrieval/passage_highlighter.py` - Sentence tokenization
- `src/retrieval/chunker.py` - Sentence-aware chunking

**Languages supported:**
- English (primary)
- Spanish, French, German, etc. (via language models)

---

## 36. JSON Data Persistence

**What it is:**
Store structured data (sources, metadata, configurations) in JSON format for easy loading/saving.

**Why it matters:**
- **Human-readable** format
- Easy debugging
- Cross-platform compatible
- No database overhead

**How we use it:**
```python
# src/persistence/__init__.py
import json
from pathlib import Path

class DataPersister:
    def save_sources(self, sources: List[SourceMetadata]):
        data = {
            "sources": [
                {
                    "name": source.name,
                    "type": source.type,
                    "loaded_at": source.loaded_at.isoformat(),
                    "chunk_count": source.chunk_count,
                    "config": source.config
                }
                for source in sources
            ],
            "version": "1.0"
        }

        with open("json_data/sources.json", "w") as f:
            json.dump(data, f, indent=2)

    def load_sources(self) -> List[SourceMetadata]:
        with open("json_data/sources.json", "r") as f:
            data = json.load(f)

        return [
            SourceMetadata(**source_data)
            for source_data in data["sources"]
        ]
```

**Where to find it:**
- `json_data/` folder - Persisted data
- `src/persistence/` - Save/load logic

**Example file structure:**
```json
{
  "sources": [
    {
      "name": "wikipedia",
      "type": "wikipedia_article",
      "loaded_at": "2024-01-15T10:30:00",
      "chunk_count": 42,
      "config": {
        "chunk_size": 512,
        "overlap": 128
      }
    }
  ],
  "version": "1.0"
}
```

---

## 37. Multi-Source Document Loading

**What it is:**
Load documents from multiple sources (Wikipedia, local files, URLs) into separate ChromaDB collections.

**Why it matters:**
- **Mixed knowledge bases**
- Source-specific weighting
- Easier debugging (know which source failed)
- Domain separation

**How we use it:**
```python
# src/retrieval/loader.py
class DocumentLoader:
    LOADERS = {
        "wikipedia": WikipediaLoader,
        "file": FileLoader,
        "url": URLLoader,
        "arxiv": ArxivLoader
    }

    def load(self, source_type: str, identifier: str):
        # 1. Get appropriate loader
        loader_class = self.LOADERS.get(source_type)
        if not loader_class:
            raise ValueError(f"Unknown source: {source_type}")

        # 2. Load documents
        loader = loader_class()
        documents = loader.load(identifier)

        # 3. Chunk
        chunks = self.chunker.chunk(" ".join(documents))

        # 4. Store in separate collection
        collection_name = f"{source_type}_{identifier}"
        collection = self.chroma_client.get_or_create_collection(collection_name)
        collection.add(
            documents=[c.text for c in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": source_type, "chunk_id": i}
                      for i in range(len(chunks))]
        )

        return collection_name
```

**Where to find it:**
- `src/retrieval/loader.py` - Multi-source support
- `chroma_db/` - Separate collections per source

**Example:**
```bash
> load wikipedia "Neural Networks"
✓ Loaded 35 chunks into collection 'wikipedia_neural_networks'

> load file "my_notes.txt"
✓ Loaded 12 chunks into collection 'file_my_notes'

> load url "https://example.com/ml-guide"
✓ Loaded 28 chunks into collection 'url_example_ml_guide'

> list
Loaded sources:
  1. wikipedia_neural_networks (35 chunks)
  2. file_my_notes (12 chunks)
  3. url_example_ml_guide (28 chunks)

> query What are activation functions?

Searching all 3 collections...
Results:
  [wikipedia_neural_networks] "Activation functions introduce non-linearity..." (0.08)
  [file_my_notes] "ReLU, sigmoid, tanh are common activations..." (0.12)
  [url_example_ml_guide] "Choosing the right activation function..." (0.18)
```

---

# PART 6: ENGINEERING PATTERNS (6 Techniques)

## 38. Orchestrator Pattern

**What it is:**
Central component (`RAGSystem`) that coordinates all other components (retrieval, generation, evaluation) in sequence.

**Why it matters:**
- **Single entry point** for queries
- Enforces pipeline sequence
- Easier testing (mock components)
- Clear separation of concerns

**How we use it:**
```python
# src/core/rag_system.py
class RAGSystem:
    def __init__(self):
        self.retriever = HybridSearchEngine()
        self.generator = LLMAnswerGenerator()
        self.evaluator = RAGASEvaluator()
        self.hallucination_detector = HallucinationDetector()
        self.config = Config()

    def process_query(self, query: str) -> RAGResponse:
        # STAGE 1: Retrieval
        documents = self.retriever.retrieve(query, n_results=50)

        # STAGE 2: Optional reranking
        if self.config.enable_reranking:
            documents = self.reranker.rerank(query, documents, top_k=5)

        # STAGE 3: Generation
        answer = self.generator.generate(query, documents)

        # STAGE 4: Evaluation
        ragas_metrics = self.evaluator.evaluate(query, answer, documents)
        hallucination_report = self.hallucination_detector.analyze(answer, documents)

        # STAGE 5: Response assembly
        return RAGResponse(
            query=query,
            answer=answer,
            sources=documents,
            ragas_metrics=ragas_metrics,
            hallucination_report=hallucination_report,
            confidence_score=self._calculate_confidence(ragas_metrics, hallucination_report)
        )
```

**Where to find it:**
- `src/core/rag_system.py` - Main orchestrator
- Called by CLI for every query

**Benefits:**
```
WITHOUT ORCHESTRATOR:
  CLI → Retriever
  CLI → Generator
  CLI → Evaluator
  ❌ CLI knows too much internal logic
  ❌ Hard to change pipeline

WITH ORCHESTRATOR:
  CLI → RAGSystem → Retriever → Generator → Evaluator
  ✅ CLI only knows RAGSystem
  ✅ Easy to modify pipeline
  ✅ Testable components
```

---

## 39. Strategy Pattern

**What it is:**
Define family of interchangeable algorithms (e.g., semantic vs keyword search) that can be swapped at runtime.

**Why it matters:**
- **Flexible retrieval strategies**
- Easy A/B testing
- Add new strategies without changing code
- Runtime configuration

**How we use it:**
```python
# src/retrieval/base.py
from abc import ABC, abstractmethod

class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str, n_results: int) -> List[Document]:
        pass

# Concrete strategies
class SemanticSearchStrategy(SearchStrategy):
    def search(self, query: str, n_results: int):
        return self.chroma_collection.query(query, n_results)

class KeywordSearchStrategy(SearchStrategy):
    def search(self, query: str, n_results: int):
        return self.bm25_index.get_top_n(query, n_results)

class HybridSearchStrategy(SearchStrategy):
    def __init__(self, semantic: SemanticSearchStrategy, keyword: KeywordSearchStrategy):
        self.semantic = semantic
        self.keyword = keyword

    def search(self, query: str, n_results: int):
        semantic_results = self.semantic.search(query, n_results * 2)
        keyword_results = self.keyword.search(query, n_results * 2)
        return self._fuse(semantic_results, keyword_results, n_results)

# Usage
class RAGSystem:
    def __init__(self, strategy: SearchStrategy):
        self.search_strategy = strategy  # ← Injected!

    def retrieve(self, query: str):
        return self.search_strategy.search(query, n_results=5)

# Runtime switching
if config.search_mode == "semantic":
    strategy = SemanticSearchStrategy()
elif config.search_mode == "keyword":
    strategy = KeywordSearchStrategy()
else:
    strategy = HybridSearchStrategy()  # Default

rag_system = RAGSystem(strategy)
```

**Where to find it:**
- `src/retrieval/base.py` - Strategy interfaces
- `src/retrieval/hybrid_search.py` - Concrete strategies
- `src/generation/answer_generator.py` - Generation strategies

---

## 40. Factory Pattern

**What it is:**
Create objects without specifying exact classes. Used for instantiating loaders, evaluators, etc.

**Why it matters:**
- **Decouples creation from usage**
- Easy to add new types
- Configuration-driven instantiation
- Cleaner code

**How we use it:**
```python
# src/factories.py (example pattern, not actual file)
class LoaderFactory:
    @staticmethod
    def create_loader(source_type: str) -> DocumentLoader:
        loaders = {
            "wikipedia": WikipediaLoader,
            "file": FileLoader,
            "url": URLLoader,
            "arxiv": ArxivLoader
        }

        loader_class = loaders.get(source_type)
        if not loader_class:
            raise ValueError(f"Unknown loader: {source_type}")

        return loader_class()

# Usage
loader = LoaderFactory.create_loader("wikipedia")
documents = loader.load("Machine Learning")

# Adding new loader type:
# 1. Create WikipediaLoaderV2 class
# 2. Add to loaders dict: "wikipedia_v2": WikipediaLoaderV2
# 3. Done! No changes to calling code
```

**Where to find it:**
- `src/retrieval/loader.py` - Loader factory pattern
- Implicit in many component instantiations

---

## 41. Abstract Base Classes (ABC)

**What it is:**
Python mechanism for defining interfaces that concrete classes must implement.

**Why it matters:**
- **Type safety** at development time
- Clear contracts between components
- IDE autocomplete support
- Prevents missing implementations

**How we use it:**
```python
# src/generation/answer_generator.py
from abc import ABC, abstractmethod

class AnswerGenerator(ABC):
    """Interface for all answer generators"""

    @abstractmethod
    def generate(self, query: str, context: List[Document]) -> str:
        """Generate answer from query and context"""
        pass

    @abstractmethod
    def stream_generate(self, query: str, context: List[Document]) -> Iterator[str]:
        """Stream tokens as they're generated"""
        pass

# Concrete implementation MUST implement all abstract methods
class LLMAnswerGenerator(AnswerGenerator):
    def generate(self, query: str, context: List[Document]) -> str:
        # Implementation
        pass

    def stream_generate(self, query: str, context: List[Document]) -> Iterator[str]:
        # Implementation
        pass

# If you forget to implement a method:
class BrokenGenerator(AnswerGenerator):
    def generate(self, query: str, context: List[Document]) -> str:
        pass
    # ❌ Missing stream_generate → TypeError at instantiation!
```

**Where to find it:**
- `src/generation/answer_generator.py` - Generator interface
- `src/retrieval/base.py` - Retrieval interfaces
- `src/evaluation/base.py` - Evaluator interfaces

**Benefits:**
```python
# Without ABC
def process(generator):  # What type? What methods?
    answer = generator.generate(query, docs)  # Hope this exists!

# With ABC
def process(generator: AnswerGenerator):  # ✅ Clear type
    answer = generator.generate(query, docs)  # ✅ IDE knows this method exists
```

---

## 42. Dataclasses

**What it is:**
Python decorator that auto-generates boilerplate code (`__init__`, `__repr__`, `__eq__`) for data-holding classes.

**Why it matters:**
- **Less boilerplate** (10 lines → 3 lines)
- Type hints integration
- Immutable option (frozen=True)
- Readable and maintainable

**How we use it:**
```python
# src/models/data_models.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RetrievedDocument:
    text: str
    source: str
    distance: float
    chunk_id: int
    metadata: Optional[dict] = None

# Equivalent manual implementation (15 lines):
# class RetrievedDocument:
#     def __init__(self, text: str, source: str, distance: float,
#                  chunk_id: int, metadata: Optional[dict] = None):
#         self.text = text
#         self.source = source
#         self.distance = distance
#         self.chunk_id = chunk_id
#         self.metadata = metadata
#
#     def __repr__(self):
#         return f"RetrievedDocument(text={self.text[:50]}..., source={self.source}, ...)"
#
#     def __eq__(self, other):
#         return (self.text == other.text and self.source == other.source and ...)

# Usage
doc = RetrievedDocument(
    text="Machine learning is...",
    source="wikipedia",
    distance=0.08,
    chunk_id=0
)

print(doc.source)  # "wikipedia"
print(doc)  # RetrievedDocument(text='Machine learning is...', source='wikipedia', ...)
```

**Where to find it:**
- `src/models/data_models.py` - All data models use dataclasses
- `RAGResponse`, `RAGASMetrics`, `HallucinationReport`, etc.

**Advanced features:**
```python
@dataclass(frozen=True)  # Immutable
class Config:
    model_name: str = "llama-3.1-8b"
    temperature: float = 0.2
    max_tokens: int = 1000

config = Config()
config.temperature = 0.5  # ❌ FrozenInstanceError (immutable!)

@dataclass
class RAGResponse:
    query: str
    answer: str
    confidence: float = 0.0  # Default value

response = RAGResponse(query="test", answer="test")
response.confidence  # 0.0 (used default)
```

---

## 43. Decorator Pattern (Configuration)

**What it is:**
Wrap functions/classes to add behavior (logging, timing, caching) without modifying original code.

**Why it matters:**
- **Cross-cutting concerns** (logging, metrics)
- Keep business logic clean
- Reusable decorators
- Easy to enable/disable features

**How we use it:**
```python
# Example decorator (not in actual codebase, but pattern is used)
import functools
import time
from src.utils.logger import logger

def timeit(func):
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

def cache_embeddings(func):
    """Decorator to cache embedding generation"""
    cache = {}

    @functools.wraps(func)
    def wrapper(text: str):
        if text in cache:
            logger.debug(f"Cache hit: {text[:50]}")
            return cache[text]

        result = func(text)
        cache[text] = result
        return result

    return wrapper

# Usage
@timeit
@cache_embeddings
def generate_embedding(text: str) -> List[float]:
    # Actual expensive computation
    return model.encode(text)

# Calling:
emb1 = generate_embedding("machine learning")  # Logs: "generate_embedding took 0.12s"
emb2 = generate_embedding("machine learning")  # Logs: "Cache hit: machine learning" (instant!)
```

**Where to find it:**
- Python's `@staticmethod`, `@property` decorators used throughout
- Caching logic in `src/retrieval/cache.py`
- Logging aspects across all modules

---

# PART 7: SAFETY & RELIABILITY (3 Techniques)

## 44. Domain Guard

**What it is:**
Validates queries are within system's knowledge domain before processing to prevent answering out-of-scope questions.

**Why it matters:**
- **Prevents confidently wrong answers**
- Saves compute on irrelevant queries
- Better user experience (clear boundaries)
- Safety for production systems

**How we use it:**
```python
# src/retrieval/domain_guard.py
class DomainGuard:
    def __init__(self, allowed_domains: List[str]):
        self.allowed_domains = set(allowed_domains)

    def is_in_domain(self, query: str) -> Tuple[bool, float, str]:
        # Check query against each domain
        scores = []
        for domain in self.allowed_domains:
            # Semantic similarity between query and domain description
            similarity = cosine_similarity(
                embed(query),
                embed(domain)
            )
            scores.append((domain, similarity))

        best_domain, best_score = max(scores, key=lambda x: x[1])

        if best_score >= 0.4:
            return (True, best_score, best_domain)
        else:
            return (False, best_score, "OUT_OF_DOMAIN")

# Usage in RAG system
def process_query(self, query: str) -> RAGResponse:
    # Domain check first
    if self.config.enable_domain_guard:
        in_domain, score, domain = self.domain_guard.is_in_domain(query)

        if not in_domain:
            return RAGResponse(
                query=query,
                answer=f"Sorry, this query is outside my knowledge domain. "
                       f"I can only answer questions about: {', '.join(self.allowed_domains)}",
                sources=[],
                error="OUT_OF_DOMAIN"
            )

    # Continue with normal pipeline
    return self._process_in_domain_query(query)
```

**Where to find it:**
- `src/retrieval/domain_guard.py`
- Optional feature (toggle in config)

**Example:**
```
Loaded sources: Wikipedia articles on "Machine Learning", "Neural Networks"

Query 1: "How does backpropagation work?"
Domain check: ✅ IN DOMAIN (similarity: 0.78 to "Machine Learning")
Processing...

Query 2: "What's the best Italian restaurant nearby?"
Domain check: ❌ OUT OF DOMAIN (similarity: 0.15 to any loaded domain)
Response: "Sorry, this query is outside my knowledge domain.
          I can only answer questions about: Machine Learning, Neural Networks"
```

---

## 45. Streaming Responses

**What it is:**
Generate and display LLM output token-by-token instead of waiting for complete response.

**Why it matters:**
- **Better UX** - See progress immediately
- Lower perceived latency
- Handle long responses gracefully
- Standard for production LLMs

**How we use it:**
```python
# src/generation/llm_answer_generator.py
def stream_generate(self, query: str, context: List[Document]) -> Iterator[str]:
    prompt = self._build_prompt(query, context)

    # Enable streaming in API call
    response = self.client.chat.completions.create(
        model=self.config.model_name,
        messages=prompt,
        temperature=self.config.temperature,
        max_tokens=self.config.max_tokens,
        stream=True  # ← Key parameter
    )

    # Yield tokens as they arrive
    for chunk in response:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            yield token

# CLI usage
answer_stream = generator.stream_generate(query, documents)

print("Answer: ", end="", flush=True)
full_answer = ""
for token in answer_stream:
    print(token, end="", flush=True)  # Display immediately
    full_answer += token
print()  # Newline at end
```

**Where to find it:**
- `src/generation/llm_answer_generator.py` - Stream implementation
- CLI uses streaming for all answers

**User experience:**
```
WITHOUT STREAMING:
> query What is machine learning?

[5 seconds of silence...]

Answer: "Machine learning is a subset of artificial intelligence that
enables computers to learn from data without explicit programming..."

❌ User waits 5 seconds staring at blank screen


WITH STREAMING:
> query What is machine learning?

Answer: Machine█  ← Appears in 200ms
Answer: Machine learning█  ← 400ms
Answer: Machine learning is a subset of█  ← 600ms
...
Answer: Machine learning is a subset of artificial intelligence that
enables computers to learn from data without explicit programming...

✅ User sees progress immediately!
```

**Performance:**
- Time to first token: ~200ms
- Tokens per second: ~15-30 (model dependent)
- Total time: Same as non-streaming, but **perceived** as faster

---

## 46. Adversarial Testing Suite

**What it is:**
Collection of challenging test queries designed to break the system and reveal edge cases.

**Why it matters:**
- **Find bugs before users do**
- Measure robustness
- Guide improvements
- Continuous quality monitoring

**How we use it:**
```python
# src/reasoning/adversarial_suite.py
class AdversarialTestSuite:
    def __init__(self):
        self.test_cases = [
            # Ambiguous queries
            AdversarialTest(
                query="What is it?",
                expected_behavior="Request clarification",
                category="AMBIGUITY"
            ),

            # Out-of-domain
            AdversarialTest(
                query="What's the weather today?",
                expected_behavior="Reject as out-of-domain",
                category="DOMAIN_VIOLATION"
            ),

            # Multi-hop reasoning required
            AdversarialTest(
                query="If A implies B and B implies C, does A imply C?",
                expected_behavior="Correct transitive reasoning",
                category="COMPLEX_REASONING"
            ),

            # Contradictory context
            Ad versarialTest(
                query="Is the sky blue?",
                injected_context=["The sky is green", "The sky is blue"],
                expected_behavior="Acknowledge contradiction or cite both",
                category="CONTRADICTION"
            ),

            # Hallucination trigger
            AdversarialTest(
                query="How many Nobel Prizes did Einstein win?",
                expected_behavior="Answer '1' not fabricate",
                category="FACTUAL_ACCURACY"
            )
        ]

    def run_suite(self) -> TestReport:
        results = []
        for test in self.test_cases:
            response = rag_system.process_query(test.query)
            passed = self._evaluate_response(response, test.expected_behavior)
            results.append(TestResult(
                test=test,
                response=response,
                passed=passed
            ))

        return TestReport(
            total=len(results),
            passed=sum(r.passed for r in results),
            failed=sum(not r.passed for r in results),
            results=results
        )
```

**Where to find it:**
- `src/reasoning/adversarial_suite.py`
- Run with `python -m src.reasoning.adversarial_suite`

**Example output:**
```
Running Adversarial Test Suite...

Test 1: Ambiguous Query
  Query: "What is it?"
  Response: "I need more context. What specific topic are you asking about?"
  Expected: Request clarification
  Result: ✅ PASS

Test 2: Out-of-Domain
  Query: "What's the weather today?"
  Response: "Sorry, this query is outside my knowledge domain..."
  Expected: Reject as out-of-domain
  Result: ✅ PASS

Test 3: Hallucination Trigger
  Query: "How many Nobel Prizes did Einstein win?"
  Response: "Einstein won the Nobel Prize in Physics in 1921. [Source: Wikipedia]"
  Expected: Answer '1' not fabricate
  Result: ✅ PASS (correctly stated 1 prize)

Test 4: Contradictory Context
  Query: "Is the sky blue?"
  Context: ["The sky is green", "The sky is blue"]
  Response: "The sources provide conflicting information about sky color."
  Expected: Acknowledge contradiction
  Result: ✅ PASS

========================================
Summary: 4/4 tests passed (100%)
========================================

Category Breakdown:
  AMBIGUITY: 1/1 ✅
  DOMAIN_VIOLATION: 1/1 ✅
  FACTUAL_ACCURACY: 1/1 ✅
  CONTRADICTION: 1/1 ✅
```

**Test categories:**
1. **Ambiguity** - Vague queries
2. **Domain violations** - Out-of-scope questions
3. **Factual accuracy** - Common misconceptions
4. **Contradictions** - Conflicting sources
5. **Complex reasoning** - Multi-step logic
6. **Edge cases** - Empty input, very long input, special characters
7. **Prompt injection** - Security attacks

---

# 🎓 Putting It All Together

## Complete Query Flow with All 46 Techniques

```
USER QUERY: "How does supervised learning work?"

┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: QUERY PROCESSING                                   │
├─────────────────────────────────────────────────────────────┤
│ #23 Query Expansion: Generate 4 variations                   │
│   1. "How does supervised learning work?"                    │
│   2. "Explain the supervised learning process"               │
│   3. "What is the mechanism of supervised ML?"               │
│   4. "How do supervised algorithms learn?"                   │
│                                                              │
│ #44 Domain Guard: Check if in scope                         │
│   Similarity to "Machine Learning": 0.85 ✅ IN DOMAIN       │
│                                                              │
│ #25 Self-Query Decomposition: Single aspect detected        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: RETRIEVAL (Two-Stage)                             │
├─────────────────────────────────────────────────────────────┤
│ #14 Stage 1: Fast retrieval (50 candidates)                │
│   #12 Hybrid Search:                                        │
│     #5  Semantic Search (ChromaDB):                         │
│       #3  Vector Embeddings (768-dim)                       │
│       #4  Cosine Similarity matching                        │
│       #16 Bi-Encoder (fast, 40ms)                          │
│       #21 LRU Cache (check for cached embeddings)          │
│     #13 BM25 Keyword Search:                                │
│       #9  Tokenization (NLTK word_tokenize)                │
│       Okapi BM25 scoring                                    │
│     #20 Batch Processing (all 4 variations together)       │
│   Retrieved: 50 documents                                   │
│                                                              │
│ #14 Stage 2: Accurate reranking (top 5)                    │
│   #15 Cross-Encoder Reranking:                             │
│       MS MARCO model jointly encodes query+doc             │
│       Batch size: 32 pairs (200ms)                         │
│   #17 MMR Diversity Filter:                                 │
│       λ=0.7 (balance relevance + diversity)                │
│       #18 Jaccard Similarity (compute diversity)           │
│   Final: 5 highly relevant, diverse documents              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: DOCUMENT PROCESSING                                │
├─────────────────────────────────────────────────────────────┤
│ #19 Passage Highlighting:                                   │
│   #9  Sentence Tokenization (NLTK sent_tokenize)           │
│   Calculate relevance scores per sentence                   │
│   Extract top 10 most relevant sentences                    │
│                                                              │
│ Documents retrieved from:                                    │
│   #33 ChromaDB Vector Database                             │
│   #34 Adaptive Chunking (512-token chunks, 128 overlap)    │
│   #37 Multi-Source Loading (Wikipedia + local files)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: ANSWER GENERATION                                  │
├─────────────────────────────────────────────────────────────┤
│ #1  RAG Pattern: context + query → LLM                     │
│ #6  Prompt Engineering: Build system + user prompts        │
│ #2  Large Language Model:                                   │
│     Model: Llama-3.1-8B Instruct                           │
│     #7  Temperature: 0.2 (factual)                         │
│     #8  Context Window: 8192 tokens (using ~2500)          │
│     #9  Tokenization: Process input tokens                 │
│     #45 Streaming: Token-by-token output                   │
│                                                              │
│ Generated answer:                                            │
│ "Supervised learning uses labeled training data where       │
│  inputs are paired with correct outputs. The model learns   │
│  patterns by minimizing prediction errors. [Source: Wiki]"  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: EVALUATION & QUALITY CONTROL                       │
├─────────────────────────────────────────────────────────────┤
│ #26 RAGAS Evaluation:                                       │
│   #27 Context Relevance: 0.95 (all docs relevant)         │
│   #28 Answer Relevance: 0.92 (on-topic)                   │
│   #29 Faithfulness: 0.98 (grounded in sources)            │
│       Context Precision: 0.90 (best docs at top)          │
│                                                              │
│ #30 Hallucination Detection:                                │
│   Claims extracted: 3                                       │
│   Grounded claims: 3/3 (100%)                              │
│   Risk level: LOW ✅                                        │
│                                                              │
│ #31 Fact Checking:                                          │
│   All facts verified against sources                        │
│   Verdict: PASS                                             │
│                                                              │
│ #32 Confidence Scoring:                                     │
│   Combined score: 94/100 (EXCELLENT)                       │
│   #10 Grounding: HIGH (all claims have sources)           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ STAGE 6: RESPONSE DELIVERY                                  │
├─────────────────────────────────────────────────────────────┤
│ #38 Orchestrator Pattern: RAGSystem coordinates all         │
│ #39 Strategy Pattern: Hybrid search strategy selected      │
│ #40 Factory Pattern: Components instantiated               │
│ #41 Abstract Base Classes: Type-safe interfaces            │
│ #42 Dataclasses: RAGResponse model                         │
│                                                              │
│ #43 Decorator Pattern: Logging, timing applied             │
│ #36 JSON Persistence: Save query logs                      │
│ #22 Graceful Degradation: All components available         │
└─────────────────────────────────────────────────────────────┘

RESPONSE TO USER:

Answer: "Supervised learning uses labeled training data where inputs
are paired with correct outputs. The model learns patterns by
minimizing prediction errors and generalizes to new data.
[Source: Wikipedia: Supervised Learning]"

Confidence: 94/100 (EXCELLENT)

Sources:
  1. [Wikipedia: Supervised Learning] (distance: 0.08)
  2. [Wikipedia: Training Data] (distance: 0.12)
  3. [Wikipedia: Machine Learning] (distance: 0.15)

RAGAS Metrics:
  Context Relevance: 0.95
  Answer Relevance: 0.92
  Faithfulness: 0.98
  Context Precision: 0.90

Hallucination Risk: LOW

Top Passages:
  1. "Supervised learning is the machine learning task of learning..."
  2. "Training data consists of input-output pairs..."
  3. "The goal is to approximate the mapping function..."
```

---

# 📚 Quick Reference Tables

## By Category

### Core ML/AI (11)
1. RAG, 2. LLMs, 3. Vector Embeddings, 4. Cosine Similarity, 5. Semantic Search, 6. Prompt Engineering, 7. Temperature, 8. Context Window, 9. Tokens, 10. Grounding, 11. Hallucination

### Search & Retrieval (11)
12. Hybrid Search, 13. BM25, 14. Two-Stage Retrieval, 15. Cross-Encoder, 16. Bi-Encoder, 17. MMR, 18. Jaccard Similarity, 19. Passage Highlighting, 20. Batch Processing, 21. LRU Cache, 22. Graceful Degradation

### Reasoning (3)
23. Query Expansion, 24. Multi-Hop Reasoning, 25. Self-Query Decomposition

### Evaluation (7)
26. RAGAS, 27. Context Relevance, 28. Answer Relevance, 29. Faithfulness, 30. Hallucination Detection, 31. Fact Checking, 32. Confidence Scoring

### Data & Storage (5)
33. ChromaDB, 34. Adaptive Chunking, 35. NLTK Tokenization, 36. JSON Persistence, 37. Multi-Source Loading

### Engineering (6)
38. Orchestrator Pattern, 39. Strategy Pattern, 40. Factory Pattern, 41. Abstract Base Classes, 42. Dataclasses, 43. Decorator Pattern

### Safety (3)
44. Domain Guard, 45. Streaming Responses, 46. Adversarial Testing

---

## By File Location

| Technique                           | File Path                                  |
| ----------------------------------- | ------------------------------------------ |
| #1 RAG                              | `src/core/rag_system.py`                   |
| #2 LLMs                             | `src/generation/llm_answer_generator.py`   |
| #3-5 Embeddings/Similarity/Semantic | `src/retrieval/hybrid_search.py`           |
| #6 Prompt Engineering               | `src/generation/llm_answer_generator.py`   |
| #7-9 Temperature/Context/Tokens     | `src/config.py`, `src/generation/`         |
| #10-11 Grounding/Hallucination      | `src/evaluation/hallucination_detector.py` |
| #12-13 Hybrid/BM25 Search           | `src/retrieval/hybrid_search.py`           |
| #14-15 Two-Stage/Cross-Encoder      | `src/retrieval/reranker.py`                |
| #16 Bi-Encoder                      | ChromaDB (implicit)                        |
| #17-18 MMR/Jaccard                  | `src/retrieval/reranker.py`                |
| #19 Passage Highlighting            | `src/retrieval/passage_highlighter.py`     |
| #20 Batch Processing                | Multiple files                             |
| #21 LRU Cache                       | `src/retrieval/cache.py`                   |
| #22 Graceful Degradation            | All component files                        |
| #23 Query Expansion                 | `src/reasoning/query_expander.py`          |
| #24 Multi-Hop                       | `src/reasoning/multi_hop_reasoner.py`      |
| #25 Self-Query                      | `src/reasoning/self_query_decomposer.py`   |
| #26-29 RAGAS Metrics                | `src/evaluation/ragas_evaluator.py`        |
| #30-31 Hallucination/Fact Check     | `src/evaluation/`                          |
| #32 Confidence Scoring              | `src/models/data_models.py`                |
| #33 ChromaDB                        | `src/retrieval/loader.py`                  |
| #34 Adaptive Chunking               | `src/retrieval/chunker.py`                 |
| #35 NLTK Tokenization               | `src/retrieval/`, `src/reasoning/`         |
| #36 JSON Persistence                | `src/persistence/`                         |
| #37 Multi-Source Loading            | `src/retrieval/loader.py`                  |
| #38-43 Engineering Patterns         | Throughout `src/`                          |
| #44 Domain Guard                    | `src/retrieval/domain_guard.py`            |
| #45 Streaming                       | `src/generation/llm_answer_generator.py`   |
| #46 Adversarial Testing             | `src/reasoning/adversarial_suite.py`       |

---

# 🎯 Next Steps for Learning

## Beginner Path (Week 1-2)
1. **Understand RAG basics** (#1): Read, run queries, see sources
2. **Explore LLMs** (#2): Change temperature, observe behavior
3. **Learn embeddings** (#3-5): Visualize similarity scores
4. **Practice prompting** (#6): Modify system prompts, see effects

## Intermediate Path (Week 3-4)
1. **Deep dive retrieval** (#12-19): Hybrid search, reranking, MMR
2. **Master evaluation** (#26-32): RAGAS metrics, tune thresholds
3. **Study reasoning** (#23-25): Query expansion, multi-hop
4. **Explore storage** (#33-37): ChromaDB internals, chunking strategies

## Advanced Path (Week 5-6)
1. **Engineering patterns** (#38-43): Refactor code, apply patterns
2. **Production hardening** (#44-46): Domain guards, adversarial tests
3. **Performance optimization**: Caching, batch processing, streaming
4. **Custom extensions**: Add new loaders, evaluators, strategies

## Hands-On Exercises

### Exercise 1: Modify Temperature
```bash
# Edit src/config.py: temperature = 0.8
# Run query, compare to temperature = 0.2
# Observe creativity vs. factuality trade-off
```

### Exercise 2: Add Custom Loader
```python
# Create src/retrieval/custom_loader.py
# Implement load_from_csv() method
# Register in LoaderFactory
# Load custom knowledge base
```

### Exercise 3: Tune MMR Lambda
```bash
# Set mmr_lambda = 0.3 (high diversity)
# Compare results to mmr_lambda = 0.9 (high relevance)
# Find optimal balance for your use case
```

### Exercise 4: Build Adversarial Test
```python
# Add test case to adversarial_suite.py
# Design query to break system
# Implement fix
# Verify test passes
```

---

# 📖 Further Reading

## Papers
- **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **RAGAS**: "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)
- **Sentence-BERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)

## Books
- "Designing Data-Intensive Applications" (Martin Kleppmann) - For engineering patterns
- "Natural Language Processing with Transformers" (Lewis Tunstall) - For LLM fundamentals

## Online Resources
- LangChain Documentation - Similar RAG framework
- Pinecone Vector Database Blog - Retrieval best practices
- OpenAI Cookbook - Prompt engineering guides

---

**Document Version:** 1.0
**Last Updated:** January 2024
**Techniques Covered:** 46/46 ✅
**Target Audience:** New AI engineers transitioning from software engineering

---

## Contributing

Found an error or want to add examples? This is a living document designed to help learners!

**How to contribute:**
1. Identify which technique needs improvement (#1-46)
2. Add practical examples from your experience
3. Include code snippets that worked for you
4. Share pitfalls you encountered

**Example contribution:**
```markdown
## #17 MMR - Additional Insight

When I tuned λ for my dataset:
- Medical papers: λ=0.9 (precision matters more)
- News articles: λ=0.5 (diversity prevents echo chamber)
- Code documentation: λ=0.7 (balanced)

Lesson: Domain affects optimal λ value!
```

Happy learning! 🚀

