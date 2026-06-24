# Branch: `project/agent-rag` — Agentic RAG

This branch builds a full **Agentic RAG** system — an agent that dynamically decides *when* and *how* to retrieve documentation before answering questions. It crawls the LangChain documentation site, embeds the content into a Pinecone vector store, and serves it through an intelligent agent.

---

## Branches

**1. `project/hello-world`**
The starting point. Sets up the project, installs dependencies, and runs a basic LangChain agent with a mock search tool. No real API calls — just enough to verify everything works and understand the basic structure of an agent.

**2. `project/search-agent`**
Builds a real search agent using `create_agent()` with Tavily for live web search. Introduces structured output via Pydantic models (`AgentResponse`, `Source`). This is Layer 0 — LangChain and LangGraph handle the entire agent loop for you.

**3. `projects/agents-under-the-hood`**
Breaks open the black box. Shows how the agent loop works internally by implementing it manually, without relying on `create_react_agent()`. Contains two files covering Layer 1 in two ways — once with LangChain primitives, once with raw Ollama.

**4. `projects/rag-gist`**
Introduces Retrieval-Augmented Generation from first principles. Covers document loading, chunking strategies, embeddings, and vector stores. Shows three RAG implementations: raw LLM, manual RAG without LCEL, and a full LCEL pipeline.

**5. `project/agent-rag` ← you are here**
Combines agents and RAG. An LLM-powered agent dynamically decides when and how to retrieve from a Pinecone vector store populated with LangChain documentation. Uses `@tool` with `response_format="content_and_artifact"` for structured retrieval.

---

## What Is RAG?

RAG (Retrieval-Augmented Generation) extends an LLM's knowledge by pulling relevant documents from a vector store at query time, so the model can answer questions beyond its training data.

---

## RAG Architectures

There are three main ways to implement RAG — this branch uses **Agentic RAG**.

| Architecture | Description | Control | Flexibility | Latency |
|---|---|---|---|---|
| **2-Step RAG** | Retrieval always happens before generation. Simple and predictable. | High | Low | Fast |
| **Agentic RAG** | An LLM-powered agent decides *when* and *how* to retrieve during reasoning. | Low | High | Variable |
| **Hybrid** | Combines both approaches with validation steps. | Medium | Medium | Variable |

### 2-Step RAG (Classic)
```
User Query → Retriever → LLM → Answer
```
Fixed pipeline. Retrieval always runs, no matter what. Great for simple Q&A.

### Agentic RAG (This Branch)
```
User Query → Agent → [thinks] → calls retrieve_context() tool → [reads docs] → Answer
```
The agent decides **whether** to retrieve, **what to search for**, and can call the tool **multiple times** with different queries before producing a final answer.

---

## Architecture of This Branch

```
                    ┌─────────────────────────────────┐
                    │         ingestion.py             │
                    │                                  │
  LangChain Docs ──►│  TavilyCrawl → Chunking          │──► Pinecone
  (python.langchain │  (27 pages → 57 chunks)          │    (langchain-docs-2026)
   .com)            └─────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │           main.py               │
                    │                                  │
  User Query ──────►│  create_agent()                  │
                    │    └─ gpt-4o-mini                │
                    │    └─ retrieve_context() tool     │──► Pinecone
                    │         └─ similarity search      │◄── (top 4 docs)
                    └─────────────────────────────────┘
                                  │
                              Final Answer
                          (with source citations)
```

---

## Files

| File | Purpose |
|---|---|
| `ingestion.py` | Crawls LangChain docs, chunks them, and stores embeddings in Pinecone |
| `main.py` | Runs the Agentic RAG — agent + retrieval tool |
| `logger.py` | Colored terminal logging utilities |
| `consts.py` | Shared constants (index name) |
| `pyproject.toml` | uv dependency file (converted from Pipfile) |

---

## ingestion.py — The Ingestion Pipeline

The ingestion pipeline has three phases:

### Phase 1: Crawl
Uses **Tavily Crawl** to scrape `https://python.langchain.com/` — no manual HTML cleaning needed.

```python
tavily_crawl = TavilyCrawl()
res = tavily_crawl.invoke({
    "url": "https://python.langchain.com/",
    "max_depth": 2,
    "extract_depth": "advanced",
})
# Result: 27 pages of clean markdown content
```

### Phase 2: Chunk
Splits each page into overlapping chunks for better retrieval accuracy.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
splitted_docs = text_splitter.split_documents(all_docs)
# Result: 57 chunks from 27 documents
```

### Phase 3: Embed & Store
Converts chunks into 1536-dimensional vectors and stores them in Pinecone.

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index_name="langchain-docs-2026", embedding=embeddings)
await vectorstore.aadd_documents(splitted_docs)
```

**Why async?** Documents are uploaded in parallel batches using `asyncio.gather()`, making ingestion faster for large document sets.

---

## main.py — The Agentic RAG

### The Retrieval Tool

The tool uses `response_format="content_and_artifact"` — it returns **two things** at once:
- A **string summary** (what the model reads)
- The **raw Document objects** (stored as an artifact for structured access)

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs  # (content, artifact)
```

### The Agent

```python
model = init_chat_model("gpt-4o-mini", model_provider="openai")

agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)
response = agent.invoke({"messages": [{"role": "user", "content": query}]})
```

The agent runs a ReAct loop internally:
1. Receives the user query
2. Calls `retrieve_context()` with a search query
3. Reads the returned documentation
4. Produces a final answer with source citations

---

## `response_format` Explained

| Value | Returns | Use case |
|---|---|---|
| `"content"` (default) | A plain string | Simple tool responses |
| `"content_and_artifact"` | A tuple `(string, any)` | When you need both a readable summary AND the raw data (e.g. Document objects) |

The `artifact` is stored in the `ToolMessage` and accessible programmatically after the agent run.

---

## Setup & Run

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) installed
- `.env` file with required keys

### `.env` file

```env
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
TAVILY_API_KEY=your-tavily-key
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agent-rag
```

### Install dependencies

```bash
uv sync
```

### Step 1 — Create Pinecone index

Create an index named `langchain-docs-2026` with **1536 dimensions** and **cosine** metric in your Pinecone dashboard, or programmatically:

```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="your-key")
pc.create_index(
    name="langchain-docs-2026",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### Step 2 — Run ingestion

```bash
uv run python ingestion.py
```

Expected output:
```
DOCUMENTATION INGESTION PIPELINE
TavilyCrawl: Successfully crawled https://python.langchain.com/ [27 pages]
...
DOCUMENT CHUNKING PHASE
Text Splitter: Created 57 chunks from 27 documents

VECTOR STORAGE PHASE
VectorStore Indexing: All batches processed successfully! (1/1)

PIPELINE COMPLETE
Documentation ingestion pipeline finished successfully!
```

### Step 3 — Run the agent

```bash
uv run python main.py
```

Example query and response:
```
Query: "what are deep agents?"

Answer: Deep Agents are a "batteries-included" agent type in LangChain with built-in
features like automatic context compression, a virtual filesystem, and subagent-spawning.
They are built on top of LangChain agents (which themselves run on LangGraph).
Source: https://python.langchain.com/oss/python/langchain/quickstart
```

---

## Key Concepts

### Why Agentic RAG over 2-Step RAG?

| Scenario | 2-Step RAG | Agentic RAG |
|---|---|---|
| Simple factual Q&A | Works well | Overkill |
| Multi-hop questions (requires chaining lookups) | Fails — only one retrieval pass | Works — agent calls tool multiple times |
| Deciding whether retrieval is needed | Always retrieves | Skips retrieval for simple questions |
| Searching with refined queries | Uses original user query | Can reformulate the search query |

### Why Tavily for crawling?

Tavily returns **clean markdown content** from web pages without needing to handle raw HTML, CSS, or JavaScript parsing. It also handles JavaScript-rendered pages that tools like `requests` + `BeautifulSoup` would miss.

### Why `text-embedding-3-small`?

- 1536-dimensional vectors
- Best cost-to-performance ratio for RAG use cases
- OpenAI's recommended embedding model for retrieval

### Why chunk with overlap (`chunk_overlap=200`)?

Overlap ensures that sentences or concepts that span a chunk boundary are not lost. Without overlap, a retrieval query might miss context that sits right at the edge of two chunks.

---

## LangSmith Tracing

Because `LANGSMITH_TRACING=true` is set, every agent run is automatically traced. You can inspect:
- Which tool calls were made
- What the retrieval returned
- The full message history of the agent loop

Go to [smith.langchain.com](https://smith.langchain.com) → project `agent-rag` to see traces.
