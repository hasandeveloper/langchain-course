# Branch: `project/data-policy` — Using Managed LLMs: Privacy, Data Retention & EULA

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

**5. `project/agent-rag`**
Combines agents and RAG. An LLM-powered agent dynamically decides when and how to retrieve from a Pinecone vector store populated with LangChain documentation. Uses `@tool` with `response_format="content_and_artifact"` for structured retrieval.

**6. `project/data-policy` ← you are here**
Covers privacy, data retention, and EULA considerations when using managed LLMs (OpenAI, Anthropic, Google) vs self-managed LLMs (Ollama). Essential reading before taking any LLM application to production.

---

When taking an LLM application to production, understanding what happens to your data is critical. This branch covers the key differences between **managed LLMs** (cloud APIs) and **self-managed LLMs** (local models) from a privacy, compliance, and data governance perspective.

---

## The Core Question

Every time you send a prompt to a cloud LLM API:

```
Your App → [your data travels over the internet] → OpenAI / Anthropic / Google servers
                                                          ↓
                                                   Model processes it
                                                          ↓
                                                   Response sent back
```

That data may be **logged, retained, and used for model training** depending on the provider's terms.

---

## Managed LLMs (OpenAI, Anthropic, Google)

You call an API. The model runs on someone else's infrastructure. Fast, powerful, and easy to use — but your data leaves your environment.

### Data Retention

| Provider | Default Retention | Zero Retention Option |
|---|---|---|
| OpenAI | 30 days | Yes — Zero Data Retention (ZDR), enterprise plans only |
| Anthropic | 30 days | Yes — enterprise agreement required |
| Google Gemini | Up to 30 days | Yes — Vertex AI with a Data Processing Addendum (DPA) |

By default, most providers retain your API inputs and outputs for up to 30 days for abuse monitoring and debugging.

### Privacy Risks

- **PII (Personally Identifiable Information)** — names, emails, phone numbers sent in prompts are processed by a third party
- **Trade secrets** — internal documents used in RAG pipelines are sent to external servers
- **User conversations** — chat history is stored on the provider's infrastructure
- **Model training** — some providers may use your prompts to improve their models unless you explicitly opt out

### EULA Restrictions

- Sending **medical data (PHI)** to a managed LLM without a HIPAA Business Associate Agreement (BAA) violates HIPAA
- Sending **classified or export-controlled data** to any cloud API is typically prohibited
- You are responsible for ensuring compliance with **GDPR**, **HIPAA**, **SOC2**, and any other applicable regulations based on the data you send
- Provider terms of service can change — what is allowed today may not be tomorrow

---

## Self-Managed LLMs (Ollama, local models)

You download and run the model on your own hardware or private cloud. Data **never leaves your environment**.

```bash
# Run a model locally — no API call, no data transfer
ollama run llama3.2

# Use with LangChain
from langchain.chat_models import init_chat_model
model = init_chat_model("ollama:llama3.2")
```

### Benefits

- **Full data sovereignty** — zero third-party exposure, ever
- **No EULA restrictions** — open-source models (Llama, Mistral, Qwen) have permissive licenses
- **GDPR / HIPAA compliant by default** — no data transfer means no cross-border data flow issues
- **No per-token cost** — you pay for hardware, not API calls
- **Predictable latency** — no network round-trip to an external server

### Trade-offs

- **Less capable models** — local models (7B–70B parameters) are generally weaker than GPT-4o or Claude for complex reasoning
- **Hardware requirements** — needs a GPU for acceptable performance (Apple Silicon M-series works well)
- **You manage everything** — model updates, security patches, uptime, and scaling are your responsibility

---

## Side-by-Side Comparison

| | Managed LLM (OpenAI) | Self-Managed LLM (Ollama) |
|---|---|---|
| **Data location** | Third-party servers | Your machine / private infra |
| **Data retention** | Up to 30 days (default) | None — never stored externally |
| **Model training on your data** | Possible unless opted out | Never |
| **GDPR compliance** | Requires DPA with provider | Compliant by default |
| **HIPAA compliance** | Requires BAA with provider | Compliant by default |
| **Cost** | Per token (usage-based) | Hardware cost only |
| **Model capability** | State of the art | Good but smaller |
| **Setup complexity** | Simple (API key) | Moderate (install + hardware) |
| **Internet required** | Yes | No |

---

## Decision Guide

| Your situation | Recommendation |
|---|---|
| Learning / prototyping | Managed (OpenAI) — fast, easy, best models |
| Handling user PII | Self-managed, or managed with ZDR enterprise plan |
| Healthcare app with patient data (HIPAA) | Self-managed, or managed with a signed HIPAA BAA |
| Internal company documents in RAG | Self-managed, or ZDR enterprise plan |
| Financial / legal regulated industry | Self-managed or legal review of provider terms |
| Production app, no sensitive data | Managed with data retention opt-out enabled |
| Fully air-gapped environment | Self-managed only |

---

## How This Applies to This Course

In the `project/agent-rag` branch, `ingestion.py` sends **LangChain documentation** (publicly available data) to OpenAI for embedding — this is safe and appropriate.

But if you were building a real product and ingesting **internal company documents** or **customer data**, you would need to choose:

**Option A — Stay managed, get ZDR:**
```python
# OpenAI ZDR — requires enterprise agreement
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# Works the same in code, but your data is not retained by OpenAI
```

**Option B — Switch to local embeddings:**
```python
# Ollama — runs entirely on your machine, zero data leaves
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
```

Option B requires no enterprise agreement and works offline — but you need Ollama installed and the model downloaded (`ollama pull nomic-embed-text`).

---

## Key Takeaway

> **The model doesn't care about your data. Your legal and compliance team does.**

Choosing between managed and self-managed LLMs is not a technical decision — it is a **data governance decision**. Match your infrastructure to the sensitivity of the data you handle.
