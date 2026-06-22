# Agents Under the Hood

This branch shows how an AI agent works internally, without relying on high-level abstractions.

## Architecture

![Architecture](Architecture%20of%20this%20branch.png)

- **Layer 0** — High-level: `create_agent()`, `TavilySearch` — LangChain handles everything
- **Layer 1** — Mid-level: manual agent loop using LangChain primitives (`@tool`, `bind_tools`, `ToolMessage`)
- **Layer 2** — Low-level: raw ReAct prompt, regex, scratchpad — no framework at all

---

## ReAct Loop

![ReAct Loop](image.png)

Every iteration of the agent loop follows this pattern:

1. LLM receives the question and decides which tool to call
2. The tool executes and returns a result
3. The result is added back to the message history
4. LLM is called again with the updated context
5. Repeat until the LLM returns a final answer (no more tool calls)

---

## File 1 — `1.agent_loop_langchain_tool_calling.py` (Layer 1 — LangChain)

**Uses: LangChain + OpenAI**

Manually implements the agent loop using LangChain abstractions:

- `@tool` decorator — automatically generates the JSON schema for each function from its type hints and docstring
- `bind_tools()` — tells the LLM which tools are available
- `init_chat_model()` — initializes the LLM
- `ToolMessage` — feeds the tool result back into the message history
- Messages are LangChain objects (`SystemMessage`, `HumanMessage`, `ToolMessage`)
- Tool calls come back as dicts — accessed with `.get("name")`, `.get("args")`

```
Question → LLM decides → tool.invoke(args) → ToolMessage → LLM again → Final Answer
```

---

## File 2 — `2_agent_loop_raw_function_calling.py` (Layer 1 — RAW)

**Uses: Ollama directly (no LangChain)**

This is still **Layer 1** — the exact same agent loop as File 1, but with zero LangChain. It shows exactly what `@tool` and `bind_tools` were hiding under the hood:

- **No `@tool`** — JSON schema for each function is written manually
- **No `bind_tools()`** — tools are passed directly to `ollama.chat()`
- **No `ToolMessage`** — tool result is appended as a plain dict `{"role": "tool", "content": "..."}`
- **No `.invoke()`** — tools are called directly as regular Python functions: `tool_to_use(**tool_args)`
- Tool calls come back as objects — accessed with `.function.name`, `.function.arguments`

```
Question → ollama.chat() → tool(**args) → {"role": "tool"} → ollama.chat() → Final Answer
```

### Key Insights for File 2

**Messages are plain dicts, not LangChain objects.**
File 2 uses `{"role": "system", "content": ...}` instead of `SystemMessage(content=...)`. This format works for Ollama/OpenAI but is provider-specific — if you switch providers, the structure may break. LangChain's message types (`SystemMessage`, `HumanMessage`, `ToolMessage`) provide a universal format that works across all providers.

**Tools are called directly with `**tool_args`, not `.invoke()`.**
`tool_to_use.invoke(tool_args)` in LangChain adds validation, tracing, and error handling around the call. `tool_to_use(**tool_args)` is a plain Python function call — none of that happens automatically.

**Tool call access uses attribute access, not dict access.**
LangChain normalizes tool calls into dicts so you use `.get("name")`. Ollama returns typed objects so you must use `.function.name` and `.function.arguments`. If you switched providers in File 2, this code would break — LangChain hides this difference from you.

**No `tool_call_id` needed for Ollama.**
LangChain's `ToolMessage` requires a `tool_call_id` to match tool results back to the correct call (OpenAI strictly enforces this). Ollama's local models don't require it, so File 2 appends tool results as `{"role": "tool", "content": ...}` with no ID. This is another provider-specific detail LangChain handles automatically.

**Tracing must be done manually.**
Without LangChain, you must decorate every function with `@traceable(run_type=...)` to get LangSmith observability. With LangChain, `init_chat_model()` and `@tool` report traces automatically — no extra decoration needed.

**What stays the same, what changes.**
The agent loop pattern is identical in both files — iterate, check for tool calls, execute the tool, append the result, repeat. What changes is everything around it: schemas, message formatting, response parsing, tool invocation, and tracing all must be handled manually in File 2.

---

## Key Differences Between File 1 and File 2

| | File 1 (LangChain) | File 2 (Raw Ollama) |
|---|---|---|
| Tool definition | `@tool` decorator | Manual JSON schema |
| LLM call | `llm_with_tools.invoke()` | `ollama.chat()` |
| Tool execution | `tool.invoke(args)` | `tool(**args)` |
| Tool result | `ToolMessage(...)` | `{"role": "tool", ...}` |
| Tool call access | `.get("name")` (dict) | `.function.name` (object) |
| Tracing | automatic via LangSmith | manual `@traceable` |

---

## Resources

- [Ollama Docs](https://docs.ollama.com/) — how to run local LLMs, API reference, Python & JavaScript libraries
- [Ollama API Reference](https://docs.ollama.com/) — used in File 2 via `ollama.chat()` to call the model directly without LangChain

---

## Shortcut — Layer 0

If you don't need to understand the internals, `create_react_agent` from `langgraph.prebuilt` handles the entire loop in one line:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [HumanMessage(content="...")]})
```

This replaces the entire manual `run_agent()` loop — tool dispatch, message building, iteration tracking — all handled internally. The manual loop in this branch exists purely to show what's happening inside that one line.
