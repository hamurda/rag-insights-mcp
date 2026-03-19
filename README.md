# Unanswered Questions MCP Server

MCP server that tracks questions a RAG chatbot can't answer, clusters them
by semantic similarity, and suggests what documentation to add.

Built to plug into a multi-tenant RAG chatbot in production. When the chatbot
gives a low-confidence answer, it logs the question here. Run pattern analysis
periodically to find out what docs are missing.

**Article**: [Build an MCP Server That Finds Your RAG Chatbot's Blind Spots](https://dev.to/hamurda/build-an-mcp-server-that-finds-your-rag-chatbots-blind-spots-3hb1)

## How It Works

```
User question → RAG chatbot → low confidence?
                                    │
                    log_unanswered_question
                          │
                     SQLite + embedding
                          │
              get_question_patterns (weekly)
                          │
                 clusters by similarity
                          │
               suggest_documents → write docs → mark_resolved
```

**4 tools exposed over MCP:**

| Tool                      | What it does                                |
|---------------------------|---------------------------------------------|
| `log_unanswered_question` | Store a question + its embedding            |
| `get_question_patterns`   | Cluster unresolved questions, return topics |
| `suggest_documents`       | AI-generated doc outline for a topic        |
| `mark_resolved`           | Close the loop after adding documentation   |

## Quick Start

```bash
uv sync
cp .env.example .env   # add your OPENAI_API_KEY
python test_server.py # Run the test suite (needs API key)
python server.py # Start the MCP server
```

### Claude Desktop config

```json
{
  "mcpServers": {
    "unanswered-questions": {
      "command": "python",
      "args": ["/path/to/unanswered-questions-mcp/server.py"]
    }
  }
}
```

## Why It's Built This Way

**SQLite over a vector database** — Batch analysis, not real-time retrieval. Zero-config and handles 10K+ questions fine.

**Greedy cosine-similarity clustering** — Embeddings via `text-embedding-3-small`, cosine similarity matrix, then greedy 
assignment above a configurable threshold.

**Multi-tenant from day one** — Every operation takes an optional `tenant_id`. Matches the production RAG chatbot it plugs into.


## Limitations

- **Batch, not real-time** — Pattern analysis is meant to run periodically, not as a streaming pipeline.
- **Greedy clustering is order-dependent** — Results can vary slightly across runs. Fine for broad patterns, not precise categorization.
- **No auth** — Relies on the MCP client handling access. Add authentication if exposing over HTTP.
