# 🛠️ Portfolio Projects — Detailed Build Specs

> **Rule**: For every 🟢 MUST video you watch, spend **2x the time building**. These 6 projects ARE your resume.

> [!IMPORTANT]
> Projects **stack on each other**. By Project 6, you're deploying Projects 3+4+5 as one production system. Treat them as one evolving repo.

---

## Project 1: Professional Python Scaffold *(Module 1, Week 1–2)*

### What You're Building
A reusable project template that demonstrates modern Python best practices — the foundation every other project will use.

### Tech Stack
`Python 3.12` · `uv` · `python-dotenv` · `asyncio` · `mypy` · `ruff`

### Features to Build

| # | Feature | Details |
|---|---------|---------|
| 1 | **uv project setup** | `uv init`, `pyproject.toml`, lockfile, `.python-version` |
| 2 | **Type-hinted async code** | Write 3–4 async functions that call a free API (e.g., weather, jokes) with full type hints |
| 3 | **Pattern matching** | Use `match/case` to route different API response types |
| 4 | **Secrets management** | `.env` file for API keys, loaded via `python-dotenv`, never committed |
| 5 | **Linting + type checking** | `ruff` for linting, `mypy --strict` passes with zero errors |
| 6 | **Clean `.gitignore`** | Exclude `.env`, `__pycache__`, `.venv`, `uv.lock` |

### Folder Structure
```
ai-agent-scaffold/
├── pyproject.toml          # uv project config
├── .python-version         # Python 3.12
├── .env.example            # Template (no real keys)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── main.py             # Entry point with async main()
│   ├── api_client.py       # Async API calls with type hints
│   └── config.py           # Load .env, export settings
└── README.md
```

### Interview Talking Points
- *"I use uv instead of pip because it's 10–100x faster and handles Python versions + virtualenvs + dependencies in one tool"*
- *"Every function is type-hinted and passes mypy strict mode — this matters for PydanticAI later"*

---

## Project 2: Customer Support Triage Agent *(Module 2, Week 3–4)*

### What You're Building
A multi-agent customer support system that routes customer queries to the right specialist agent, with safety guardrails.

### Tech Stack
`PydanticAI` · `OpenAI Agents SDK` · `Pydantic V2` · `uv` · any LLM API (OpenAI / Gemini / Groq)

### Features to Build

| # | Feature | Details |
|---|---------|---------|
| 1 | **PydanticAI version** | Build a single agent with structured output: classifies queries into `billing`, `technical`, `sales` using Pydantic models |
| 2 | **Structured outputs** | Define response models: `TriageResult(category, urgency, summary, suggested_action)` |
| 3 | **Tool injection** | Agent has tools: `lookup_order(order_id)`, `check_account_status(email)` — return mock data |
| 4 | **OpenAI SDK rebuild** | Rebuild the same system using OpenAI Agents SDK with 3 specialist agents |
| 5 | **Handoffs** | Triage Agent → hands off to `BillingAgent`, `TechSupportAgent`, or `SalesAgent` |
| 6 | **Guardrails** | Input guardrail: block profanity/PII. Output guardrail: ensure response doesn't promise refunds without approval |
| 7 | **Compare both** | Write a short `COMPARISON.md` explaining PydanticAI vs SDK trade-offs you discovered |

### Folder Structure
```
customer-triage-agent/
├── pyproject.toml
├── .env.example
├── src/
│   ├── pydantic_version/
│   │   ├── agent.py         # PydanticAI agent
│   │   ├── models.py        # Pydantic response models
│   │   └── tools.py         # Tool functions
│   ├── sdk_version/
│   │   ├── triage_agent.py  # Router agent
│   │   ├── billing_agent.py
│   │   ├── tech_agent.py
│   │   ├── sales_agent.py
│   │   └── guardrails.py    # Input + output guardrails
│   └── run.py               # CLI to test both versions
├── COMPARISON.md
└── README.md
```

### Interview Talking Points
- *"I built the same system twice — once with PydanticAI, once with OpenAI SDK — so I can articulate trade-offs"*
- *"The guardrails prevent the agent from making financial commitments without human approval"*
- *"I chose PydanticAI for type-safety in production, SDK for rapid prototyping"*

---

## Project 3: Research & Report Pipeline *(Module 3, Week 5–6)*

### What You're Building
A multi-agent pipeline that takes a topic, researches it from the web, analyzes findings, and generates a structured report. This is the **core project** — Projects 4, 5, and 6 build on it.

### Tech Stack
`LangGraph` · `CrewAI` · `Tavily Search API` (free tier) · `PydanticAI` or SDK agents

### Features to Build

| # | Feature | Details |
|---|---------|---------|
| 1 | **LangGraph state graph** | Define a graph with 3 nodes: `research` → `analyze` → `write` |
| 2 | **Research Agent** | Uses Tavily Search API to find 5–10 relevant sources on a given topic |
| 3 | **Analyst Agent** | Takes raw search results, extracts key facts, identifies contradictions |
| 4 | **Writer Agent** | Generates a structured Markdown report with citations |
| 5 | **State persistence** | Use LangGraph checkpointing — pipeline can resume if interrupted |
| 6 | **Conditional routing** | If Research Agent finds < 3 sources, it retries with broader query |
| 7 | **CrewAI version (optional)** | Rebuild the same pipeline using CrewAI Flows for comparison |
| 8 | **Human-in-the-loop** | Add a breakpoint after `analyze` — user can approve/reject before writing |

### Folder Structure
```
research-pipeline/
├── pyproject.toml
├── .env.example
├── src/
│   ├── graph.py             # LangGraph state graph definition
│   ├── state.py             # State schema (Pydantic)
│   ├── agents/
│   │   ├── researcher.py    # Web search agent
│   │   ├── analyst.py       # Data processing agent
│   │   └── writer.py        # Report generation agent
│   ├── tools/
│   │   └── search.py        # Tavily search tool
│   └── run.py               # CLI entry point
├── output/                  # Generated reports go here
└── README.md
```

### Interview Talking Points
- *"I chose LangGraph over CrewAI because I needed fine-grained control over state and conditional routing"*
- *"The pipeline has checkpointing — if the LLM API times out mid-analysis, it resumes from where it stopped"*
- *"I added human-in-the-loop at the analysis stage because I learned from Reddit that fully autonomous agents fail silently"*

---

## Project 4: AI Knowledge Base with MCP + RAG *(Module 4, Week 7–8)*

### What You're Building
A personal knowledge base agent that answers questions using your own documents, powered by both vector search AND graph relationships.

### Tech Stack
`FastMCP` · `PostgreSQL + pgvector` · `Neo4j` · `Qdrant` (optional) · `LangGraph` (from Project 3)

### Features to Build

| # | Feature | Details |
|---|---------|---------|
| 1 | **MCP Server** | Build a FastMCP server exposing 3 tools: `search_docs`, `get_related`, `summarize` |
| 2 | **Document ingestion** | Script to chunk Markdown/PDF files, generate embeddings (OpenAI or local), store in pgvector |
| 3 | **Vector search** | `search_docs` tool queries pgvector using cosine similarity, returns top-5 chunks |
| 4 | **Graph relationships** | Store document → section → concept relationships in Neo4j |
| 5 | **Graph-enhanced RAG** | When user asks a question: vector search finds relevant chunks + Neo4j finds related concepts = richer context |
| 6 | **Agent integration** | Connect the MCP server to your Project 3 agent system — it now has "memory" |
| 7 | **Hybrid retrieval** | Combine vector similarity + graph traversal results before sending to LLM |

### Folder Structure
```
knowledge-base-agent/
├── pyproject.toml
├── .env.example
├── docker-compose.yml       # PostgreSQL + Neo4j containers
├── src/
│   ├── mcp_server/
│   │   ├── server.py        # FastMCP server definition
│   │   ├── tools.py         # search_docs, get_related, summarize
│   │   └── config.py
│   ├── ingestion/
│   │   ├── chunker.py       # Split docs into chunks
│   │   ├── embedder.py      # Generate embeddings
│   │   └── graph_builder.py # Build Neo4j relationships
│   ├── retrieval/
│   │   ├── vector_search.py # pgvector queries
│   │   └── graph_search.py  # Neo4j traversal
│   └── run.py
├── data/                    # Sample docs to ingest
└── README.md
```

### Interview Talking Points
- *"I built an MCP server so any MCP-compatible client (Claude, Cursor, etc.) can use my knowledge base"*
- *"Pure vector search missed relationships between concepts — adding Neo4j increased answer relevance by ~30% in my tests"*
- *"I used pgvector instead of a dedicated vector DB because my data fits in PostgreSQL and I avoid adding infrastructure"*

---

## Project 5: Observability & Eval Suite *(Module 5, Week 9–10)*

### What You're Building
Add production-grade monitoring and automated testing to your Project 3 pipeline. This turns a "demo" into a "production-ready system."

### Tech Stack
`LangSmith` · `Evals (LLM-as-Judge)` · project 3 pipeline · `pytest`

### Features to Build

| # | Feature | Details |
|---|---------|---------|
| 1 | **LangSmith integration** | Trace every LLM call, tool invocation, and handoff in Project 3 pipeline |
| 2 | **Custom trace metadata** | Tag traces with: `run_id`, `topic`, `model`, `total_cost`, `latency` |
| 3 | **Golden dataset** | Create 20 test cases: `{topic, expected_sections, expected_facts, quality_criteria}` |
| 4 | **LLM-as-Judge eval** | For each generated report, a judge LLM scores: `factual_accuracy`, `completeness`, `coherence`, `citation_quality` |
| 5 | **Scoring rubric** | Use named categories not numbers: `excellent`, `acceptable`, `needs_improvement`, `failed` |
| 6 | **Regression tests** | `pytest` suite that runs 5 core test cases, asserts quality scores ≥ `acceptable` |
| 7 | **Dashboard metrics** | Track across runs: avg latency, cost/run, success rate, common failure categories |

### Folder Structure
```
# Add to your research-pipeline/ project:
research-pipeline/
├── ...existing files...
├── evals/
│   ├── golden_dataset.json  # 20 test cases
│   ├── judge.py             # LLM-as-Judge scoring
│   ├── rubric.py            # Named category rubrics
│   └── run_evals.py         # Run all evals, output report
├── tests/
│   └── test_pipeline.py     # pytest regression tests
└── observability/
    ├── tracing.py           # LangSmith setup + custom metadata
    └── dashboard.py         # Metrics aggregation
```

### Interview Talking Points
- *"I test my agents the same way you'd test software — golden datasets, regression suites, and automated scoring"*
- *"I use named categories instead of 1-10 scales because Reddit discussions showed LLMs are better at categorical judgments"*
- *"LangSmith tracing helped me find that 40% of my latency was in the research step — I optimized by caching search results"*

---

## Project 6: Production Deployment *(Module 6, Week 11–12)*

### What You're Building
Deploy your full agent system (Projects 3+4+5) as a production API with caching, containerization, and cloud hosting. **This is your capstone.**

### Tech Stack
`FastAPI` · `Docker + Docker Compose` · `Redis` · `Modal` or `Cloud Run` · GitHub Actions CI/CD

### Features to Build

| # | Feature | Details |
|---|---------|---------|
| 1 | **FastAPI wrapper** | REST API: `POST /research` (start pipeline), `GET /status/{run_id}`, `GET /report/{run_id}` |
| 2 | **WebSocket streaming** | Stream agent progress to client in real-time: "Researching… 3/10 sources found" |
| 3 | **Redis caching** | Cache search results (TTL: 1hr) + cache completed reports (TTL: 24hr) |
| 4 | **Redis agent memory** | Store conversation history per session in Redis |
| 5 | **Dockerize everything** | `Dockerfile` for the app + `docker-compose.yml` with FastAPI + Redis + PostgreSQL + Neo4j |
| 6 | **Health checks** | `/health` endpoint checking all services (Redis, DB, LLM API connectivity) |
| 7 | **Deploy to cloud** | Pick one: Modal (for GPU) or Cloud Run (for API). Deploy with one command |
| 8 | **CI/CD** | GitHub Actions: on push → run evals → if pass → deploy |
| 9 | **README as portfolio** | Screenshot/GIF of the API working, architecture diagram, tech stack badges |

### Folder Structure
```
ai-agent-production/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml       # FastAPI + Redis + PostgreSQL + Neo4j
├── .github/
│   └── workflows/
│       └── deploy.yml       # CI/CD pipeline
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app
│   │   ├── routes.py        # /research, /status, /report
│   │   ├── websocket.py     # Real-time streaming
│   │   └── middleware.py    # Auth, rate limiting, CORS
│   ├── cache/
│   │   └── redis_client.py  # Redis caching + session memory
│   ├── agents/              # From Project 3
│   ├── mcp_server/          # From Project 4
│   ├── evals/               # From Project 5
│   └── config.py
├── deploy/
│   ├── modal_deploy.py      # Modal deployment script
│   └── cloudrun.yaml        # Cloud Run config
└── README.md                # ⭐ This IS your portfolio piece
```

### Interview Talking Points
- *"I deployed a multi-agent system with 4 services (API + Redis + Postgres + Neo4j) using Docker Compose, then pushed to Cloud Run"*
- *"Redis semantic caching cut my LLM costs by 35% by avoiding duplicate queries"*
- *"I have CI/CD — every push runs my eval suite and only deploys if all 20 test cases pass"*
- *"The API streams agent progress via WebSocket so users see what's happening, not just a loading spinner"*

---

## 🗺️ How the projects connect

```
Project 1 (scaffold) ──sets up──▶ Every other project

Project 2 (triage agent) ──teaches you──▶ Agent patterns for Projects 3–6

Project 3 (pipeline) ──is enhanced by──▶ Project 4 (knowledge base)
                      ──is tested by──▶ Project 5 (evals)
                      ──is deployed by──▶ Project 6 (production)

Project 6 = Project 3 + 4 + 5 deployed together 🚀
```

> [!TIP]
> **GitHub tip**: Create ONE repo called `ai-agent-system` and evolve it across all 6 projects. Interviewers love seeing a commit history that shows progressive learning.
