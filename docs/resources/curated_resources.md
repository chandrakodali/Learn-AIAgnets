# 🎬 AI Agents Development — Video-First Learning Path

> **How to use this guide:**
> - 🟢 **MUST** = Watch this one. It's the best resource for the topic.
> - 🟡 **DEEP DIVE** = Only if 🟢 wasn't enough or you want more depth.
> - 🔵 **OPTIONAL** = Alternative angle or quick recap. Skip unless curious.
> - 📖 **Reference** = Docs to check only if a video didn't click.

---

## Module 1: Development Environment (Week 1–2)

### 🎬 Videos

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 1.1 | 🟢 MUST | [ArjanCodes — Modern Python Patterns](https://www.youtube.com/@ArjanCodes) | Clean Python, type hints, async/await, design patterns |
| 1.2 | 🟡 DEEP DIVE | Search: [`"Python 3.10 new features" tutorial`](https://www.youtube.com/results?search_query=python+3.10+new+features+structural+pattern+matching+tutorial) | `match/case`, `ParamSpec`, type aliases |
| 1.3 | 🟢 MUST | Search: [`uv python package manager tutorial 2025`](https://www.youtube.com/results?search_query=uv+python+package+manager+tutorial+getting+started+2025) | uv install, `uv init`, `uv add`, `uv run` |

### 📖 Reference If Stuck

| Resource | Link |
|----------|------|
| Python 3.10 What's New | [docs.python.org/3/whatsnew/3.10.html](https://docs.python.org/3/whatsnew/3.10.html) |
| RealPython — Type Checking | [realpython.com/python-type-checking/](https://realpython.com/python-type-checking/) |
| uv Official Docs | [docs.astral.sh/uv/](https://docs.astral.sh/uv/) |
| RealPython — Projects with uv | [realpython.com/python-uv/](https://realpython.com/python-uv/) |
| DataCamp — uv Guide | [datacamp.com/tutorial/python-uv](https://www.datacamp.com/tutorial/python-uv) |
| python-dotenv | [pypi.org/project/python-dotenv/](https://pypi.org/project/python-dotenv/) |
| Python asyncio | [docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html) |

---

## Module 2: The "Brain" — Pydantic + PydanticAI + OpenAI Agents SDK (Week 3–4)

### 🎬 Videos — Pydantic

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 2.1 | 🟢 MUST | [ArjanCodes — "Why Python Needs Pydantic"](https://www.youtube.com/results?search_query=arjancodes+why+python+needs+pydantic) | BaseModel, validation, serialization fundamentals |
| 2.2 | 🟡 DEEP DIVE | Search: [`Pydantic V2 Full Course`](https://www.youtube.com/results?search_query=pydantic+v2+full+course) | Validators, nested models, settings, advanced patterns |

### 🎬 Videos — PydanticAI

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 2.3 | 🟢 MUST | Search: [`"Master Pydantic AI in Under 1 Hour" 2025`](https://www.youtube.com/results?search_query=master+pydantic+ai+under+1+hour+2025) | Type-safe agents, practical coding, structured outputs |
| 2.4 | 🟡 DEEP DIVE | Search: [`"Pydantic AI Crash Course" FastAPI NextJS agentic`](https://www.youtube.com/results?search_query=pydantic+ai+crash+course+fastapi+nextjs+agentic) | Full-stack agent app: FastAPI + PydanticAI + Gemini |
| 2.5 | 🔵 OPTIONAL | Search: [`"Building a production AI agent system with Pydantic"`](https://www.youtube.com/results?search_query=building+production+ai+agent+system+pydantic+2025) | Production patterns, API failures, multi-agent coord |
| 2.6 | 🔵 OPTIONAL | Search: [`"How to Build AI Agents with PydanticAI" beginner`](https://www.youtube.com/results?search_query=how+to+build+ai+agents+pydanticai+beginner+tutorial) | Alternative beginner walkthrough |

### 🎬 Videos — OpenAI Agents SDK

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 2.7 | 🟢 MUST | Search: [`Aurelio Labs "Agents SDK from OpenAI" Full Tutorial`](https://www.youtube.com/results?search_query=aurelio+labs+agents+sdk+openai+full+tutorial) | Agent Loop, Guardrails, Function Tools, Tracing |
| 2.8 | 🟡 DEEP DIVE | Search: [`Aurelio Labs "Multi-Agent Systems in OpenAI's Agents SDK"`](https://www.youtube.com/results?search_query=aurelio+labs+multi+agent+systems+openai+agents+sdk) | Handoffs, orchestrator-subagent pattern |
| 2.9 | 🔵 OPTIONAL | Search: [`DataCamp OpenAI Agents SDK tutorial`](https://www.youtube.com/results?search_query=datacamp+openai+agents+sdk+tutorial) | Alternative walkthrough, structured outputs |

### 📖 Reference If Stuck

| Resource | Link |
|----------|------|
| Pydantic V2 Docs | [docs.pydantic.dev/latest/](https://docs.pydantic.dev/latest/) |
| PydanticAI Docs + Examples | [ai.pydantic.dev/](https://ai.pydantic.dev/) |
| PydanticAI Examples | [ai.pydantic.dev/examples/](https://ai.pydantic.dev/examples/) |
| OpenAI Agents SDK Docs | [openai.github.io/openai-agents-python/](https://openai.github.io/openai-agents-python/) |
| SDK — Guardrails | [openai.github.io/openai-agents-python/guardrails/](https://openai.github.io/openai-agents-python/guardrails/) |
| SDK — Handoffs | [openai.github.io/openai-agents-python/handoffs/](https://openai.github.io/openai-agents-python/handoffs/) |
| r/PydanticAI | [reddit.com/r/PydanticAI](https://www.reddit.com/r/PydanticAI/) |

---

## Module 3: The "Spine" — LangGraph + CrewAI + A2A Protocol (Week 5–6)

### 🎬 Videos — LangGraph

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 3.1 | 🟢 MUST | [DeepLearning.AI — "AI Agents in LangGraph" (FREE, 1.5hr)](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | Build agents from scratch, then rebuild with LangGraph |
| 3.2 | 🟡 DEEP DIVE | [LangChain Academy — Intro to LangGraph (FREE)](https://academy.langchain.com/courses/intro-to-langgraph) | Full curriculum: state, memory, streaming, sub-graphs |
| 3.3 | 🔵 OPTIONAL | Search: [`"LangGraph Crash Course" beginners 2025 8 hour`](https://www.youtube.com/results?search_query=langgraph+crash+course+beginners+2025+full+course) | Same as 3.1+3.2 but in one giant video |
| 3.4 | 🔵 OPTIONAL | Search: [`"1 Hour LangGraph Crash Course" zero to hero`](https://www.youtube.com/results?search_query=1+hour+langgraph+crash+course+zero+to+hero+react+agent) | Quick recap after you've already built something |

### 🎬 Videos — CrewAI

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 3.5 | 🟢 MUST | [DeepLearning.AI — "Multi AI Agent Systems with crewAI" (FREE, 2hr)](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) | Role-playing, memory, tools, multi-agent collaboration |
| 3.6 | 🟡 DEEP DIVE | Search: [`"CrewAI 2025 Build Your First AI Agent"`](https://www.youtube.com/results?search_query=crewai+2025+build+first+ai+agent) | Hands-on project setup with various providers |
| 3.7 | 🔵 OPTIONAL | Search: [`"Build AI Agents with CrewAI MCP and Gemini"`](https://www.youtube.com/results?search_query=build+ai+agents+crewai+mcp+gemini+tutorial) | CrewAI + MCP integration (the 2026 pattern) |

### 🎬 Videos — A2A Protocol

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 3.8 | 🟢 MUST | Search: [`"A2A protocol" Google agent-to-agent explained`](https://www.youtube.com/results?search_query=a2a+protocol+google+agent+to+agent+explained+2025) | What A2A is, how it complements MCP |

### 📖 Reference If Stuck

| Resource | Link |
|----------|------|
| LangGraph Docs | [langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/) |
| CrewAI Docs | [docs.crewai.com/](https://docs.crewai.com/) |
| A2A GitHub Repo | [github.com/google/A2A](https://github.com/google/A2A) |
| freeCodeCamp — LangGraph Guide | [freecodecamp.org](https://www.freecodecamp.org/news/how-to-use-langchain-and-langgraph/) |
| r/LangChain | [reddit.com/r/LangChain](https://www.reddit.com/r/LangChain/) |

---

## Module 4: The "Senses & Memory" — MCP + Vector DBs + GraphRAG (Week 7–8)

### 🎬 Videos — MCP & FastMCP

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 4.1 | 🟢 MUST | Search: [`"Model Context Protocol MCP" explained tutorial 2025`](https://www.youtube.com/results?search_query=model+context+protocol+mcp+explained+tutorial+2025) | What MCP is, architecture, tools/resources/prompts |
| 4.2 | 🟡 DEEP DIVE | Search: [`"Build MCP Server" Python tutorial`](https://www.youtube.com/results?search_query=build+mcp+server+python+tutorial+tools+resources+2025) | Build your own MCP server with Python/FastMCP |
| 4.3 | 🔵 OPTIONAL | Search: [`FastMCP tutorial Python build server`](https://www.youtube.com/results?search_query=fastmcp+tutorial+python+build+mcp+server) | FastMCP 2.0 advanced features |

### 🎬 Videos — pgvector + Qdrant

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 4.4 | 🟢 MUST | Search: [`"Build RAGs with PostgreSQL" beginner pgvector`](https://www.youtube.com/results?search_query=build+rags+postgresql+beginner+pgvector+tutorial+2025) | Setup pgvector, store embeddings, build RAG |
| 4.5 | 🔵 OPTIONAL | Search: [`"AI on Postgres" pgvector RAG vectors`](https://www.youtube.com/results?search_query=ai+on+postgres+pgvector+rag+vectors+embeddings) | pgvector as AI backend, indexing deep dive |
| 4.6 | 🟢 MUST | Search: [`Qdrant vector database tutorial Python 2025`](https://www.youtube.com/results?search_query=qdrant+vector+database+tutorial+getting+started+python+2025) | Qdrant setup, ANN search, payload filtering |

### 🎬 Videos — Neo4j + GraphRAG

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 4.7 | 🟢 MUST | Search: [`Neo4j GraphRAG tutorial knowledge graph 2025`](https://www.youtube.com/results?search_query=neo4j+graphrag+tutorial+knowledge+graph+2025) | Knowledge graphs, relationships, multi-hop reasoning |
| 4.8 | 🔵 OPTIONAL | Search: [`Microsoft GraphRAG tutorial Python`](https://www.youtube.com/results?search_query=microsoft+graphrag+tutorial+python+implementation+2025) | Microsoft's GraphRAG implementation |

### 📖 Reference If Stuck

| Resource | Link |
|----------|------|
| MCP Official Docs | [modelcontextprotocol.io/](https://modelcontextprotocol.io/) |
| FastMCP Docs | [gofastmcp.com/](https://gofastmcp.com/) |
| pgvector GitHub | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) |
| Qdrant Quickstart | [qdrant.tech/documentation/quickstart/](https://qdrant.tech/documentation/quickstart/) |
| Neo4j GraphRAG Tutorial | [neo4j.com — GraphRAG](https://neo4j.com/developer-blog/graphrag-tutorial/) |
| Microsoft GraphRAG GitHub | [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag) |
| r/modelcontextprotocol | [reddit.com/r/modelcontextprotocol](https://www.reddit.com/r/modelcontextprotocol/) |
| r/vectordatabase | [reddit.com/r/vectordatabase](https://www.reddit.com/r/vectordatabase/) |

---

## Module 5: The "Nervous System" — LangSmith + Evals (Week 9–10)

### 🎬 Videos

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 5.1 | 🟢 MUST | [DeepLearning.AI — "Agentic AI" (FREE)](https://www.deeplearning.ai/short-courses/agentic-ai/) | Task decomposition, **evaluation**, design patterns |
| 5.2 | 🟡 DEEP DIVE | [DeepLearning.AI — "Agent Memory" (FREE)](https://www.deeplearning.ai/short-courses/llms-as-operating-systems-agent-memory/) | Agentic memory from scratch, tool-calling |
| 5.3 | 🟢 MUST | Search: [`LangSmith tutorial tracing debugging agents 2025`](https://www.youtube.com/results?search_query=langsmith+tutorial+tracing+debugging+agents+2025) | Set up LangSmith, trace calls, debug workflows |
| 5.4 | 🟡 DEEP DIVE | Search: [`"LLM as judge" evaluation AI agents tutorial`](https://www.youtube.com/results?search_query=llm+as+judge+evaluation+ai+agents+tutorial+2025) | Eval suites, golden datasets, scoring rubrics |

### 📖 Reference If Stuck

| Resource | Link |
|----------|------|
| LangSmith Docs | [docs.smith.langchain.com/](https://docs.smith.langchain.com/) |
| LangSmith Quickstart | [docs.smith.langchain.com/observability/quickstart](https://docs.smith.langchain.com/observability/quickstart) |
| LangSmith Evaluation Guide | [docs.smith.langchain.com/evaluation](https://docs.smith.langchain.com/evaluation) |
| OpenAI Evals Cookbook | [cookbook.openai.com — Evals](https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals) |

---

## Module 6: The "Home" — FastAPI + Docker + Redis + Deploy (Week 11–12)

### 🎬 Videos — FastAPI

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 6.1 | 🟢 MUST | Search: [`"FastAPI Full Course for Beginners" 2025 Python`](https://www.youtube.com/results?search_query=fastapi+full+course+beginners+2025+python+tutorial) | Routes, Pydantic models, async, Swagger docs |
| 6.2 | 🔵 OPTIONAL | Search: [`"FastAPI Crash Course" Python 2025`](https://www.youtube.com/results?search_query=fastapi+crash+course+python+fastest+web+framework+2025) | Quick recap: HTTP methods, query params |

### 🎬 Videos — Docker

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 6.3 | 🟢 MUST | Search: [`"Docker Full Course for Beginners" 2025`](https://www.youtube.com/results?search_query=docker+full+course+beginners+2025) | Images, containers, Dockerfile, volumes, Compose |
| 6.4 | 🟡 DEEP DIVE | Search: [`"Docker Essentials for Python Developers" 2025`](https://www.youtube.com/results?search_query=docker+essentials+python+developers+tutorial+2025) | Python-specific containerization, multi-stage builds |
| 6.5 | 🔵 OPTIONAL | Search: [`"Python Web App in Docker Container" guide`](https://www.youtube.com/results?search_query=python+web+app+docker+container+guide+tutorial+2025) | Focused walkthrough: Python + Docker |

### 🎬 Videos — Redis

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 6.6 | 🟢 MUST | Search: [`"Master Redis with Python" crash course beginners`](https://www.youtube.com/results?search_query=master+redis+python+crash+course+beginners) | Redis basics: data types, caching, Python ops |
| 6.7 | 🟡 DEEP DIVE | Search: [`"AI Agent That Never Forgets Redis LangGraph"`](https://www.youtube.com/results?search_query=ai+agent+never+forgets+redis+langgraph+memory+tutorial) | Redis as AI agent memory + LangGraph |
| 6.8 | 🔵 OPTIONAL | Search: [`"Semantic Caching for AI Agents" Redis`](https://www.youtube.com/results?search_query=semantic+caching+ai+agents+redis+course) | Semantic cache for faster, cheaper agents |

### 🎬 Videos — Cloud Deployment

| # | Priority | Video | What You'll Learn |
|---|----------|-------|-------------------|
| 6.9 | 🟢 MUST | Search: [`Modal tutorial deploy python serverless GPU`](https://www.youtube.com/results?search_query=modal+deploy+python+serverless+gpu+tutorial+2025) | Modal: deploy AI workloads, pay-per-use |
| 6.10 | 🔵 OPTIONAL | Search: [`Google Cloud Run deploy FastAPI Python`](https://www.youtube.com/results?search_query=google+cloud+run+deploy+fastapi+python+tutorial+2025) | Alternative: deploy to Cloud Run |

### 📖 Reference If Stuck

| Resource | Link |
|----------|------|
| FastAPI Tutorial | [fastapi.tiangolo.com/tutorial/](https://fastapi.tiangolo.com/tutorial/) |
| Docker Getting Started | [docs.docker.com/get-started/](https://docs.docker.com/get-started/) |
| Docker Compose | [docs.docker.com/compose/](https://docs.docker.com/compose/) |
| Redis Docs | [redis.io/docs/](https://redis.io/docs/) |
| Redis for AI | [redis.io/solutions/ai/](https://redis.io/solutions/ai/) |
| Modal Docs | [modal.com/docs/guide](https://modal.com/docs/guide) |
| Cloud Run Docs | [cloud.google.com/run/docs](https://cloud.google.com/run/docs) |

---

## 🎓 Bonus: Free Courses (Pair with Modules)

| When | Course | Platform | Duration | Link |
|------|--------|----------|----------|------|
| Module 3 | AI Agents in LangGraph | DeepLearning.AI | 1h 32m | [Link](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) |
| Module 3 | Multi AI Agent Systems with crewAI | DeepLearning.AI | 2h 14m | [Link](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) |
| Module 3 | Intro to LangGraph | LangChain Academy | Self-paced | [Link](https://academy.langchain.com/courses/intro-to-langgraph) |
| Module 5 | Agentic AI Best Practices | DeepLearning.AI | ~2h | [Link](https://www.deeplearning.ai/short-courses/agentic-ai/) |
| Module 5 | Agent Memory | DeepLearning.AI | ~2h | [Link](https://www.deeplearning.ai/short-courses/llms-as-operating-systems-agent-memory/) |
| Any time | AI Agents in Python | Coursera | 2h | [Link](https://www.coursera.org/learn/ai-agents-and-agentic-ai-in-python) |

---

## ⚡ TL;DR — The "Must Watch Only" Fast Track

> If you're short on time, watch **only the 🟢 MUST** videos. That's **~15 videos** total across 12 weeks.

| Module | 🟢 Must Watch Videos |
|--------|---------------------|
| 1 | 1.1 (ArjanCodes Python) + 1.3 (uv) |
| 2 | 2.1 (Pydantic) + 2.3 (PydanticAI 1hr) + 2.7 (OpenAI SDK) |
| 3 | 3.1 (LangGraph DeepLearning.AI) + 3.5 (CrewAI DeepLearning.AI) + 3.8 (A2A) |
| 4 | 4.1 (MCP) + 4.4 (pgvector) + 4.6 (Qdrant) + 4.7 (Neo4j GraphRAG) |
| 5 | 5.1 (Agentic AI eval) + 5.3 (LangSmith) |
| 6 | 6.1 (FastAPI) + 6.3 (Docker) + 6.6 (Redis) + 6.9 (Modal deploy) |
