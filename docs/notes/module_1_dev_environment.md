# Module 1: Development Environment

> **Goal**: Set up a modern Python environment that every AI agent project will use.
> **Time**: Week 1–2 | **Watch alongside**: Videos 1.1, 1.2, 1.3 from curated resources

---

## 🎯 What & Why

You're building AI agents. Before writing any AI code, you need a **rock-solid Python setup** — the same way a chef prepares their kitchen before cooking.

This module covers 3 things:
1. **Modern Python features** (3.10+) that agent frameworks depend on
2. **uv** — the new package manager that replaces pip, poetry, pyenv, and virtualenv
3. **.env** — how to safely manage API keys (you'll have many: OpenAI, Tavily, LangSmith, etc.)

> **Why this matters**: If you skip this, you'll fight your tools instead of learning agents. Every tutorial from Module 2 onward assumes you know this.

---

## 🧠 Section 1: Type Hints — Why AI Agents Need Them

### What are type hints?

Type hints tell Python (and your IDE) **what kind of data** a variable holds or a function accepts/returns.

```python
# ❌ Without type hints — what does this return? A string? A dict? Who knows?
def process(data):
    return data["name"]

# ✅ With type hints — crystal clear
def process(data: dict[str, str]) -> str:
    return data["name"]
```

### Why do AI agents care?

PydanticAI and the OpenAI Agents SDK use type hints to **automatically validate data**. If your agent returns `{"name": 123}` but you said it should return `str`, Pydantic catches the error before it causes problems.

```mermaid
flowchart LR
    A["LLM Output\n(messy JSON)"] --> B["Pydantic Model\n(type hints)"]
    B -->|Valid| C["✅ Clean data"]
    B -->|Invalid| D["❌ ValidationError\n(caught early!)"]
```

### The key type hints you'll use

```python
from typing import Optional

# Basic types
name: str = "Koushik"
age: int = 25
score: float = 98.5
is_active: bool = True

# Collections
tags: list[str] = ["python", "ai"]
config: dict[str, int] = {"retries": 3, "timeout": 30}

# Optional — the value might be None
nickname: Optional[str] = None  # Could be a string OR None

# Python 3.10+ shorthand for Optional (much cleaner!)
nickname: str | None = None     # Same thing, but modern syntax
```

### Union types (Python 3.10+)

Before Python 3.10, combining types was ugly:
```python
# ❌ Old way (Python 3.9 and below)
from typing import Union
def get_id() -> Union[int, str]:
    ...

# ✅ New way (Python 3.10+) — use the pipe operator
def get_id() -> int | str:
    ...
```

> **Bottom line**: Always use `X | Y` instead of `Union[X, Y]`. Every agent framework tutorial uses the modern syntax.

---

## 🧠 Section 2: Structural Pattern Matching (`match/case`)

### What is it?

Python 3.10 added `match/case` — a powerful way to check data shape and route logic. Think of it as a smarter `if/elif/else`.

### Why agents need it

Agents receive different types of responses: tool calls, text, errors, handoffs. Pattern matching lets you handle each cleanly:

```python
# 🤖 Handling different agent response types
def handle_response(response: dict) -> str:
    match response:
        case {"type": "text", "content": content}:
            return f"Agent said: {content}"

        case {"type": "tool_call", "tool": tool_name, "args": args}:
            return f"Agent wants to call: {tool_name}({args})"

        case {"type": "error", "message": msg}:
            return f"Agent error: {msg}"

        case {"type": "handoff", "target": agent_name}:
            return f"Handing off to: {agent_name}"

        case _:  # The underscore matches ANYTHING (default case)
            return "Unknown response type"
```

### Before vs After

```python
# ❌ Old way — messy nested if/elif
def handle_response(response):
    if response.get("type") == "text":
        content = response.get("content", "")
        return f"Agent said: {content}"
    elif response.get("type") == "tool_call":
        tool_name = response.get("tool", "")
        args = response.get("args", {})
        return f"Agent wants to call: {tool_name}({args})"
    elif response.get("type") == "error":
        # ... you get the idea. This gets ugly fast.

# ✅ New way — match/case is cleaner AND extracts variables for you
# (see example above)
```

### Three patterns you'll use most

```python
# 1. Match by value
match status_code:
    case 200:
        print("OK")
    case 404:
        print("Not found")
    case 500:
        print("Server error")

# 2. Match by structure (extracting values)
match user:
    case {"name": name, "role": "admin"}:
        print(f"Admin: {name}")
    case {"name": name, "role": "user"}:
        print(f"User: {name}")

# 3. Match with guard conditions
match age:
    case n if n < 0:
        print("Invalid")
    case n if n < 18:
        print("Minor")
    case _:
        print("Adult")
```

---

## 🧠 Section 3: `async/await` — Why Agents MUST Be Async

### The restaurant analogy (from Reddit)

Imagine a restaurant with **one chef** (your Python program has one thread):

```mermaid
flowchart TB
    subgraph "❌ Synchronous Chef (Blocking)"
        S1["Start pasta water"] --> S2["⏳ Wait 10 min\n(doing NOTHING)"]
        S2 --> S3["Cook pasta"]
        S3 --> S4["Start grilling chicken"]
        S4 --> S5["⏳ Wait 15 min\n(doing NOTHING)"]
        S5 --> S6["Serve both dishes"]
    end
```

**Total time**: 25+ minutes. The chef wastes time just staring at pots.

```mermaid
flowchart TB
    subgraph "✅ Async Chef (Non-Blocking)"
        A1["Start pasta water\n(await boil)"] --> A2["While waiting...\nstart grilling chicken\n(await grill)"]
        A2 --> A3["Water boils!\nCook pasta"]
        A3 --> A4["Chicken done!\nServe both dishes"]
    end
```

**Total time**: ~15 minutes. Same chef, same kitchen, much faster.

### Why agents need this

AI agents make many **slow external calls**: LLM APIs (1-5 seconds each), web searches, database queries. Without async, your agent waits idle after each call. With async, it can fire multiple calls and process results as they arrive.

### Basic async code

```python
import asyncio

# 🔑 Step 1: Define async functions with "async def"
async def call_llm(prompt: str) -> str:
    """Simulate an LLM API call that takes 2 seconds"""
    print(f"📤 Sending: {prompt}")
    await asyncio.sleep(2)  # Simulates network wait (not real sleep!)
    print(f"📥 Got response for: {prompt}")
    return f"Response to: {prompt}"

# 🔑 Step 2: Run multiple calls concurrently with asyncio.gather()
async def main():
    # ❌ Sequential — takes 6 seconds (2+2+2)
    # result1 = await call_llm("What is Python?")
    # result2 = await call_llm("What is async?")
    # result3 = await call_llm("What is an agent?")

    # ✅ Concurrent — takes ~2 seconds (all 3 run at the same time!)
    results = await asyncio.gather(
        call_llm("What is Python?"),
        call_llm("What is async?"),
        call_llm("What is an agent?"),
    )
    print(f"Got {len(results)} results")

# 🔑 Step 3: Run the async entry point
asyncio.run(main())
```

**Output (concurrent version):**
```
📤 Sending: What is Python?
📤 Sending: What is async?        ← All 3 start immediately!
📤 Sending: What is an agent?
📥 Got response for: What is Python?
📥 Got response for: What is async?      ← All 3 finish ~2 sec later
📥 Got response for: What is an agent?
Got 3 results
```

### The rules

| Rule | Details |
|------|---------|
| Use `async def` | For any function that calls APIs, databases, or other async functions |
| Use `await` | Before any async operation — it means "start this, but I can do other stuff while waiting" |
| Use `asyncio.gather()` | To run multiple async tasks concurrently |
| Use `asyncio.run()` | To start the async event loop from regular (sync) code |
| **Never** use `time.sleep()` | In async code, use `await asyncio.sleep()` instead |

---

## 🧠 Section 4: uv — The Modern Package Manager

### What problem does uv solve?

Before uv, Python developers needed **4–5 separate tools**:

```mermaid
flowchart LR
    subgraph "❌ Old Way (4+ tools)"
        A["pyenv\n(Python versions)"]
        B["virtualenv\n(environments)"]
        C["pip\n(install packages)"]
        D["pip-tools\n(lock dependencies)"]
        E["poetry\n(project management)"]
    end

    subgraph "✅ New Way (1 tool)"
        F["uv\n(does ALL of it)\n10-100x faster"]
    end
```

### Getting started with uv

```bash
# 📦 Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# ✅ Verify it worked
uv --version
```

### Create your first project

```bash
# 🏗️ Create a new project
uv init my-agent-project
cd my-agent-project

# 📂 What uv created for you:
# my-agent-project/
# ├── pyproject.toml      ← Project config (replaces setup.py + requirements.txt)
# ├── .python-version     ← Pins Python version
# ├── hello.py            ← Starter script
# └── README.md
```

### Daily workflow commands

```bash
# ➕ Add a dependency (like pip install, but faster)
uv add pydantic-ai
uv add openai
uv add python-dotenv

# ➕ Add a development dependency (not shipped to production)
uv add --dev pytest
uv add --dev ruff
uv add --dev mypy

# 🔄 Sync your environment (install everything from lockfile)
uv sync

# ▶️ Run your script (auto-creates venv if needed!)
uv run main.py

# 🔧 Run a dev tool without installing it permanently
uvx ruff check .      # Lint your code
uvx mypy --strict .   # Type check

# 🌳 See your dependency tree
uv tree

# ❌ Remove a package
uv remove openai
```

### pip → uv Translation Table

| What You Want | pip (old) | uv (new) |
|---------------|-----------|----------|
| Install a package | `pip install requests` | `uv add requests` |
| Install dev dependency | `pip install pytest` | `uv add --dev pytest` |
| Install from requirements.txt | `pip install -r requirements.txt` | `uv add -r requirements.txt` |
| Create virtual environment | `python -m venv .venv` | `uv venv` (or auto-created) |
| Run a script | `python main.py` | `uv run main.py` |
| Freeze dependencies | `pip freeze > requirements.txt` | `uv lock` (creates uv.lock) |
| Install specific Python | `pyenv install 3.12` | `uv python install 3.12` |
| Pin Python version | `pyenv local 3.12` | `uv python pin 3.12` |

---

## 🧠 Section 5: .env & Secrets Management

### The problem

AI agent projects need **many API keys**:
- OpenAI / Gemini / Groq (LLM providers)
- Tavily (web search)
- LangSmith (observability)
- Database credentials

**NEVER** hardcode these in your code. One accidental `git push` and your keys are public.

### The solution: `.env` files

```mermaid
flowchart LR
    A[".env file\n(SECRET - never commit)"] -->|python-dotenv loads| B["os.environ\n(available in code)"]
    C[".env.example\n(TEMPLATE - commit this)"] -->|Tells teammates| D["What keys are needed"]
    E[".gitignore\n(blocks .env)"] -->|Prevents| F["Accidental exposure"]
```

### Step-by-step setup

**1. Create `.env` (your actual secrets):**
```bash
# .env — ⚠️ NEVER commit this file
OPENAI_API_KEY=sk-abc123...
TAVILY_API_KEY=tvly-xyz789...
LANGSMITH_API_KEY=lsv2-...
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
```

**2. Create `.env.example` (template for others):**
```bash
# .env.example — ✅ Commit this file (no real values!)
OPENAI_API_KEY=your-openai-key-here
TAVILY_API_KEY=your-tavily-key-here
LANGSMITH_API_KEY=your-langsmith-key-here
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
```

**3. Add to `.gitignore`:**
```
# .gitignore
.env
.venv/
__pycache__/
```

**4. Load in your Python code:**
```python
# config.py
import os
from dotenv import load_dotenv

# Load .env file into environment variables
load_dotenv()

# Access with type safety
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]     # Crashes if missing (good!)
TAVILY_API_KEY: str = os.environ["TAVILY_API_KEY"]
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///local.db")  # Fallback value
```

**5. Use in your agent code:**
```python
# main.py
from config import OPENAI_API_KEY

# Now use it — the key is loaded from .env, never hardcoded
agent = Agent(model="openai:gpt-4o", api_key=OPENAI_API_KEY)
```

---

## 🔗 How Module 1 Connects to Everything Else

```mermaid
flowchart TB
    M1["Module 1\nDev Environment"]

    M1 -->|"Type hints power"| M2["Module 2\nPydantic + PydanticAI"]
    M1 -->|"async/await powers"| M3["Module 3\nLangGraph agents"]
    M1 -->|"uv manages"| M4["Module 4\nMCP + vector DB deps"]
    M1 -->|".env secures"| M5["Module 5\nLangSmith API keys"]
    M1 -->|"Project structure feeds"| M6["Module 6\nDocker + deployment"]

    style M1 fill:#22c55e,color:#fff
```

Everything you set up here is used in **every** future module. This is the foundation.

---

## ⚡ Cheat Sheet

### Python 3.10+ Quick Reference

| Feature | Syntax | Example |
|---------|--------|---------|
| Union types | `X \| Y` | `def foo() -> int \| str:` |
| Optional | `X \| None` | `name: str \| None = None` |
| Pattern matching | `match/case` | `match response: case {...}:` |
| Type alias | `type Name = ...` | `type UserID = int \| str` |

### uv Quick Reference

| Action | Command |
|--------|---------|
| New project | `uv init my-project` |
| Add package | `uv add pydantic-ai` |
| Add dev package | `uv add --dev pytest` |
| Run script | `uv run main.py` |
| Run tool | `uvx ruff check .` |
| See deps | `uv tree` |
| Install Python | `uv python install 3.12` |
| Pin Python | `uv python pin 3.12` |

### .env Quick Reference

| File | Purpose | Git? |
|------|---------|------|
| `.env` | Real API keys | ❌ NEVER commit |
| `.env.example` | Template (no secrets) | ✅ Commit |
| `.gitignore` | Blocks `.env` from git | ✅ Commit |
| `config.py` | Loads `.env` into code | ✅ Commit |

---

## ✅ Checkpoint Quiz

Test yourself before moving to Module 2:

**1.** What's the modern way to write `Union[int, str]` in Python 3.10+?

<details><summary>Answer</summary>

`int | str` — use the pipe operator.
</details>

**2.** In async Python, what happens when the code hits `await`?

<details><summary>Answer</summary>

The current function **pauses** and gives control back to the event loop, which can run other tasks. When the awaited operation completes, the function resumes from where it paused. (The chef analogy: "I started boiling water, let me chop veggies while I wait.")
</details>

**3.** What's the difference between `uv add requests` and `uv add --dev pytest`?

<details><summary>Answer</summary>

`uv add requests` adds it as a **production dependency** (shipped with your app). `uv add --dev pytest` adds it as a **development dependency** (only used during development/testing, not shipped).
</details>

**4.** Why should you use `os.environ["KEY"]` instead of `os.getenv("KEY")`?

<details><summary>Answer</summary>

`os.environ["KEY"]` **raises a KeyError** if the key is missing — this is good because it catches misconfiguration immediately. `os.getenv("KEY")` silently returns `None`, which can cause hard-to-debug errors later.
</details>

**5.** You accidentally committed your `.env` file with API keys. What should you do?

<details><summary>Answer</summary>

1. **Immediately rotate/regenerate ALL API keys** in the `.env` file (the old ones are compromised)
2. Add `.env` to `.gitignore`
3. Remove from git history: `git rm --cached .env`
4. Force push the fix
5. Consider using tools like `git-secrets` to prevent this in the future
</details>

---

> **Next up**: [Module 2 — The "Brain": Pydantic + PydanticAI + OpenAI Agents SDK](module_2_brain.md)
