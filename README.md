# LLM Chatbot with Key Failover and Response Caching

Full-stack resilient LLM chat gateway built with **React**, **FastAPI**, **LiteLLM**, and **SQLite**.

## 1. Live URLs

- **Frontend URL**: _Add your deployed Vercel / Netlify / Cloudflare Pages URL here_
- **Backend URL**: _Add your deployed Render / Railway / Fly.io URL here_
- **GitHub Repository**: https://github.com/Krishna-0028/lyftr-llm-gateway

## 2. Run locally: with Docker and without

### With Docker (`make up`)

1. Copy `.env.example` to `.env` and fill in at least two API keys in `LLM_API_KEYS`:

```bash
cp .env.example .env
```

2. Start both services (backend on `http://localhost:8000`, frontend on `http://localhost:3000`):

```bash
make up
```

3. Stop the stack:

```bash
make down
```

SQLite data is persisted at `/data/app.db` inside the `sqlite_data` Docker volume.

### Without Docker

#### Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export LLM_API_KEYS="your-key-1,your-key-2"
export LLM_MODEL="gemini/gemini-2.0-flash"
export DATABASE_URL="sqlite:///./data/app.db"
export ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
export ENABLE_FAILURE_SIMULATION="true"

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Run backend tests:

```bash
cd backend
.venv/bin/pytest -v
```

#### Frontend

```bash
cd frontend
npm install
VITE_API_BASE_URL="http://localhost:8000" npm run dev
```

Open `http://localhost:3000`.

## 3. Design decisions: key fallback, caching, SQLite session storage

- **Key fallback (`backend/app/llm.py`)**: API keys are loaded in order from `LLM_API_KEYS` and mapped deterministically to `key_1`, `key_2`, etc. Each live attempt calls `litellm.completion(..., num_retries=0)` so rate-limit (`429`), quota, `5xx`, or connection errors fail over immediately to the next key rather than stalling on per-key retries. `fallback_attempts` tracks failed keys before a successful reply; if every key fails, FastAPI returns `503 {"detail": "all providers unavailable"}` and no messages are written to SQLite.
- **Response caching (`backend/app/llm.py`)**: Uses LiteLLM's built-in `Cache` (`local` in-memory by default, or `disk` via `CACHE_TYPE=disk`). The cache key is derived from the model and full message list (`system prompt + history + new message`), so identical first-turn prompts across different sessions hit `LLM_CACHE` with `key_id: null` and `fallback_attempts: 0`. When `X-Simulate-Failure` is active (and `ENABLE_FAILURE_SIMULATION=true`), cache reads are bypassed (`cache={"no-cache": True, "no-store": True}`) so failover behavior can always be demonstrated deterministically even on repeated prompts.
- **SQLite session storage (`backend/app/db.py`)**: Uses Python's standard `sqlite3` with WAL mode (`PRAGMA journal_mode=WAL`) and two tables (`sessions` and `messages`) created automatically on startup. History is queried oldest-first per `session_id`, and `save_turn` runs in a single transaction only after a successful LLM call so failed requests (`422` or `503`) never pollute conversation history.

## 4. Verification: how to see failover, cache hits and per-message token usage

### Health check

```bash
curl -s http://localhost:8000/health
```

Expected:

```json
{"status":"ok","keys_configured":2,"cache_enabled":true}
```

### Live inference vs. cache hit + per-request token usage

```bash
# Turn 1 (session s-demo-1) -> LIVE_INFERENCE
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-1","message":"What is the capital of France?"}'

# Identical first message in a new session (s-demo-2) -> LLM_CACHE, key_id: null
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s-demo-2","message":"What is the capital of France?"}'
```

### Simulated key failover (`X-Simulate-Failure`)

```bash
# Simulate key_1 returning 429 -> fails over to key_2
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id":"s-demo-3","message":"Explain caching in one sentence."}'

# Simulate all keys failing -> 503 {"detail": "all providers unavailable"}
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id":"s-demo-4","message":"Should not persist."}'

# Confirm failed request was not stored
curl -i http://localhost:8000/sessions/s-demo-4/messages
```

## 5. Setup Used: your editor and AI tools

- **Editor / IDE**: Antigravity IDE
- **AI Tools**: Antigravity coding assistant for architecture scaffolding, test generation, and contract verification

## 6. Bonus features (if any): what you added beyond the requirements and why

- **Optional System Prompt input (`[data-testid="system-prompt-input"]`)**: Persists per session in SQLite (`sessions` table) while remaining excluded from `GET /sessions/{session_id}/messages`, allowing live experimentation with persona/instructions without altering the API contract.
- **In-UI Failover Simulation Selector**: Lets reviewers and demos trigger `X-Simulate-Failure` (`key_1=429`, `key_1=500`, `*=429`) directly from the web UI alongside standard `curl` verification.
- **Adaptive Light/Dark UI & IME-safe keyboard submission**: Respects system `color-scheme` (`light-dark()` with `prefers-color-scheme` fallback) and guards Enter-to-submit against IME composition (`isComposing` / `keyCode === 229`).
