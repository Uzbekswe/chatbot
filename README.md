# Terminal Chatbot

A fully-featured multi-turn terminal chatbot powered by a local LLM via [Ollama](https://ollama.com).  
Built as a hands-on project to understand how conversation state, token counting, cost estimation, and resilient API design work in practice.

---

## Features

### Streaming Responses
Replies are streamed token-by-token directly to the terminal as they are generated — no waiting for the full response before seeing output.

Ollama sends a final chunk at the end of every stream that contains usage data but has an **empty `choices` list**. The code guards against this explicitly:
- `stream_options={"include_usage": True}` is passed to opt into usage data on the stream
- Each chunk is checked with `if chunk.choices:` before accessing content — preventing an `IndexError` crash on the final usage-only chunk
- `chunk.usage` is captured **inside** the streaming loop (not from `stream.usage` after it, which is always `None`)

### Real-time Token Counting
After every reply, the exact token breakdown is displayed:

```
tokens → total: 1,243 | in: 1,102 | out: 141
```

A colour-coded progress bar shows how much of the context window has been consumed:

```
context window ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  1% (1,243 / 128,000 tokens)
```

The bar changes colour as usage grows:
| Usage | Colour |
|---|---|
| 0 – 49% | Green |
| 50 – 79% | Yellow |
| 80 – 100% | Red |

A warning panel is also shown when the context reaches 80%, prompting you to `/save` then `/reset` before hitting the limit.

### Cost Estimation (`cost_estimator.py`)
Every turn is recorded by a `SessionCostTracker`. After each reply you see a per-turn cost line directly below the token count:

```
model: llama3.2:3b (local — free)
tokens → in: 1,102 | out: 141 | total: 1,243
```

Because `llama3.2:3b` runs locally via Ollama, it costs nothing. Type `/cost` at any point to see a full session summary including what the same conversation **would have cost** on paid cloud models — useful for understanding the value of running locally:

```
╭─────────────────── Cost Summary ────────────────────╮
│ Session: 6 turns | 4,210 in + 892 out tokens        │
│ Actual cost (llama3.2:3b): FREE (local)             │
│                                                     │
│ Equivalent cost on cloud models:                    │
│   gpt-4o-mini         $0.000664                     │
│   claude-haiku-4      $0.003368                     │
│   gpt-4o              $0.010525                     │
│   claude-sonnet-4     $0.012630                     │
╰─────────────────────────────────────────────────────╯
```

Pricing data lives in `cost_estimator.py` and can be updated independently of `chatbot.py`.

### Resilient Error Handling
The API call is isolated in `_call_ollama()` and wrapped with [Tenacity](https://tenacity.readthedocs.io/) for automatic retries. The two layers work together:

**Tenacity retry policy:**
- Retries up to **4 times** on `APIConnectionError` and `APIStatusError`
- Exponential backoff between attempts: 1s → 2s → 4s → 10s (capped)
- `APITimeoutError` is intentionally **not** retried — a timeout usually means Ollama is already overloaded, retrying immediately makes things worse
- `reraise=True` ensures the final failure surfaces to the error handlers below

**Per-error friendly panels (never a raw crash):**

| Error | What you see |
|---|---|
| Timeout (30s exceeded) | ⏱️ Yellow panel — Ollama might be busy, try again |
| Connection refused | 🔌 Red panel — tells you to run `ollama serve` |
| Rate limited (429) | ⚠️ Yellow panel — retried 4× and still failed |
| Server error (500/502/503) | 🔥 Red panel — Ollama may be overloaded |
| Context overflow (400) | ❌ Red panel — suggests `/reset` |
| Any other API error | ❌ Red panel with status code and message |
| Unexpected exception | 💥 Red panel with exception type and message |

In every error case the user message that was just appended to history is **popped back off**, keeping the conversation state clean so you can retry immediately.

### Conversation History
The full `messages[]` list is sent with every API request — this is how the model maintains context across turns. The chatbot manages this list manually, which demonstrates a core truth about LLMs: **they are stateless**, and the illusion of memory is entirely the application's responsibility.

---

## Project Structure

```
chatbot/
├── chatbot.py          # Main application — UI, commands, chat loop
├── cost_estimator.py   # Token cost calculation and session tracking
├── requirements.txt    # Python dependencies
├── chats/              # Saved conversation exports (auto-created)
└── venv/               # Python virtual environment
```

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) installed and running locally.

```bash
git clone https://github.com/Uzbekswe/chatbot
cd chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pull the model:

```bash
ollama pull llama3.2:3b
```

Run:

```bash
python3 chatbot.py
```

---

## Commands

| Command | Description |
|---|---|
| `/history` | Show the full conversation so far with turn numbers |
| `/cost` | Show session token usage and cloud cost equivalent |
| `/save` | Export the conversation to a timestamped file in `chats/` |
| `/reset` | Clear conversation history and reset cost tracker |
| `/help` | Show all available commands |
| `/quit` | Exit and print a session summary |

---

## Switching Models

Edit the `MODEL` constant at the top of `chatbot.py`:

```python
MODEL = "llama3.2:3b"   # change to any model you have pulled
```

And update `CONTEXT_WINDOW` to match the model's actual context size:

```python
CONTEXT_WINDOW = 128_000   # llama3.2:3b
```

If you switch to a paid API model, add its pricing to the `PRICING` dict in `cost_estimator.py`:

```python
PRICING = {
    "my-model": {"input": 1.00, "output": 5.00},  # USD per 1M tokens
    ...
}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | OpenAI-compatible client (used to talk to Ollama's local API) |
| `rich` | Terminal UI — panels, tables, progress bars, coloured text |
| `tenacity` | Automatic retries with exponential backoff on API failures |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## What This Project Demonstrates

- **LLMs are stateless** — the full `messages[]` array is rebuilt and sent on every request
- **Streaming mechanics** — how to correctly capture token usage from chunked responses
- **Context window management** — visual tracking and warnings before hitting the limit
- **Resilient API design** — separating network calls from UI logic so retries don't affect state
- **Cost transparency** — understanding the real value of local inference vs cloud APIs