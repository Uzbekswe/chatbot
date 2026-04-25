# Terminal Chatbot

A multi-turn terminal chatbot powered by a local LLM via Ollama.
Built as a hands-on exercise to understand conversation state in LLMs.

## What this demonstrates
- LLMs are stateless — conversation history is managed manually
- The full `messages[]` array is sent with every API request
- Token counting and context window monitoring

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally

```bash
git clone https://github.com/Uzbekswe/chatbot
cd chatbot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Pull a model:
```bash
ollama pull qwen2.5:1.5b
```

Run:
```bash
python3 chatbot.py
```

## Commands
| Command | Description |
|---|---|
| `/reset` | Clear conversation history |
| `/history` | Show all turns so far |
| `/save` | Export chat to `chats/` folder |
| `/help` | Show commands |
| `/quit` | Exit |

## Switch to a different model
Edit the `MODEL` variable in `chatbot.py`:
```python
MODEL = "qwen2.5:1.5b"   # or llama3.2, mistral, etc.
```
