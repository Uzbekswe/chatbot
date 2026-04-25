import logging
import os
from datetime import datetime

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from cost_estimator import SessionCostTracker, estimate

# ── config ─────────────────────────────────────────────
MODEL = "llama3.2:3b"
BASE_URL = "http://localhost:11434/v1"
CONTEXT_WINDOW = 128_000
TOKEN_WARN_THRESHOLD = int(CONTEXT_WINDOW * 0.8)
CHATS_DIR = "chats"
SYSTEM_PROMPT = "You are a helpful assistant. Be concise and clear."

# ── setup ──────────────────────────────────────────────
client = OpenAI(base_url=BASE_URL, api_key="ollama")
cost_tracker = SessionCostTracker(MODEL)
console = Console()
messages = []
session_stats = {"turns": 0, "total_tokens": 0}

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

os.makedirs(CHATS_DIR, exist_ok=True)


# ── token bar ──────────────────────────────────────────
def render_token_bar(total_tokens: int) -> None:
    pct = total_tokens / CONTEXT_WINDOW
    used_pct = int(pct * 100)
    color = "green" if pct < 0.5 else "yellow" if pct < 0.8 else "red"

    with Progress(
        TextColumn("[dim]context window[/dim]"),
        BarColumn(bar_width=30, style=color, complete_style=color),
        TextColumn(
            f"[{color}]{used_pct}%[/{color}] ({total_tokens:,} / {CONTEXT_WINDOW:,} tokens)"
        ),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=CONTEXT_WINDOW)
        progress.update(task, completed=total_tokens)


# ── retry logic ────────────────────────────────────────
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((APIStatusError, APIConnectionError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_ollama(msgs: list) -> tuple[str, int, int, int]:
    """
    Makes the actual API call. Separated from chat() so
    tenacity only wraps the network call, not our UI logic.

    stream_options include_usage=True tells Ollama to attach
    usage stats on the final chunk of the stream.

    Returns: (reply, total_tokens, input_tokens, output_tokens)
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + msgs,
        stream=True,
        stream_options={
            "include_usage": True
        },  # ← makes Ollama send usage on last chunk
        timeout=30,
    )

    console.print("\n[bold cyan]Assistant:[/bold cyan] ", end="")

    full_reply = ""
    last_usage = None

    for chunk in stream:
        # Ollama sends a final usage-only chunk with an empty choices list —
        # guard against it so we don't get an IndexError
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_reply += delta

        # usage lands on the last chunk, not on the stream object itself
        if hasattr(chunk, "usage") and chunk.usage is not None:
            last_usage = chunk.usage

    print("\n")

    if last_usage:
        return (
            full_reply,
            last_usage.total_tokens,
            last_usage.prompt_tokens,
            last_usage.completion_tokens,
        )

    return full_reply, 0, 0, 0


# ── chat ───────────────────────────────────────────────
def chat(user_input: str) -> None:
    messages.append({"role": "user", "content": user_input})

    try:
        reply, total, inp, out = _call_ollama(messages)

        messages.append({"role": "assistant", "content": reply})
        session_stats["turns"] += 1
        session_stats["total_tokens"] = total

        if total:
            turn_cost = cost_tracker.record(inp, out)
            console.print(
                f"[dim]tokens → total: {total:,} | in: {inp:,} | out: {out:,}[/dim]"
            )
            console.print(f"[dim]{turn_cost}[/dim]")
            render_token_bar(total)
        else:
            console.print("[dim]tokens → (unavailable in stream mode)[/dim]")
            render_token_bar(session_stats["total_tokens"])

        if total >= TOKEN_WARN_THRESHOLD:
            console.print(
                Panel(
                    f"⚠️  Context at {int(total / CONTEXT_WINDOW * 100)}%"
                    " — consider /save then /reset.",
                    style="bold red",
                )
            )

    except APITimeoutError:
        # remove the user message we just appended — call never completed
        messages.pop()
        console.print(
            Panel(
                "⏱️  Request timed out after 30s. Ollama might be busy. Try again.",
                style="bold yellow",
            )
        )

    except APIConnectionError:
        messages.pop()
        console.print(
            Panel(
                "🔌 Cannot reach Ollama. Is it running?\n"
                "Start it with: [bold]ollama serve[/bold]",
                style="bold red",
            )
        )

    except APIStatusError as e:
        messages.pop()
        if e.status_code == 429:
            console.print(
                Panel(
                    "⚠️  Rate limited (429). Retried 4 times and still failed. Wait a moment.",
                    style="bold yellow",
                )
            )
        elif e.status_code in (500, 502, 503):
            console.print(
                Panel(
                    f"🔥 Server error ({e.status_code}). Retried 4 times. Ollama may be overloaded.",
                    style="bold red",
                )
            )
        elif e.status_code == 400:
            console.print(
                Panel(
                    "❌ Bad request (400) — possibly context overflow.\n"
                    "Try /reset to clear history.",
                    style="bold red",
                )
            )
        else:
            console.print(
                Panel(
                    f"❌ API error {e.status_code}: {e.message}",
                    style="bold red",
                )
            )

    except Exception as e:
        messages.pop()
        console.print(
            Panel(
                f"💥 Unexpected error: {type(e).__name__}: {e}\n"
                "Your history is intact. Try again.",
                style="bold red",
            )
        )

    console.print()


# ── commands ───────────────────────────────────────────
def cmd_reset() -> None:
    global messages
    messages = []
    session_stats["turns"] = 0
    session_stats["total_tokens"] = 0
    cost_tracker.reset()
    console.print("[bold red]History cleared.[/bold red] Starting fresh.\n")


def cmd_history() -> None:
    if not messages:
        console.print("[dim]No history yet.[/dim]\n")
        return

    table = Table(title="Conversation History", show_lines=True)
    table.add_column("Turn", style="dim", width=6)
    table.add_column("Role", style="bold", width=10)
    table.add_column("Message")

    for i, msg in enumerate(messages, 1):
        color = "cyan" if msg["role"] == "assistant" else "yellow"
        table.add_row(str(i), f"[{color}]{msg['role']}[/{color}]", msg["content"])

    console.print(table)
    console.print(
        f"[dim]Total turns: {len(messages)} | Last token count: {session_stats['total_tokens']:,}[/dim]\n"
    )


def cmd_save() -> None:
    if not messages:
        console.print("[dim]Nothing to save yet.[/dim]\n")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(CHATS_DIR, f"chat_{timestamp}.txt")

    with open(filepath, "w") as f:
        f.write(f"Chat saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Turns: {len(messages)} | Tokens: {session_stats['total_tokens']:,}\n")
        f.write("=" * 50 + "\n\n")
        for msg in messages:
            f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n")

    console.print(f"[bold green]Saved →[/bold green] {filepath}\n")


def cmd_cost() -> None:
    console.print(Panel(cost_tracker.summary(), title="Cost Summary", style="cyan"))
    console.print()


def show_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="dim")
    for cmd, desc in [
        ("/reset", "Clear conversation history"),
        ("/history", "Show full conversation so far"),
        ("/save", "Export conversation to chats/"),
        ("/cost", "Show session token cost summary"),
        ("/help", "Show this menu"),
        ("/quit", "Exit"),
    ]:
        table.add_row(cmd, desc)
    console.print(table)
    console.print()


def show_session_summary() -> None:
    if session_stats["turns"] == 0:
        return
    console.print(
        Panel(
            f"Session summary — [bold]{session_stats['turns']}[/bold] turns | "
            f"[bold]{session_stats['total_tokens']:,}[/bold] tokens used",
            style="dim",
        )
    )


# ── main ───────────────────────────────────────────────
def main():
    console.print(
        Panel(
            f"[bold green]Chatbot ready[/bold green]  ·  model: [cyan]{MODEL}[/cyan]\n"
            "Type a message to chat, or [bold cyan]/help[/bold cyan] for commands.",
            style="green",
        )
    )

    while True:
        try:
            user_input = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()
            show_session_summary()
            console.print("[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        match user_input:
            case "/quit":
                show_session_summary()
                console.print("[dim]Bye![/dim]")
                break
            case "/help":
                show_help()
            case "/reset":
                cmd_reset()
            case "/history":
                cmd_history()
            case "/save":
                cmd_save()
            case "/cost":
                cmd_cost()
            case _:
                chat(user_input)


if __name__ == "__main__":
    main()
