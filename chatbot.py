import os
from datetime import datetime

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table

# ── config ─────────────────────────────────────────────
MODEL = "qwen2.5:1.5b"
BASE_URL = "http://localhost:11434/v1"
CONTEXT_WINDOW = 32_000
TOKEN_WARN_THRESHOLD = int(CONTEXT_WINDOW * 0.8)
CHATS_DIR = "chats"
SYSTEM_PROMPT = "You are a helpful assistant. Be concise and clear."

# ── setup ──────────────────────────────────────────────
client = OpenAI(base_url=BASE_URL, api_key="ollama")
console = Console()
messages = []
session_stats = {"turns": 0, "total_tokens": 0}

os.makedirs(CHATS_DIR, exist_ok=True)


def render_token_bar(total_tokens: int) -> None:
    pct = total_tokens / CONTEXT_WINDOW
    used_pct = int(pct * 100)

    if pct < 0.5:
        color = "green"
    elif pct < 0.8:
        color = "yellow"
    else:
        color = "red"

    with Progress(
        TextColumn(f"[dim]context window[/dim]"),
        BarColumn(bar_width=30, style=color, complete_style=color),
        TextColumn(
            f"[{color}]{used_pct}%[/{color}] ({total_tokens:,} / {CONTEXT_WINDOW:,} tokens)"
        ),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=CONTEXT_WINDOW)
        progress.update(task, completed=total_tokens)


def chat(user_input: str) -> None:
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )

    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    session_stats["turns"] += 1
    session_stats["total_tokens"] = total_tokens

    console.print(f"\n[bold cyan]Assistant:[/bold cyan] {assistant_reply}\n")
    console.print(
        f"[dim]tokens → total: {total_tokens:,} | in: {input_tokens:,} | out: {output_tokens:,}[/dim]"
    )
    render_token_bar(total_tokens)
    console.print()

    if total_tokens >= TOKEN_WARN_THRESHOLD:
        console.print(
            Panel(
                f"⚠️  Context window at {int(total_tokens / CONTEXT_WINDOW * 100)}% — consider /save then /reset.",
                style="bold red",
            )
        )


def cmd_reset() -> None:
    global messages
    messages = []
    session_stats["turns"] = 0
    session_stats["total_tokens"] = 0
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
        role_color = "cyan" if msg["role"] == "assistant" else "yellow"
        table.add_row(
            str(i),
            f"[{role_color}]{msg['role']}[/{role_color}]",
            msg["content"],
        )

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
        f.write(
            f"Turns: {len(messages)} | Tokens used: {session_stats['total_tokens']:,}\n"
        )
        f.write("=" * 50 + "\n\n")
        for msg in messages:
            f.write(f"[{msg['role'].upper()}]\n{msg['content']}\n\n")

    console.print(f"[bold green]Saved →[/bold green] {filepath}\n")


def show_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column(style="dim")
    for cmd, desc in [
        ("/reset", "Clear conversation history"),
        ("/history", "Show full conversation so far"),
        ("/save", "Export conversation to chats/"),
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
            f"Session summary — "
            f"[bold]{session_stats['turns']}[/bold] turns | "
            f"[bold]{session_stats['total_tokens']:,}[/bold] tokens used",
            style="dim",
        )
    )


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
            case _:
                chat(user_input)


if __name__ == "__main__":
    main()
