# Tool Calling from Scratch

LLM tool calling implemented without any framework — no LangChain, no OpenAI
function-calling API. The model outputs tool invocations as raw XML in its token
stream. A custom parser detects and executes those tags incrementally as tokens
arrive, then injects results back into the conversation loop.

Built to understand the mechanism underneath every framework that abstracts it away.

---

## Why Build From Scratch?

Ollama, OpenAI, and Anthropic all have built-in tool calling that intercepts the
model's raw token stream and returns clean structured data. This project deliberately
skips that layer to understand what's underneath:

- Tool definitions are injected into the **system prompt** as structured text
- The model outputs `<tool_call>` XML tags in its streamed response
- A custom parser detects those tags **token-by-token** as they stream
- Tool results are injected back into the conversation as `<tool_response>` messages
- The model continues until it produces a final text answer

No hidden API modes. No framework magic. Every step is visible in the UI.

---

## How It Works

```
System Prompt (tool schemas injected as text)
    │
    ▼
LLM streams response tokens (Ollama qwen3:4b)
    │
    ▼
parser.py — incremental XML parser     → detects <tool_call> mid-stream
    │
    ▼
tools.py — tool executor               → runs matched tool, returns result
    │
    ▼
Context injection                      → tool result appended to conversation
    │
    ▼
LLM continues                          → generates final answer
```

The web UI exposes every stage: raw model output, parsed tool calls, execution
timing, and the complete LLM context window at each round.

---

## Tech Stack

| Component       | Tool                               |
|-----------------|------------------------------------|
| Backend         | FastAPI + Server-Sent Events (SSE) |
| LLM             | Ollama (qwen3:4b, ~2.5 GB)        |
| Stream Parser   | Custom incremental XML parser      |
| Frontend        | Vanilla HTML / CSS / JS            |
| Package Manager | uv                                 |

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed

```bash
ollama pull qwen3:4b

git clone https://github.com/Rahulroy5/tool-calling-from-scratch.git
cd tool-calling-from-scratch
./start.sh
```

Open **http://localhost:8000** in your browser.

**Try:** "Send an email to john@example.com about tomorrow's standup" or
"Schedule a meeting with alice@co.com on 2026-04-01 at 10:00"

---

## Project Structure

```
tool-calling-from-scratch/
├── main.py          # FastAPI backend — SSE streaming + tool orchestration loop
├── parser.py        # Incremental stream parser — detects XML tags token-by-token
├── tools.py         # Tool schemas, system prompt builder, stub implementations
├── static/
│   ├── index.html   # Chat UI with pipeline visualization
│   └── styles.css
└── start.sh         # Ollama check + server launch
```

---

## Available Tools

| Tool               | Parameters                       |
|--------------------|----------------------------------|
| `send_mail`        | to, subject, body                |
| `schedule_meeting` | title, attendees[], date, time   |
| `cancel_meeting`   | title, date                      |

Tools include a 3-second delay to make execution visible in the UI.

---

## What I Learned

Tool calling is fundamentally a prompt engineering problem. The model has no special
mode — it follows formatting instructions in the system prompt. The hard engineering
is writing a parser robust enough to handle partial XML tags mid-stream, and managing
conversation context correctly as tool results are injected between turns. This is
exactly what every major SDK does internally; building it manually made those
abstractions legible.
