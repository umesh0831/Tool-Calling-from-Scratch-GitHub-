import json

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from parser import StreamParser
from tools import (
    build_tools_system_prompt,
    execute_tool,
    format_tool_result,
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "qwen3:4b"
SYSTEM_PROMPT = (
    "You are a helpful personal assistant. "
    "You can send emails, schedule meetings, and cancel meetings. "
    "Use the provided tools when the user asks you to perform these actions. "
    "Always confirm what you did after using a tool.\n\n"
    + build_tools_system_prompt()
)


class ChatRequest(BaseModel):
    messages: list[dict]


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    SSE endpoint that streams the LLM response.

    Event types sent to the frontend:
      - "thinking"       : model's thinking tokens (from <think> blocks)
      - "token"          : a chunk of assistant text
      - "raw_tool_token" : raw tokens inside a <tool_call> block (visible parsing)
      - "tool_call"      : parsed tool call (name + args)
      - "tool_running"   : tool execution has started
      - "tool_result"    : tool execution finished (with result)
      - "tool_result_done": all tools done, context being sent back to LLM
      - "done"           : stream is complete
      - "error"          : something went wrong
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + request.messages

    async def event_stream():
        try:
            async for event in _run_chat_loop(messages):
                yield event
        except Exception as e:
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_stream())


async def _run_chat_loop(messages: list[dict]):
    """
    Core loop: send messages to Ollama, stream tokens through our custom
    parser, detect tool calls, execute them, feed results back, repeat.

    NO native Ollama tool calling — we parse <tool_call> tags ourselves.
    """
    while True:
        assistant_content = ""
        tool_calls = []
        parser = StreamParser()

        # Send the full context to the frontend before each LLM call
        yield {
            "event": "llm_input",
            "data": json.dumps({"messages": messages}),
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": MODEL,
                    "messages": messages,
                    # NO "tools" parameter — we handle it ourselves
                    "stream": True,
                    "think": True,
                },
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    chunk = json.loads(line)
                    msg = chunk.get("message", {})

                    # Stream thinking tokens
                    thinking = msg.get("thinking", "")
                    if thinking:
                        yield {
                            "event": "thinking",
                            "data": json.dumps({"content": thinking}),
                        }

                    # Feed content tokens through our parser
                    content = msg.get("content", "")
                    if content:
                        assistant_content += content
                        for event_type, event_data in parser.feed(content):
                            if event_type == "text":
                                yield {
                                    "event": "token",
                                    "data": json.dumps({"content": event_data}),
                                }
                            elif event_type == "tag_open":
                                yield {
                                    "event": "raw_tool_token",
                                    "data": json.dumps({"content": event_data, "type": "open"}),
                                }
                            elif event_type == "tag_buffer":
                                yield {
                                    "event": "raw_tool_token",
                                    "data": json.dumps({"content": event_data, "type": "buffer"}),
                                }
                            elif event_type == "tool_call":
                                tool_calls.append(event_data)
                                yield {
                                    "event": "tool_call",
                                    "data": json.dumps({
                                        "name": event_data.name,
                                        "arguments": event_data.arguments,
                                    }),
                                }
                            elif event_type == "parse_error":
                                yield {
                                    "event": "error",
                                    "data": json.dumps({"error": event_data}),
                                }

        # Flush any remaining buffered content
        for event_type, event_data in parser.flush():
            if event_type == "text":
                yield {
                    "event": "token",
                    "data": json.dumps({"content": event_data}),
                }
            elif event_type == "parse_error":
                yield {
                    "event": "error",
                    "data": json.dumps({"error": event_data}),
                }

        # If no tool calls, we're done
        if not tool_calls:
            yield {"event": "done", "data": json.dumps({"finished": True})}
            return

        # --- Tool call handling ---

        # Append the assistant's raw message to history
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        # Execute each tool call and feed results back
        for tc in tool_calls:
            yield {
                "event": "tool_running",
                "data": json.dumps({"name": tc.name}),
            }

            result = await execute_tool(tc.name, tc.arguments)

            yield {
                "event": "tool_result",
                "data": json.dumps({"name": tc.name, "result": json.loads(result)}),
            }

            # Append tool result as a user message with our custom format
            # (since we're not using Ollama's native tool role)
            messages.append({
                "role": "user",
                "content": format_tool_result(tc.name, result),
            })

        # Send the full context to the frontend
        yield {
            "event": "tool_result_done",
            "data": json.dumps({
                "message": "All tool results collected, sending back to LLM",
                "llm_input": messages,
            }),
        }

        tool_calls = []
