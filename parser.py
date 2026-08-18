"""
Real-time stream parser for tool calls.

As tokens arrive one at a time from the LLM, this parser detects
<tool_call>...</tool_call> blocks incrementally and extracts the
tool name and arguments.

State machine:
  TEXT -> saw '<tool_call>' -> INSIDE_TAG (buffering) -> saw '</tool_call>' -> PARSE -> TEXT
"""

import json
from dataclasses import dataclass, field

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict
    raw: str  # The raw text inside the tags


@dataclass
class StreamParser:
    """
    Feed tokens one at a time via feed(). It returns a list of events
    for each token:

      ("text", str)       - regular text to show to user
      ("tag_open", str)   - <tool_call> tag detected
      ("tag_buffer", str) - new tokens inside tool_call (for live display)
      ("tool_call", ParsedToolCall) - complete tool call parsed
      ("parse_error", str) - tool call block couldn't be parsed
    """

    # Accumulated text not yet decided (may contain partial tags)
    _pending: str = field(default="", repr=False)
    # Full content between <tool_call> and </tool_call> (kept for parsing)
    _buffer: str = field(default="", repr=False)
    # How much of _buffer has already been emitted as tag_buffer events
    _emitted: int = field(default=0, repr=False)
    _inside_tag: bool = field(default=False, repr=False)

    def feed(self, token: str) -> list[tuple]:
        """Feed a token and get back a list of events."""
        self._pending += token
        return self._process()

    def _process(self) -> list[tuple]:
        events = []
        changed = True

        while self._pending and changed:
            changed = False

            if self._inside_tag:
                # Append pending into buffer
                self._buffer += self._pending
                self._pending = ""

                # Check for close tag in the full buffer
                close_pos = self._buffer.find(CLOSE_TAG)
                if close_pos != -1:
                    # Found close tag — extract content and parse
                    content = self._buffer[:close_pos].strip()
                    after = self._buffer[close_pos + len(CLOSE_TAG):]

                    # Emit any un-emitted buffer content up to close tag
                    unemitted = self._buffer[self._emitted:close_pos]
                    if unemitted:
                        events.append(("tag_buffer", unemitted))
                    events.append(("tag_buffer", CLOSE_TAG))

                    # Parse JSON
                    try:
                        parsed = json.loads(content)
                        events.append(("tool_call", ParsedToolCall(
                            name=parsed["name"],
                            arguments=parsed["arguments"],
                            raw=content,
                        )))
                    except (json.JSONDecodeError, KeyError) as e:
                        events.append(("parse_error", f"Failed to parse tool call: {e}\nRaw: {content}"))

                    # Reset state
                    self._buffer = ""
                    self._emitted = 0
                    self._inside_tag = False
                    self._pending = after
                    changed = True
                else:
                    # No close tag yet. Emit new safe content for live display.
                    # Hold back chars that could be a partial </tool_call>
                    safe_end = len(self._buffer)
                    for i in range(1, min(len(self._buffer), len(CLOSE_TAG)) + 1):
                        if CLOSE_TAG.startswith(self._buffer[-i:]):
                            safe_end = len(self._buffer) - i
                            break

                    new_safe = self._buffer[self._emitted:safe_end]
                    if new_safe:
                        events.append(("tag_buffer", new_safe))
                        self._emitted = safe_end

            else:
                # Outside tag — look for open tag
                open_pos = self._pending.find(OPEN_TAG)
                if open_pos != -1:
                    if open_pos > 0:
                        events.append(("text", self._pending[:open_pos]))
                    events.append(("tag_open", OPEN_TAG))
                    self._pending = self._pending[open_pos + len(OPEN_TAG):]
                    self._inside_tag = True
                    self._buffer = ""
                    self._emitted = 0
                    changed = True
                else:
                    # Check if end of pending could be start of <tool_call>
                    safe_end = len(self._pending)
                    for i in range(1, min(len(self._pending), len(OPEN_TAG)) + 1):
                        if OPEN_TAG.startswith(self._pending[-i:]):
                            safe_end = len(self._pending) - i
                            break

                    if safe_end > 0:
                        events.append(("text", self._pending[:safe_end]))
                        self._pending = self._pending[safe_end:]
                    # else: pending is just a partial tag prefix, wait for more

        return events

    def flush(self) -> list[tuple]:
        """Flush any remaining buffered content (call at end of stream)."""
        events = []
        if self._inside_tag:
            events.append(("parse_error", f"Unclosed <tool_call> tag. Buffer: {self._buffer}"))
            self._buffer = ""
            self._emitted = 0
            self._inside_tag = False
        if self._pending:
            events.append(("text", self._pending))
            self._pending = ""
        return events
