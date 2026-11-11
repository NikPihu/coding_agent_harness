import os
import platform
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

from . import providers
from .skills import skills_prompt
from .tools import TOOL_SCHEMAS

SYSTEM_PROMPT = f"""
You are Argus, a coding agent. Your job is to code. Always code.
Use the bash tool to inspect files.
Use write_file to create files and str_replace to edit them.
Answer back to the user once exploration is done.

For any task that takes more than one step, call write_todos first and plan it
out. Send the whole list every time you call it - it replaces the old one.
Keep exactly one task in_progress, mark it done the moment it is finished, and
move the next one to in_progress in the same call. Do not batch up completions
at the end. Skip the tool entirely for single-step tasks; it is noise there.

The current list is injected back to you every turn inside <todos> tags, so
that block - not the transcript - is the truth about where you are.

When you need to understand how something works - where a feature lives, how
data flows, what calls what - send a task subagent instead of grepping your
way there yourself. It explores in its own context window and hands you back
just the findings, so the search does not fill yours. It cannot see this
conversation, so write the question so it stands alone. Do all editing
yourself; the subagent only reads.

Long tool output is cut short, and the whole thing is written to a temp file
whose path is given at the cut. Page through it with head, tail, sed -n or
grep rather than asking for it again. That file only exists for the current
turn, so read it now or re-run the command later.

Your current working directory is: {os.getcwd()}
Operating system: {platform.system()}. The bash tool runs a POSIX shell{' (Git Bash on Windows - use forward slashes, e.g. C:/Users/...)' if sys.platform == 'win32' else ''}.

You have skills available. Each one is a set of instructions for a task.
If a skill matches what the user wants, call read_skill first and follow it.

{skills_prompt()}
"""


import threading

# Set by Esc in the app. Checked between stream chunks and between tool calls,
# by the main agent and by subagents alike.
CANCEL = threading.Event()


class Interrupted(Exception):
    """The user pressed Esc while the model was answering."""


@dataclass
class Reply:
    """One assistant turn, assembled from the stream.

    tool_calls mimic the SDK objects (call.id, call.function.name,
    call.function.arguments) so tools.execute() does not care where they came
    from.
    """

    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)

    def to_message(self):
        """What goes into the transcript. Reasoning is left out: it was for this
        turn only, and sending it back would just burn context."""
        message = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in self.tool_calls
            ]
        return {k: v for k, v in message.items() if v is not None}


class ThinkSplitter:
    """Some models (and some servers) put their reasoning inline as
    <think>...</think> at the start of the content. Peel it off as it streams."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buffer = ""
        self.state = "start"  # start -> thinking -> answer

    def feed(self, text):
        """Returns [(kind, text)] with kind "reasoning" or "content"."""
        if self.state == "answer":
            return [("content", text)]
        self.buffer += text
        out = []
        if self.state == "start":
            stripped = self.buffer.lstrip()
            if len(stripped) < len(self.OPEN) and self.OPEN.startswith(stripped):
                return []  # could still be the start of <think>
            if stripped.startswith(self.OPEN):
                self.state = "thinking"
                self.buffer = stripped[len(self.OPEN):]
            else:
                self.state = "answer"
                out.append(("content", self.buffer))
                self.buffer = ""
                return out
        if self.state == "thinking":
            if self.CLOSE in self.buffer:
                thought, _, rest = self.buffer.partition(self.CLOSE)
                if thought:
                    out.append(("reasoning", thought))
                self.state = "answer"
                self.buffer = ""
                if rest.lstrip():
                    out.append(("content", rest.lstrip()))
            else:
                keep = len(self.CLOSE) - 1  # a tag may be split across chunks
                ready, self.buffer = self.buffer[:-keep], self.buffer[-keep:]
                if ready:
                    out.append(("reasoning", ready))
        return out

    def flush(self):
        rest, self.buffer = self.buffer, ""
        if not rest:
            return []
        return [("reasoning" if self.state == "thinking" else "content", rest)]


def _reasoning_of(delta):
    """Providers disagree on the field name: Ollama and OpenRouter say
    `reasoning`, DeepSeek-style servers say `reasoning_content`."""
    extra = getattr(delta, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        value = extra.get(key) or getattr(delta, key, None)
        if isinstance(value, str) and value:
            return value
    return ""


def call_llm(messages, tools=None, on_event=None, cancel=None):
    """Stream one reply. Returns (Reply, usage).

    on_event(kind, text) is called as pieces arrive, kind being "reasoning" or
    "content". cancel is a threading.Event; setting it stops the stream and
    raises Interrupted.
    """
    emit = on_event or (lambda kind, text: None)
    cancel = cancel or CANCEL
    stream = providers.client().chat.completions.create(
        messages=messages,
        tools=tools or TOOL_SCHEMAS,
        stream=True,
        stream_options={"include_usage": True},
        **providers.request_options(),
    )

    reply, splitter, usage_info = Reply(), ThinkSplitter(), None
    calls = {}  # index -> {"id", "name", "arguments"}

    def take(kind, text):
        if kind == "reasoning":
            reply.reasoning += text
        else:
            reply.content += text
        emit(kind, text)

    try:
        for chunk in stream:
            if cancel.is_set():
                raise Interrupted()
            if getattr(chunk, "usage", None):
                usage_info = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            thought = _reasoning_of(delta)
            if thought:
                take("reasoning", thought)
            if delta.content:
                for kind, text in splitter.feed(delta.content):
                    take(kind, text)
            for piece in delta.tool_calls or []:
                slot = calls.setdefault(piece.index, {"id": None, "name": "", "arguments": ""})
                if piece.id:
                    slot["id"] = piece.id
                if piece.function and piece.function.name:
                    slot["name"] += piece.function.name
                if piece.function and piece.function.arguments:
                    slot["arguments"] += piece.function.arguments
    finally:
        stream.close()

    for kind, text in splitter.flush():
        take(kind, text)
    reply.content = reply.content.strip()

    for index in sorted(calls):
        slot = calls[index]
        reply.tool_calls.append(SimpleNamespace(
            id=slot["id"] or f"call_{index}",
            function=SimpleNamespace(name=slot["name"], arguments=slot["arguments"] or "{}"),
        ))

    completion_details = getattr(usage_info, "completion_tokens_details", None)
    prompt_details = getattr(usage_info, "prompt_tokens_details", None)
    usage = {
        "prompt_tokens": getattr(usage_info, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_info, "completion_tokens", 0) or 0,
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "cached_tokens": getattr(prompt_details, "cached_tokens", None),
    }
    return reply, usage
