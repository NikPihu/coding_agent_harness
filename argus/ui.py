"""The seam between the agent and whatever draws it.

Every module does `from .ui import ui` and calls methods on it - ui.note(),
ui.tool(), ui.approve() and so on. Nothing here draws anything: `ui` forwards
each call to a backend, and the app (app.py) installs its backend at start-up.
That keeps the agent loop free of any knowledge of Textual.

The methods a backend provides:

  output     user(text, attached)  note(text)  error(text)  attached(labels)
             tool(name, args, result, nested=False)  subagent(description)
             todos(todos)  usage(stats)  summary()  compacted(before, messages)
             injection(text)  debug(data)
  replies    reply_start()  reply_event(kind, text)  reply_end(reply)  reply_abort()
  busy       working(label)  -> context manager
  questions  approve(reason) -> bool     pick(title, rows) -> index or None
             secret(title, hint) -> str or None
  screen     clear()  banner()  resumed(messages, label)  replay(messages)
"""

import re


class UI:
    backend = None

    def install(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        if self.backend is None:
            raise RuntimeError(f"ui.{name}() called before the app installed a UI")
        return getattr(self.backend, name)


ui = UI()


def thought_title(reasoning, low=5, high=10):
    """A 5-10 word heading for a block of reasoning: its opening sentence,
    trimmed. Free - no second model call just to label a dropdown."""
    text = re.sub(r"[`*_#>\[\]]", "", reasoning or "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    words = []
    for sentence in sentences:
        words += sentence.split()
        if len(words) >= low:
            break
    if not words:
        return "Thought"
    title = " ".join(words[:high])
    readable = sum(ch.isascii() or ch.isalpha() for ch in title)
    if "�" in title or readable < 0.8 * len(title):
        return "Thought"
    return title.rstrip(".,;:") + ("…" if len(words) > high else "")
