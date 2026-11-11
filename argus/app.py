"""The full-screen app: chat above, input below, status bar at the bottom.

The agent runs on a worker thread (Conversation.submit blocks while the model
streams and tools run). TextualUI is the backend `ui` forwards to: it hops
every call over to the UI thread with call_from_thread, and for questions
(approve, pick, secret) it opens a dialog and waits for the answer.
"""

import json
import os
import threading
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from rich.console import Console
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Provider
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from . import commands, providers
from .content import text_of
from .context import git
from .llm import CANCEL
from .todos import MARKS
from .ui import thought_title, ui

LOGO = r"""
 █████╗ ██████╗  ██████╗ ██╗   ██╗███████╗
██╔══██╗██╔══██╗██╔════╝ ██║   ██║██╔════╝
███████║██████╔╝██║  ███╗██║   ██║███████╗
██╔══██║██╔══██╗██║   ██║██║   ██║╚════██║
██║  ██║██║  ██║╚██████╔╝╚██████╔╝███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝"""

TIPS = [
    ("/connect", "add an OpenRouter key or use local models"),
    ("/model", "switch model            ctrl+o"),
    ("/help", "all commands and keys"),
    ("@file", "attach a file or image"),
    ("esc", "stop the agent"),
]

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HISTORY_FILE = Path.home() / ".argus" / "history.jsonl"
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", "dist", "build"}
MAX_TOOL_LINES = 60
FOLD = {"collapsed_symbol": "▸", "expanded_symbol": "▾"}


# =================================================================== widgets


class UserMessage(Static):
    def __init__(self, text, attached=()):
        body = Text(text)
        for label in attached:
            body.append(f"\n📎 {label}", style="italic")
        super().__init__(body)


class Note(Static):
    pass


class ErrorNote(Static):
    pass


class Thinking(Collapsible):
    """The model's reasoning: a dim, folded dropdown under a short heading."""

    def __init__(self, collapsed=True):
        self.text = ""
        self.started = time.monotonic()
        self.body = Static("", classes="thought-text", markup=False)
        super().__init__(self.body, title="Thinking…", collapsed=collapsed, **FOLD)

    def add(self, text):
        self.text += text
        self.body.update(self.text)

    def finish(self):
        seconds = time.monotonic() - self.started
        self.title = f"{thought_title(self.text)}   ·  thought for {seconds:.0f}s"


class AssistantMessage(Vertical):
    """One reply. Thinking (if any) on top, the answer below, both streamed."""

    def __init__(self, show_thinking=False):
        super().__init__()
        self.show_thinking = show_thinking
        self.thinking = None
        self.answer = None
        self.answer_text = ""

    def add(self, kind, text):
        if kind == "reasoning":
            if self.thinking is None:
                self.thinking = Thinking(collapsed=not self.show_thinking)
                self.mount(self.thinking, before=self.answer)
            self.thinking.add(text)
        else:
            self.answer_text += text
            if self.answer is None:
                # Born with its text: an update() sent before the widget has
                # finished mounting is silently lost.
                self.answer = Markdown(self.answer_text)
                self.mount(self.answer)
            else:
                self.answer.update(self.answer_text)

    def finish(self, content=None):
        if self.thinking is not None:
            self.thinking.finish()
        if content is not None:
            self.answer_text = content
        if self.answer_text:
            if self.answer is None:
                self.answer = Markdown(self.answer_text)
                self.mount(self.answer)
            else:
                self.answer.update(self.answer_text)
        if self.thinking is None and self.answer is None:
            self.display = False  # tool calls only - nothing to show here


def describe_args(name, args):
    if not isinstance(args, dict):
        return ""
    if name == "bash":
        return args.get("command", "")
    if name in ("read_file", "write_file", "str_replace"):
        return args.get("path", "")
    if name == "read_skill":
        return args.get("name", "")
    if name == "task":
        return " ".join(str(args.get("description", "")).split())[:80]
    return json.dumps(args)[:80]


class ToolCall(Collapsible):
    def __init__(self, name, args, result, nested=False):
        lines = (result or "(no output)").rstrip().splitlines() or ["(no output)"]
        shown = "\n".join(lines[:MAX_TOOL_LINES])
        if len(lines) > MAX_TOOL_LINES:
            shown += f"\n… {len(lines) - MAX_TOOL_LINES} more lines"
        failed = (result or "").startswith(("Error", "Blocked", "The user denied", "Timed out"))
        icon = "✗" if failed else "•"
        title = f"{icon} {name}  {describe_args(name, args)}"
        classes = "failed" if failed else ""
        if nested:
            classes += " nested"
        super().__init__(
            Static(shown, classes="tool-output", markup=False),
            title=title[:160],
            collapsed=True,
            classes=classes,
            **FOLD,
        )


class Welcome(Static):
    def __init__(self):
        body = Text(LOGO.strip("\n"), style="bold")
        body.append("\n\n")
        for key, text in TIPS:
            body.append(f"  {key:<10}", style="bold")
            body.append(f"{text}\n", style="dim")
        super().__init__(body)


class TodoPanel(Static):
    def show(self, todos):
        if not todos:
            self.display = False
            return
        done = sum(1 for t in todos if t["status"] == "done")
        body = Text(f"todos {done}/{len(todos)}\n", style="bold")
        for todo in todos:
            style = {"done": "dim strike", "in_progress": "bold", "pending": "dim"}[todo["status"]]
            body.append(f"{MARKS[todo['status']]} {todo['content']}\n", style=style)
        self.update(body)
        self.display = True


class StatusBar(Static):
    pass


# ================================================================ the input


class ChatInput(TextArea):
    """Enter sends, ctrl+j adds a line. Up/down/tab/esc drive the popup when
    it is open, and walk your message history when it is not."""

    class Submitted(Message):
        def __init__(self, text):
            super().__init__()
            self.text = text

    async def _on_key(self, event: events.Key):
        app = self.app
        popup = app.suggestions_open()
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            if popup and app.accept_suggestion(submit=True):
                return
            text = self.text
            if text.strip():
                self.post_message(self.Submitted(text))
            return
        if event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        if popup and event.key in ("up", "down"):
            event.prevent_default()
            event.stop()
            app.move_suggestion(-1 if event.key == "up" else 1)
            return
        if popup and event.key == "tab":
            event.prevent_default()
            event.stop()
            app.accept_suggestion(submit=False)
            return
        if popup and event.key == "escape":
            event.prevent_default()
            event.stop()
            app.hide_suggestions()
            return
        if event.key == "up" and self.cursor_location[0] == 0:
            event.prevent_default()
            event.stop()
            app.recall(-1)
            return
        if event.key == "down" and self.cursor_location[0] == self.document.line_count - 1:
            event.prevent_default()
            event.stop()
            app.recall(1)
            return


# ================================================================ dialogs


class PickScreen(ModalScreen):
    """A list to choose from, with a filter box. Returns the row index."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, title, rows, search=True):
        super().__init__()
        self.title_text = title
        self.rows = rows
        self.search = search

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.title_text, classes="dialog-title")
            if self.search:
                yield Input(placeholder="type to filter…", id="filter")
            yield OptionList(*self.options(""), id="choices")
            yield Static("enter select  ·  esc cancel", classes="dialog-hint")

    def options(self, query):
        query = query.lower()
        return [Option(row, id=str(i)) for i, row in enumerate(self.rows) if query in row.lower()]

    def on_mount(self):
        self.query_one("#filter" if self.search else "#choices").focus()
        choices = self.query_one("#choices", OptionList)
        if choices.option_count:
            choices.highlighted = 0

    @on(Input.Changed, "#filter")
    def refilter(self, event):
        choices = self.query_one("#choices", OptionList)
        choices.clear_options()
        choices.add_options(self.options(event.value))
        if choices.option_count:
            choices.highlighted = 0

    def on_key(self, event: events.Key):
        if not self.search or event.key not in ("up", "down"):
            return
        choices = self.query_one("#choices", OptionList)
        choices.action_cursor_up() if event.key == "up" else choices.action_cursor_down()
        event.stop()

    @on(Input.Submitted, "#filter")
    def choose_highlighted(self):
        choices = self.query_one("#choices", OptionList)
        if choices.highlighted is not None:
            self.dismiss(int(choices.get_option_at_index(choices.highlighted).id))

    @on(OptionList.OptionSelected, "#choices")
    def chosen(self, event):
        self.dismiss(int(event.option.id))

    def action_cancel(self):
        self.dismiss(None)


class SecretScreen(ModalScreen):
    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, title, hint=""):
        super().__init__()
        self.title_text, self.hint = title, hint

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(self.title_text, classes="dialog-title")
            if self.hint:
                yield Static(self.hint, classes="dialog-hint")
            yield Input(password=True, placeholder="paste here", id="secret")
            yield Static("enter save  ·  esc cancel", classes="dialog-hint")

    def on_mount(self):
        self.query_one("#secret").focus()

    @on(Input.Submitted, "#secret")
    def save(self, event):
        self.dismiss(event.value.strip() or None)

    def action_cancel(self):
        self.dismiss(None)


class ApproveScreen(ModalScreen):
    BINDINGS = [
        Binding("y", "answer(True)", "allow"),
        Binding("n", "answer(False)", "deny"),
        Binding("escape", "answer(False)", "deny"),
    ]

    def __init__(self, reason):
        super().__init__()
        self.reason = reason

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog approve"):
            yield Static("Allow this?", classes="dialog-title")
            yield Static(self.reason, classes="approve-reason", markup=False)
            yield Static("y allow  ·  n deny", classes="dialog-hint")

    def action_answer(self, allowed):
        self.dismiss(allowed)


# ======================================================= the ui backend


class TextualUI:
    """What `ui` forwards to while the app runs. Called from the worker thread."""

    FLUSH_EVERY = 0.06  # seconds between screen updates while streaming

    def __init__(self, app):
        self.app = app
        self.totals = {}
        self.current = None
        self.pending = []
        self.last_flush = 0.0
        self.lock = threading.Lock()

    # -- plumbing

    def _call(self, fn, *args):
        if threading.current_thread() is threading.main_thread():
            return fn(*args)
        return self.app.call_from_thread(fn, *args)

    def _ask(self, screen):
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("questions must be asked from the worker thread")
        answered, box = threading.Event(), {}

        def done(result):
            box["result"] = result
            answered.set()

        self.app.call_from_thread(self.app.push_screen, screen, done)
        answered.wait()
        return box.get("result")

    def _post(self, widget):
        self._call(self.app.post_widget, widget)

    # -- output

    def user(self, text, attached=()):
        self._post(UserMessage(text, attached))

    def note(self, text):
        self._post(Note(Text(str(text))))

    def error(self, text):
        self._post(ErrorNote(Text(str(text))))

    def attached(self, labels):
        self._call(self.app.show_pending, labels)

    def tool(self, name, args, result, nested=False):
        if name == "write_todos" and isinstance(args, dict) and args.get("todos"):
            return self.todos(args["todos"])
        self._post(ToolCall(name, args, result, nested))

    def subagent(self, description):
        self._post(Note(Text("subagent · own context\n" + description.strip()), classes="subagent"))

    def todos(self, todos):
        self._call(self.app.query_one(TodoPanel).show, todos)

    def usage(self, stats):
        for key, value in stats.items():
            self.totals[key] = self.totals.get(key, 0) + (value or 0)
        self._call(self.app.refresh_status)

    def summary(self):
        if self.totals:
            self.note("\n".join(f"{k.replace('_', ' '):<18} {v:,}" for k, v in self.totals.items()))

    def compacted(self, before, messages):
        summary = next((text_of(m["content"]) for m in messages if "<summary>" in text_of(m.get("content"))), "")
        body = summary.replace("<summary>", "").replace("</summary>", "").strip()
        self._post(Collapsible(Static(body, markup=False, classes="thought-text"),
                               title=f"compacted · {before} → {len(messages)} messages", collapsed=True, **FOLD))

    def injection(self, text):
        self._post(Note(Text("late injection\n" + text.strip()), classes="debug"))

    def debug(self, data):
        self._post(Collapsible(Static(json.dumps(data, indent=2), markup=False),
                               title="raw reply", collapsed=True, classes="debug", **FOLD))

    # -- streamed replies

    def reply_start(self):
        widget = AssistantMessage(show_thinking=self.app.show_thinking)
        self._post(widget)
        with self.lock:
            self.current, self.pending, self.last_flush = widget, [], time.monotonic()

    def reply_event(self, kind, text):
        with self.lock:
            self.pending.append((kind, text))
            due = time.monotonic() - self.last_flush >= self.FLUSH_EVERY
        if due:
            self._flush()

    def _flush(self):
        with self.lock:
            batch, self.pending = self.pending, []
            self.last_flush = time.monotonic()
            widget = self.current
        if widget is None or not batch:
            return
        merged = []
        for kind, text in batch:  # fewer, bigger updates
            if merged and merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + text)
            else:
                merged.append((kind, text))

        def apply():
            for kind, text in merged:
                widget.add(kind, text)

        self._call(apply)

    def reply_end(self, reply):
        self._flush()
        widget, self.current = self.current, None
        if widget is not None:
            self._call(widget.finish, reply.content)

    def reply_abort(self):
        self._flush()
        widget, self.current = self.current, None
        if widget is not None:
            self._call(widget.finish)

    # -- busy

    @contextmanager
    def working(self, label="thinking"):
        self._call(self.app.push_busy, label)
        try:
            yield
        finally:
            self._call(self.app.pop_busy)

    # -- questions

    def approve(self, reason):
        return bool(self._ask(ApproveScreen(reason)))

    def pick(self, title, rows, search=True):
        return self._ask(PickScreen(title, rows, search))

    def secret(self, title, hint=""):
        return self._ask(SecretScreen(title, hint))

    def toggle_thinking(self):
        self.app.show_thinking = not self.app.show_thinking
        return self.app.show_thinking

    # -- screen

    def clear(self):
        self._call(self.app.clear_chat)

    def banner(self):
        self._post(Welcome())

    def resumed(self, messages, label="resumed"):
        turns = sum(1 for m in messages if m["role"] == "user")
        self.note(f"{label} · {len(messages)} messages · {turns} turns")

    def replay(self, messages):
        results = {m.get("tool_call_id"): m.get("content") for m in messages if m["role"] == "tool"}
        for message in messages:
            if message["role"] == "user" and "<summary>" not in text_of(message.get("content")):
                self.user(text_of(message["content"]))
            elif message["role"] == "assistant":
                if message.get("content"):
                    widget = AssistantMessage()
                    self._post(widget)
                    self._call(widget.finish, message["content"])
                for call in message.get("tool_calls") or []:
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                    except ValueError:
                        args = {}
                    self.tool(call["function"]["name"], args, results.get(call["id"], ""))


# =================================================== command palette


class ArgusCommands(Provider):
    """Our slash commands in the ctrl+p palette."""

    async def discover(self):
        for name, help_text in commands.COMMANDS.items():
            yield DiscoveryHit(name, partial(self.app.run_line, name), help=help_text)

    async def search(self, query):
        matcher = self.matcher(query)
        for name, help_text in commands.COMMANDS.items():
            score = matcher.match(f"{name} {help_text}")
            if score > 0:
                yield Hit(score, matcher.highlight(name), partial(self.app.run_line, name), help=help_text)


# ================================================================ the app


class ArgusApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "argus"
    COMMANDS = App.COMMANDS | {ArgusCommands}
    BINDINGS = [
        Binding("escape", "interrupt", "stop", show=False),
        Binding("ctrl+c", "clear_or_quit", "quit", show=False, priority=True),
        Binding("ctrl+o", "run('/model')", "model", show=False),
        Binding("ctrl+n", "run('/new')", "new chat", show=False),
        Binding("alt+v", "run('/paste')", "paste image", show=False),
    ]

    def __init__(self, conversation):
        super().__init__()
        self.conversation = conversation
        self.backend = TextualUI(self)
        self.show_thinking = False
        self.busy_labels = []
        self.spin = 0
        self.branch = ""
        self.files = None  # for @ completion, built on first use
        self.suggest_kind = None
        self.history = self.load_history()
        self.history_at = len(self.history)
        self.pending_labels = []

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        yield TodoPanel(id="todos")
        with Vertical(id="composer"):
            yield OptionList(id="suggest")
            yield Static("", id="pending")
            yield ChatInput(id="input", placeholder='Ask Argus anything…   "/" commands  ·  "@" attach  ·  enter send')
        yield StatusBar(id="status")

    def on_mount(self):
        self.theme = "tokyo-night"
        ui.install(self.backend)
        self.query_one("#chat").anchor()
        self.query_one(TodoPanel).display = False
        self.query_one("#pending").display = False
        self.branch = git("branch --show-current").strip()
        self.set_interval(0.1, self.tick)
        self.refresh_status()
        self.query_one(ChatInput).focus()

        if self.conversation.resumed:
            self.run_task(lambda: (self.backend.resumed(self.conversation.messages),
                                   self.backend.replay(self.conversation.messages)))
        else:
            self.post_widget(Welcome())
        if providers.first_run():
            self.run_task(lambda: commands.connect(first_run=True))

    # -- running things on the worker thread

    @property
    def busy(self):
        return any(w.is_running for w in self.workers if w.group == "agent")

    def run_line(self, text):
        if self.busy:
            self.notify("Argus is still working - press esc to stop it first", severity="warning")
            return
        self.run_task(lambda: self.conversation.submit(text))

    @work(thread=True, exclusive=True, group="agent")
    def run_task(self, task):
        try:
            if task() is False:
                self.call_from_thread(self.exit)
        except Exception as failure:  # noqa: BLE001 - never take the app down
            self.backend.error(f"{type(failure).__name__}: {failure}")
        finally:
            self.call_from_thread(self.refresh_status)

    # -- chat area

    def post_widget(self, widget):
        chat = self.query_one("#chat")
        for welcome in chat.query(Welcome):
            if not isinstance(widget, Welcome):
                welcome.remove()
        chat.mount(widget)

    def clear_chat(self):
        self.query_one("#chat").remove_children()

    def show_pending(self, labels):
        self.pending_labels += labels
        pending = self.query_one("#pending", Static)
        pending.update("📎 " + ", ".join(self.pending_labels) + "  - goes with your next message")
        pending.display = True

    # -- busy indicator and status bar

    def push_busy(self, label):
        self.busy_labels.append(label)
        self.refresh_status()

    def pop_busy(self):
        if self.busy_labels:
            self.busy_labels.pop()
        self.refresh_status()

    def tick(self):
        if self.busy_labels:
            self.spin = (self.spin + 1) % len(SPINNER)
            self.refresh_status()

    def refresh_status(self):
        model = providers.current()
        used = commands.LAST_PROMPT_TOKENS
        totals = self.backend.totals
        left = Text()
        if self.busy_labels:
            left.append(f"{SPINNER[self.spin]} {self.busy_labels[-1]}… ", style="bold")
            left.append("esc to stop", style="dim")
        else:
            cwd = Path.cwd()
            short = cwd.name if len(str(cwd)) > 40 else str(cwd).replace(str(Path.home()), "~")
            left.append(short, style="dim")
            if self.branch:
                left.append(f"  ⎇ {self.branch}", style="dim")
        right = Text()
        right.append(model.label, style="bold")
        right.append(f"   {used / 1000:.1f}K/{model.context // 1000}K ({used / max(model.context, 1):.0%})", style="dim")
        if totals:
            right.append(f"   ↑{totals.get('prompt_tokens', 0) / 1000:.1f}K ↓{totals.get('completion_tokens', 0) / 1000:.1f}K", style="dim")
        right.append("   ctrl+p commands", style="dim")
        width = max(self.size.width - 4, 20)
        if len(left) + len(right) + 2 > width:
            left.truncate(max(width - len(right) - 3, 0), overflow="ellipsis")
        gap = max(width - len(left) - len(right), 2)
        self.query_one(StatusBar).update(left + Text(" " * gap) + right)

    # -- input

    @on(ChatInput.Submitted)
    def submitted(self, event):
        text = event.text.strip()
        self.query_one(ChatInput).clear()
        self.hide_suggestions()
        self.remember(text)
        if self.pending_labels:
            self.pending_labels = []
            self.query_one("#pending").display = False
        self.run_line(text)

    @on(TextArea.Changed, "#input")
    def typed(self, event):
        self.update_suggestions()

    def action_run(self, line):
        self.run_line(line)

    def action_interrupt(self):
        if self.busy:
            CANCEL.set()
            self.notify("stopping…")

    def action_clear_or_quit(self):
        field = self.query_one(ChatInput)
        if field.text:
            field.clear()
            return
        CANCEL.set()
        self.exit()

    # -- message history (up / down)

    def load_history(self):
        try:
            return [json.loads(line) for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines()][-500:]
        except (OSError, ValueError):
            return []

    def remember(self, text):
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            try:
                HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
                with HISTORY_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(text) + "\n")
            except OSError:
                pass
        self.history_at = len(self.history)

    def recall(self, step):
        if not self.history:
            return
        self.history_at = min(max(self.history_at + step, 0), len(self.history))
        field = self.query_one(ChatInput)
        field.text = self.history[self.history_at] if self.history_at < len(self.history) else ""
        field.move_cursor(field.document.end)

    # -- the / and @ popup

    def suggestions_open(self):
        return self.query_one("#suggest").display and self.query_one("#suggest", OptionList).option_count > 0

    def hide_suggestions(self):
        self.query_one("#suggest").display = False
        self.suggest_kind = None

    def file_index(self):
        if self.files is None:
            found = []
            for root, dirs, names in os.walk("."):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                for name in names:
                    found.append(os.path.relpath(os.path.join(root, name)).replace("\\", "/"))
                    if len(found) >= 5000:
                        break
            self.files = sorted(found)
        return self.files

    def current_word(self):
        field = self.query_one(ChatInput)
        row, col = field.cursor_location
        line = field.document.get_line(row)[:col]
        return line.split(" ")[-1] if line else ""

    def update_suggestions(self):
        field = self.query_one(ChatInput)
        text = field.text
        options = []
        if text.startswith("/") and " " not in text and "\n" not in text:
            self.suggest_kind = "command"
            options = [Option(Text.assemble((f"{n:<11}", "bold"), (h, "dim")), id=n)
                       for n, h in commands.COMMANDS.items() if n.startswith(text)]
        else:
            word = self.current_word()
            if word.startswith("@"):
                self.suggest_kind = "file"
                query = word[1:].lower()
                matches = [f for f in self.file_index() if query in f.lower()]
                options = [Option(f, id=f) for f in matches[:10]]
        popup = self.query_one("#suggest", OptionList)
        popup.clear_options()
        if options:
            popup.add_options(options)
            popup.highlighted = 0
            popup.display = True
        else:
            self.hide_suggestions()

    def move_suggestion(self, step):
        popup = self.query_one("#suggest", OptionList)
        popup.action_cursor_up() if step < 0 else popup.action_cursor_down()

    def accept_suggestion(self, submit):
        popup = self.query_one("#suggest", OptionList)
        if popup.highlighted is None:
            return False
        choice = popup.get_option_at_index(popup.highlighted).id
        field = self.query_one(ChatInput)
        kind = self.suggest_kind
        self.hide_suggestions()
        if kind == "command":
            if submit:
                field.clear()
                self.remember(choice)
                self.run_line(choice)
            else:
                field.text = choice + " "
                field.move_cursor(field.document.end)
            return True
        word = self.current_word()
        field.delete((field.cursor_location[0], field.cursor_location[1] - len(word)), field.cursor_location)
        field.insert(f"@{choice} ")
        return True

    # -- leaving

    def print_summary(self):
        totals = self.backend.totals
        if not totals:
            return
        console = Console()
        parts = " · ".join(f"{v:,} {k.replace('_tokens', '')}" for k, v in totals.items() if v)
        console.print(f"[dim]argus · {providers.current().label} · {parts}[/dim]")
