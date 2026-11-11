"""Slash commands. Anything typed starting with / lands here.

handle() returns the (possibly new) message list, or None to quit. These run on
the agent's worker thread, so ui.pick() / ui.secret() can simply wait for an
answer from the screen.
"""

from . import attachments
from . import compact as compaction
from . import config
from . import providers
from . import session
from .skills import SKILLS
from .todos import TODOS
from .ui import ui

COMMANDS = {
    "/connect": "add an OpenRouter key, or use local Ollama / LM Studio models",
    "/model": "switch model  (/model gemma searches)",
    "/new": "start a fresh chat",
    "/sessions": "open a past chat",
    "/rewind": "jump back to an earlier point in this chat",
    "/compact": "summarise the chat so far to free up context",
    "/paste": "attach the image on the clipboard (also alt+v)",
    "/thinking": "show or hide the model's thinking",
    "/skills": "list the skills Argus can use",
    "/usage": "tokens used and how full the context is",
    "/clear": "clear the screen",
    "/help": "list commands and keys",
    "/exit": "quit",
}

LAST_PROMPT_TOKENS = 0  # set by agent.py after every call, for /usage


# ------------------------------------------------------------------ models


def available_models(search="", only=None, free_only=None):
    """[(label, row text)] across every provider that answers.

    OpenRouter has hundreds of models, so unless you search, only its free
    ones are listed. `only` limits it to one provider.
    """
    if free_only is None:
        free_only = not search
    found = []
    for provider in providers.all_providers().values():
        if only and provider.name != only:
            continue
        for model in providers.list_models(provider):
            if provider.name == "openrouter" and free_only and not model["free"]:
                continue
            label = f"{provider.name}/{model['id']}"
            if search.lower() not in label.lower():
                continue
            details = []
            if model["context"]:
                details.append(f"{int(model['context']) // 1000}K")
            if model["vision"]:
                details.append("vision")
            if provider.name == "openrouter":
                details.append("free" if model["free"] else "paid")
            elif provider.local:
                details.append("local")
            found.append((label, f"{label}   {' · '.join(details)}"))
    return found


def switch(spec):
    try:
        model = providers.select(spec)
    except ValueError as problem:
        ui.error(str(problem))
        return False
    ui.note(f"model: {model.label}  ·  context {model.context:,} tokens")
    return True


def model(argument="", only=None):
    # A full "provider/model" name switches straight away, even to a model
    # that is not in the list (any OpenRouter model id works).
    provider_name, name = providers.split(argument)
    if name and provider_name in providers.all_providers():
        return switch(argument)

    with ui.working("asking providers for their models"):
        found = available_models(argument, only=only)
    if not found:
        ui.error(
            "no models found. Start Ollama or LM Studio, or add an OpenRouter key with /connect."
            + (f" Nothing matched '{argument}'." if argument else "")
        )
        return False
    choice = ui.pick(f"Select model  (now: {providers.current().label})", [row for _, row in found])
    if choice is None:
        return False
    return switch(found[choice][0])


# ----------------------------------------------------------------- connect


def connect(first_run=False):
    """Pick where models come from. On first run this is the welcome screen."""
    names = providers.all_providers()
    with ui.working("looking for local model servers"):
        local = {
            name: providers.list_models(names[name], timeout=2) for name in ("ollama", "lmstudio")
        }

    def local_row(name, title):
        count = len(local[name])
        state = f"running · {count} model{'s' * (count != 1)}" if count else "not running"
        return f"{title}   local, free, private  ·  {state}"

    openrouter_state = "connected" if config.api_key("openrouter") else "needs an API key"
    rows = [
        f"OpenRouter   cloud, free and paid models  ·  {openrouter_state}",
        local_row("ollama", "Ollama"),
        local_row("lmstudio", "LM Studio"),
    ]
    title = "Welcome to Argus - where should models come from?" if first_run else "Connect a provider"
    choice = ui.pick(title, rows, search=False)
    if choice is None:
        return

    if choice == 0:
        key = ui.secret(
            "OpenRouter API key",
            "Create one at openrouter.ai/settings/keys - it is saved to ~/.argus/auth.json",
        )
        if not key:
            return
        with ui.working("checking the key with OpenRouter"):
            ok, message = providers.check_openrouter_key(key.strip())
        if not ok:
            ui.error(message)
            return
        config.save_api_key("openrouter", key.strip())
        ui.note(f"OpenRouter {message}. Pick a model:")
        model(only="openrouter")
        return

    name = "ollama" if choice == 1 else "lmstudio"
    if not local[name]:
        how = {
            "ollama": "Install it from ollama.com, then run:  ollama pull qwen3:4b",
            "lmstudio": "Open LM Studio, load a model, and switch the server on in the Developer tab.",
        }
        ui.error(f"{'Ollama' if name == 'ollama' else 'LM Studio'} is not running. {how[name]}")
        return
    model(only=name)


# ----------------------------------------------------------------- history


def preview(message):
    if message.get("tool_calls"):
        return "-> " + message["tool_calls"][0]["function"]["name"]
    from .content import text_of
    return " ".join(text_of(message.get("content")).split())[:70]


def redraw(messages, label):
    """The screen no longer matches the history, so wipe it and draw again."""
    ui.clear()
    ui.resumed(messages, label)
    ui.replay(messages)
    return messages


def rewind(messages):
    rows = [f"{m['role']:<9} {preview(m)}" for m in messages]
    choice = ui.pick("Rewind to", rows)
    if choice is None:
        return messages
    session.rewind_to(choice + 1)
    return redraw(messages[: choice + 1], "rewound")


def sessions(messages):
    saved = session.all_sessions()
    if not saved:
        ui.note("no saved chats yet")
        return messages
    rows = [f"{s['id']}  {s['title']}" for s in saved]
    choice = ui.pick("Open a past chat", rows)
    if choice is None:
        return messages

    return redraw(session.open_session(saved[choice]["id"]), "opened")


def new(messages):
    session.new()
    TODOS.clear()
    ui.todos([])
    ui.clear()
    ui.banner()
    return messages[:1]  # keep only the system prompt


def compact(messages):
    before = len(messages)
    try:
        with ui.working("compacting"):
            compacted = compaction.compact(messages)
    except Exception as failure:  # noqa: BLE001
        # Compaction is one more API call, and it fires when the window is
        # nearly full - the worst moment to lose the session over a rate limit.
        ui.note(f"compaction failed ({type(failure).__name__}); transcript kept as is")
        return messages
    if len(compacted) == before:
        ui.note("nothing old enough to compact yet")
        return messages
    session.compacted(compacted)
    ui.compacted(before, compacted)
    return compacted


# ------------------------------------------------------------------- other


def paste():
    try:
        label = attachments.paste()
    except attachments.AttachmentError as problem:
        ui.error(str(problem))
        return
    except Exception as failure:  # noqa: BLE001 - clipboard access is OS-specific
        ui.error(f"could not read the clipboard ({type(failure).__name__}: {failure})")
        return
    ui.attached([label])


def usage():
    window = config.CONTEXT_WINDOW
    used = LAST_PROMPT_TOKENS
    ui.note(
        f"model: {providers.current().label}\n"
        f"context: {used:,} / {window:,} tokens ({used / window:.0%}), "
        f"auto-compacts at {config.COMPACT_AT:.0%}"
    )
    ui.summary()


KEYS = [
    ("enter", "send"),
    ("ctrl+j", "new line"),
    ("esc", "stop the agent"),
    ("ctrl+p", "command palette"),
    ("ctrl+o", "switch model"),
    ("ctrl+n", "new chat"),
    ("alt+v", "attach clipboard image"),
    ("@path", "attach a file or image"),
    ("ctrl+c", "clear input / quit"),
]


def help_text():
    width = max(len(name) for name in COMMANDS)
    lines = [f"{name:<{width}}  {text}" for name, text in COMMANDS.items()]
    lines += [""] + [f"{key:<{width}}  {text}" for key, text in KEYS]
    return "\n".join(lines)


def handle(command, messages):
    name, _, argument = command.strip().partition(" ")
    argument = argument.strip()

    if name in ("/exit", "/quit"):
        return None
    if name == "/model" or name == "/models":
        model(argument)
    elif name == "/connect":
        connect()
    elif name == "/new":
        return new(messages)
    elif name == "/compact":
        return compact(messages)
    elif name == "/rewind":
        return rewind(messages)
    elif name == "/sessions":
        return sessions(messages)
    elif name == "/paste":
        paste()
    elif name == "/thinking":
        shown = ui.toggle_thinking()
        ui.note("thinking is now " + ("shown expanded" if shown else "folded away"))
    elif name == "/skills":
        ui.note("\n".join(f"{n}: {s['description']}" for n, s in SKILLS.items()) or "no skills found")
    elif name == "/usage":
        usage()
    elif name == "/clear":
        ui.clear()
    else:
        if name != "/help":
            ui.error(f"unknown command {name}")
        ui.note(help_text())
    return messages
