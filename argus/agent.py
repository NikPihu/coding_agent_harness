"""The agent loop, with no opinion about how anything looks.

Conversation.submit() takes one line typed by the user - a message or a /command
- and runs it to completion: call the model, run the tools it asks for, feed
the results back, repeat until it answers. Everything it shows goes through
`ui`, which the app (app.py) plugs in.
"""

import argparse

import openai

from . import attachments
from . import commands
from . import compact
from . import history
from . import providers
from . import session
from .content import image_count
from .context import reminder
from .llm import CANCEL, SYSTEM_PROMPT, Interrupted, call_llm
from .todos import active_form
from .tools import execute
from .ui import ui


def describe(error):
    """One readable line for an API failure, instead of a traceback."""
    if isinstance(error, openai.APIConnectionError):
        model = providers.current()
        hint = " Is it running?" if model.provider.local else ""
        return f"could not reach {model.provider.name} at {model.provider.base_url}.{hint}"
    if isinstance(error, openai.APIStatusError):
        hints = {
            401: "the API key was rejected - run /connect",
            402: "out of credits for this model",
            404: "the provider does not know this model - try /model",
            429: "rate limited - wait a bit, or /model to switch",
        }
        detail = getattr(error, "message", "") or str(error)
        return f"{error.status_code}: {hints.get(error.status_code, detail)}  ({detail[:200]})"
    return f"{type(error).__name__}: {error}"


def answer_dangling_calls(messages):
    """Every tool call must be followed by its result, or the next request is
    rejected. After an interrupt, fill in the ones that never ran."""
    last_call = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("tool_calls")), None
    )
    if last_call is None:
        return
    answered = {m.get("tool_call_id") for m in messages[last_call + 1:]}
    for call in messages[last_call]["tool_calls"]:
        if call["id"] not in answered:
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": "Interrupted by the user before this ran.",
            })


def run_turn(messages, debug=False):
    """The agent loop for one user message: call, run tools, repeat."""
    while True:
        if CANCEL.is_set():
            raise Interrupted()
        injection = reminder()
        if debug:
            ui.injection(injection["content"])

        if history.fit(messages):
            ui.note("dropped old tool output to make this request fit")

        ui.reply_start()
        with ui.working(active_form()):
            reply, usage = call_llm(messages + [injection], on_event=ui.reply_event)
        ui.reply_end(reply)

        messages.append(reply.to_message())
        session.save(messages)
        ui.usage(usage)
        commands.LAST_PROMPT_TOKENS = usage["prompt_tokens"]

        if debug:
            ui.debug(reply.to_message())

        if not reply.tool_calls:
            if not reply.content:
                # Reasoning models sometimes spend the whole reply thinking.
                ui.note("the model returned no answer - ask again, or try another /model")
            return usage

        for tool_call in reply.tool_calls:
            if CANCEL.is_set():
                raise Interrupted()
            with ui.working(f"running {tool_call.function.name}"):
                args, result = execute(tool_call)
            ui.tool(tool_call.function.name, args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
            session.save(messages)


class Conversation:
    def __init__(self, resume=False, debug=False):
        self.debug = debug
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.resumed = False
        if resume:
            saved = session.all_sessions()
            if saved:
                self.messages = session.open_session(saved[0]["id"])
                history.strip(self.messages)
                self.resumed = True

    def submit(self, text):
        """Handle one line from the user. Returns False when it is time to quit."""
        text = text.strip()
        if not text:
            return True
        CANCEL.clear()

        if text.startswith("/"):
            result = commands.handle(text, self.messages)
            if result is None:
                return False
            self.messages = result
            session.save(self.messages)
            return True

        try:
            content, attached = attachments.build(text)
        except (attachments.AttachmentError, OSError) as problem:
            ui.error(str(problem))
            return True
        ui.user(text, attached)
        if image_count(content) and providers.supports_images() is False:
            ui.error(
                f"{providers.current().label} cannot see images. "
                "Pick a vision model with /model, then send again."
            )
            return True

        turn_start = len(self.messages)
        self.messages.append({"role": "user", "content": content})

        try:
            usage = run_turn(self.messages, self.debug)
        except Interrupted:
            answer_dangling_calls(self.messages)
            session.save(self.messages)
            ui.reply_abort()
            ui.note("stopped. Send a message to carry on, or give new instructions.")
            return True
        except openai.APIError as error:
            ui.reply_abort()
            ui.error(describe(error))
            if len(self.messages) == turn_start + 1:
                self.messages.pop()  # nothing happened yet: forget the message
                ui.note("your message was not sent - press up to get it back")
            else:
                answer_dangling_calls(self.messages)
                session.save(self.messages)
                ui.note("the work so far is kept - send a message to carry on")
            return True
        finally:
            history.sweep()  # the turn is over: bin its temp files

        history.strip(self.messages)  # ...and shrink the tool output it produced

        if compact.needed(usage):
            self.messages = commands.compact(self.messages)
        return True


def main():
    parser = argparse.ArgumentParser(prog="argus", description="Argus, a minimal coding agent.")
    parser.add_argument("--resume", action="store_true", help="continue the last session")
    parser.add_argument("--debug", action="store_true", help="show raw model replies and injections")
    parser.add_argument("-m", "--model", help="model to use, e.g. ollama/qwen3:4b")
    cli = parser.parse_args()

    if cli.model:
        try:
            providers.select(cli.model)
        except ValueError as problem:
            print(f"argus: {problem}")
            return

    from .app import ArgusApp  # imported late: Textual is slow to load

    app = ArgusApp(Conversation(resume=cli.resume, debug=cli.debug))
    app.run()
    app.print_summary()


if __name__ == "__main__":
    main()
