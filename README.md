# Argus

A coding agent that lives in your terminal, in the spirit of OpenCode. It reads
and edits your files, runs commands, follows skills, and works with free cloud
models (OpenRouter) or models running on your own machine (Ollama, LM Studio).

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) (it brings the right Python with it).

```
uv tool install git+https://github.com/<your-username>/<this-repo>
```

Then open a terminal in any project folder and run:

```
argus
```

The first time, Argus asks where models should come from:

- **OpenRouter**: paste an API key from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
  Free models (names ending in `:free`) cost nothing; they are rate-limited.
- **Ollama**: install it from [ollama.com](https://ollama.com), then `ollama pull qwen3:4b`.
  Runs on your machine, private, slower without a GPU.
- **LM Studio**: load a model and switch the server on in the Developer tab.

Change your mind any time with `/connect` and `/model`.

## Using it

| | |
|---|---|
| `enter` / `ctrl+j` | send / new line |
| `/` | command menu (`/model`, `/connect`, `/new`, `/sessions`, `/rewind`, `/compact`, `/thinking`, `/usage`, …) |
| `@path` | attach a file or image (a list of matching files pops up) |
| drag an image in | attached automatically |
| `alt+v` or `/paste` | attach the screenshot on your clipboard |
| `esc` | stop the agent |
| `ctrl+o` | switch model |
| `ctrl+n` | new chat |
| `ctrl+p` | command palette (also themes) |
| `ctrl+c` | clear the input, or quit when it is empty |
| `↑` / `↓` | earlier messages |

The model's thinking is folded under a short heading; click it to read it, or
`/thinking` to always show it.

Options: `argus -m openrouter/google/gemma-4-31b-it:free`, `argus --resume`, `argus --debug`.

## Skills

A skill is a folder with a `SKILL.md`: a name, a one-line description of when
to use it, and the instructions. Argus sees every description and opens a
skill when a task matches.

```markdown
---
name: my-skill
description: When to use this skill.
---
Step-by-step instructions.
```

Argus looks in, last one winning: the skills it ships with (`argus/builtin_skills/`),
`~/.agents/skills/` (yours, everywhere), and `.agents/skills/` in the project.

## Where things are stored

| | |
|---|---|
| `~/.argus/auth.json` | API keys from `/connect` (plain text - keep your home folder private) |
| `~/.argus/state.json` | the model you picked |
| `~/.argus/history.jsonl` | what you typed, for ↑ |
| `~/.agents/sessions/` | saved chats |

Environment variables override the files: `OPENROUTER_API_KEY`, `OLLAMA_URL`,
`LMSTUDIO_URL`, `LOCAL_CONTEXT_WINDOW` (match Ollama's `OLLAMA_CONTEXT_LENGTH`),
`LOCAL_REASONING_EFFORT`, or `BASE_URL` + `API_KEY` for any other
OpenAI-compatible server.

## Safety

Read-only shell commands run straight away; anything else asks first, and a
few (`rm`, `curl`, `git push`, …) are always refused. On macOS and Linux shell
commands also run in an OS sandbox; on Windows they run through Git Bash with
no sandbox, so the approval prompt is the only guard.

## Developing

```
git clone <this repo> && cd <this repo>
uv tool install --editable .
```

Code changes apply the next time you start `argus`. Re-run the install with
`--force` after adding a dependency.

| File | What it does |
|---|---|
| `argus/agent.py` | the agent loop |
| `argus/llm.py` | streaming calls, splitting thinking from the answer |
| `argus/app.py`, `app.tcss` | the full-screen interface (Textual) |
| `argus/providers.py` | OpenRouter / Ollama / LM Studio, model lists |
| `argus/commands.py` | slash commands |
| `argus/tools.py`, `permissions.py`, `sandbox.py` | tools and what they may do |
| `argus/attachments.py` | files and images in messages |
| `argus/history.py`, `compact.py`, `session.py` | keeping long chats in the context window |
| `argus/subagent.py`, `skills.py`, `todos.py` | subagents, skills, the plan |
