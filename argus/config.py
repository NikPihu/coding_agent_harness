"""Settings.

Where things live (the same split OpenCode uses):
  ~/.argus/auth.json   API keys you entered with /connect
  ~/.argus/state.json  the model you picked with /model

Environment variables still win over both, and a .env file is read from
~/.agents/env or, for development, next to the source checkout.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path.home() / ".argus"
AUTH_FILE = APP_DIR / "auth.json"

HARNESS_ENV = Path(__file__).resolve().parent.parent / ".env"
USER_ENV = Path.home() / ".agents" / "env"

for env_file in (HARNESS_ENV, USER_ENV):
    if env_file.exists():
        load_dotenv(env_file, override=False)  # never overrides what is already set


def _get(name, default=None):
    value = os.environ.get(name, "").strip()
    return value or default


# --- providers (see providers.py) ---------------------------------------
OLLAMA_URL = _get("OLLAMA_URL", "http://localhost:11434/v1")
LMSTUDIO_URL = _get("LMSTUDIO_URL", "http://localhost:1234/v1")

def _auth():
    try:
        return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def api_key(provider):
    """The key for a provider: environment first, then ~/.argus/auth.json."""
    return _get(f"{provider.upper()}_API_KEY") or _auth().get(provider)


def save_api_key(provider, key):
    auth = _auth()
    if key:
        auth[provider] = key
    else:
        auth.pop(provider, None)
    APP_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_FILE.write_text(json.dumps(auth, indent=2), encoding="utf-8")
    os.environ.pop(f"{provider.upper()}_API_KEY", None)  # the saved one wins now


CUSTOM_BASE_URL = _get("BASE_URL")  # any other OpenAI-compatible endpoint
CUSTOM_API_KEY = _get("API_KEY")

# Used until you pick something with /model (the pick is remembered).
DEFAULT_MODEL = _get("DEFAULT_MODEL", "ollama/qwen3:4b")

# Local servers do not report how much context they actually allocate, so say
# it here. For Ollama this must match OLLAMA_CONTEXT_LENGTH.
LOCAL_CONTEXT_WINDOW = int(_get("LOCAL_CONTEXT_WINDOW", 32_768))
DEFAULT_CONTEXT_WINDOW = int(_get("CONTEXT_WINDOW", 128_000))

# "none", "low", "medium" or "high", sent to local models only. Reasoning
# models like qwen3 think for minutes on a CPU; "none" turns that off.
LOCAL_REASONING_EFFORT = _get("LOCAL_REASONING_EFFORT")

# How much room the current model has, and how we spend it. providers.select()
# overwrites CONTEXT_WINDOW whenever the model changes.
CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW
COMPACT_AT = 0.85  # compact once the prompt crosses this much of the window
COMPACT_TO = 0.35  # and cut back to this much, so it does not retrigger soon
