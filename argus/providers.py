"""Who we talk to, and which model.

A provider is nothing more than an OpenAI-compatible base URL plus a key - the
same request works against Ollama on this laptop or OpenRouter's servers. A
model is always named "provider/model", e.g.

    ollama/qwen3:4b
    openrouter/google/gemma-4-31b-it:free

The current pick lives in ~/.argus/state.json, so the next start uses it.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from . import config

OPENROUTER_URL = "https://openrouter.ai/api/v1"
STATE_FILE = config.APP_DIR / "state.json"


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    local: bool = False


def all_providers():
    """Every provider we can talk to. Local ones are always listed; whether
    they are actually running is found out when we ask them for models."""
    found = {
        "ollama": Provider("ollama", config.OLLAMA_URL, "ollama", local=True),
        "lmstudio": Provider("lmstudio", config.LMSTUDIO_URL, "lmstudio", local=True),
    }
    openrouter_key = config.api_key("openrouter")
    if openrouter_key:
        found["openrouter"] = Provider("openrouter", OPENROUTER_URL, openrouter_key)
    if config.CUSTOM_BASE_URL:
        found["custom"] = Provider(
            "custom", config.CUSTOM_BASE_URL, config.CUSTOM_API_KEY or "none"
        )
    return found


# ------------------------------------------------------------- the current pick


@dataclass
class Model:
    provider: Provider
    name: str  # what the provider calls it, e.g. "google/gemma-4-31b-it:free"
    context: int

    @property
    def label(self):
        return f"{self.provider.name}/{self.name}"


CURRENT = None
_CLIENT = None


def split(spec):
    """'openrouter/google/gemma' -> ('openrouter', 'google/gemma')."""
    provider, _, name = spec.strip().partition("/")
    return provider, name


def select(spec, context=None, remember=True):
    """Switch to a model. Raises ValueError with a readable reason."""
    global CURRENT, _CLIENT
    provider_name, name = split(spec)
    providers = all_providers()
    if provider_name not in providers:
        known = ", ".join(providers)
        hint = " - add your key with /connect" if provider_name == "openrouter" else ""
        raise ValueError(f"unknown provider '{provider_name}'{hint}. Known: {known}")
    if not name:
        raise ValueError(f"say which model, e.g. {provider_name}/<model-name>")

    provider = providers[provider_name]
    if context is None:
        context = config.LOCAL_CONTEXT_WINDOW if provider.local else lookup_context(provider, name)

    CURRENT = Model(provider, name, context)
    _CLIENT = None
    config.CONTEXT_WINDOW = context
    if remember:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps({"model": CURRENT.label, "context": context}), encoding="utf-8")
    return CURRENT


def load():
    """Start-up: the remembered model, else DEFAULT_MODEL from .env."""
    try:
        saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return select(saved["model"], saved.get("context"), remember=False)
    except (OSError, ValueError, KeyError):
        return select(config.DEFAULT_MODEL, remember=False)


def current():
    return CURRENT or load()


def first_run():
    """Nothing chosen and no key saved yet: show the connect screen."""
    return not STATE_FILE.exists() and not config.api_key("openrouter")


def check_openrouter_key(key):
    """(ok, message). Asks OpenRouter whether the key is real."""
    request = urllib.request.Request(
        f"{OPENROUTER_URL}/key", headers={"Authorization": f"Bearer {key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.load(response).get("data", {})
    except urllib.error.HTTPError as error:
        return False, "OpenRouter rejected that key" if error.code in (401, 403) else f"OpenRouter said {error.code}"
    except OSError as error:
        return False, f"could not reach OpenRouter ({error})"
    label = data.get("label") or "key"
    return True, f"connected ({label})"


def client():
    global _CLIENT
    if _CLIENT is None:
        model = current()
        _CLIENT = OpenAI(base_url=model.provider.base_url, api_key=model.provider.api_key)
    return _CLIENT


def request_options():
    """Extra parameters for chat.completions.create for the current model."""
    model = current()
    options = {"model": model.name}
    if model.provider.local and config.LOCAL_REASONING_EFFORT:
        options["reasoning_effort"] = config.LOCAL_REASONING_EFFORT
    return options


# ------------------------------------------------------------- listing models


def list_models(provider, timeout=4):
    """[{id, context, vision, free}] from the provider, or [] if it is down."""
    try:
        api = OpenAI(base_url=provider.base_url, api_key=provider.api_key,
                     timeout=timeout, max_retries=0)
        data = api.models.list().data
    except Exception:  # noqa: BLE001 - a server that is off just has no models
        return []

    models = []
    for item in data:
        raw = item.model_dump()
        architecture = raw.get("architecture") or {}
        modalities = architecture.get("input_modalities") or []
        outputs = architecture.get("output_modalities") or []
        if outputs and "text" not in outputs:
            continue  # music / image generators cannot hold a conversation
        pricing = raw.get("pricing") or {}
        models.append({
            "id": item.id,
            "context": raw.get("context_length"),
            "vision": ("image" in modalities) if modalities else None,
            "free": item.id.endswith(":free") or (
                pricing.get("prompt") == "0" and pricing.get("completion") == "0"
            ),
        })
    return sorted(models, key=lambda m: m["id"])


def lookup_context(provider, name):
    for model in list_models(provider):
        if model["id"] == name and model["context"]:
            return int(model["context"])
    return config.DEFAULT_CONTEXT_WINDOW


def supports_images():
    """True / False, or None when the provider will not say."""
    model = current()
    if model.provider.name == "ollama":
        host = model.provider.base_url.rstrip("/").removesuffix("/v1")
        try:
            request = urllib.request.Request(
                f"{host}/api/show", data=json.dumps({"model": model.name}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=4) as response:
                shown = json.load(response)
            return "vision" in (shown.get("capabilities") or [])
        except Exception:  # noqa: BLE001
            return None
    for listed in list_models(model.provider):
        if listed["id"] == model.name:
            return listed["vision"]
    return None
