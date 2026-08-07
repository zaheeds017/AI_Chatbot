"""
ai_providers.py - streaming connectors that turn the chatbot into a
ChatGPT / Gemini / Blackbox-style AI assistant.

Supported providers (pick one in the app's sidebar):
  * "gemini"  - Google Gemini models        (REST streaming endpoint)
  * "openai"  - OpenAI models and any OpenAI-compatible API. Works with
                api.openai.com and also with Gemini's OpenAI-compatible
                endpoint (https://generativelanguage.googleapis.com/v1beta/openai/)
  * "claude"  - Anthropic Claude models

Each provider streams the assistant's answer as text chunks arrive, so the UI
can show a typewriter effect. Only the standard library + `requests` are needed
(requests ships with Streamlit anyway).

API keys can be passed directly or read from environment variables:
  GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
"""

import json
import os

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:  # optional - only needed for the "Gemini (OAuth 2.0)" provider
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    OAUTH_AVAILABLE = True
except ImportError:  # pragma: no cover
    OAUTH_AVAILABLE = False

# ---- Provider catalogue ---------------------------------------------------
PROVIDERS = {
    "gemini": {
        "label": "Gemini (Google)",
        "icon": "✨",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "env_key": "GEMINI_API_KEY",
        "default": "gemini-2.5-flash",
    },
    "gemini_oauth": {
        "label": "Gemini (OAuth 2.0)",
        "icon": "✨",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "env_key": "",
        "default": "gemini-2.5-flash",
    },
    "openai": {
        "label": "OpenAI / ChatGPT",
        "icon": "🟢",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3-mini"],
        "env_key": "OPENAI_API_KEY",
        "default": "gpt-4o-mini",
    },
    "claude": {
        "label": "Claude (Anthropic)",
        "icon": "🟠",
        "models": ["claude-opus-4", "claude-sonnet-4", "claude-haiku-4"],
        "env_key": "ANTHROPIC_API_KEY",
        "default": "claude-sonnet-4",
    },
}

DEFAULT_MAX_TOKENS = 1024


def _get_key(provider, api_key):
    """Prefer the passed-in key, then the provider's environment variable."""
    if api_key and api_key.strip():
        return api_key.strip()
    env = PROVIDERS[provider]["env_key"]
    return os.environ.get(env, "").strip()


def _sse_json_lines(resp):
    """Parse JSON payloads out of an SSE ('data: ...') streaming response."""
    for raw in resp.iter_lines(decode_unicode=True):
        line = (raw or "").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        if not payload:
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


def _check_response(resp):
    if not resp.ok:
        try:
            err = resp.json()
            msg = (err.get("error", {}).get("message") or
                   err.get("message") or
                   str(err))
        except Exception:
            msg = resp.text[:300]
        raise RuntimeError("API error (%s): %s" % (resp.status_code, msg))


# ---- Gemini OAuth 2.0 ------------------------------------------------------
GEMINI_OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
    "https://www.googleapis.com/auth/generative-language.retriever",
]


def _oauth_client_config(client_id, client_secret):
    """Build an 'installed app' client config from a raw client ID + secret."""
    return {
        "installed": {
            "client_id": client_id.strip(),
            "client_secret": client_secret.strip(),
            "auth_uri": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def _save_oauth_token(creds, token_path):
    try:
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    except OSError:
        pass


def _gemini_oauth_token(client_id, client_secret, token_path=None):
    """Return a valid Google access token for the Gemini API via OAuth 2.0.

    Uses the cached token in `token_path` (refreshing it when expired) or runs
    the installed-app browser flow on first use. Requires the optional packages
    `google-auth` and `google-auth-oauthlib`.
    """
    if not OAUTH_AVAILABLE:
        raise RuntimeError(
            "OAuth 2.0 support needs extra packages. Install with:\n"
            "    pip install google-auth google-auth-oauthlib")
    token_path = token_path or os.path.join(os.path.dirname(__file__),
                                            "gemini_oauth_token.json")

    creds = None
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path,
                                                          GEMINI_OAUTH_SCOPES)
        except Exception:
            creds = None
    if creds and creds.valid:
        return creds.token
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            _save_oauth_token(creds, token_path)
            return creds.token
        except Exception:
            creds = None

    flow = InstalledAppFlow.from_client_config(
        _oauth_client_config(client_id, client_secret), GEMINI_OAUTH_SCOPES)
    creds = flow.run_local_server(port=0, access_type="offline",
                                  prompt="consent")
    _save_oauth_token(creds, token_path)
    return creds.token


# ---- Gemini ---------------------------------------------------------------
def _gemini_stream(model, messages, api_key, system, max_tokens, bearer=None):
    if requests is None:
        raise RuntimeError("The 'requests' package is required for AI providers.")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/%s"
           ":streamGenerateContent" % model)
    contents = [{
        "role": "model" if m["role"] == "assistant" else "user",
        "parts": [{"text": m["content"]}],
    } for m in messages]
    body = {"contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens}}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    params, headers = {"alt": "sse"}, {}
    if bearer:
        headers["Authorization"] = "Bearer " + bearer
    else:
        params["key"] = api_key
    resp = requests.post(url, params=params, headers=headers,
                         json=body, stream=True, timeout=90)
    _check_response(resp)
    for data in _sse_json_lines(resp):
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            continue
        for part in parts:
            text = part.get("text")
            if text:
                yield text


# ---- OpenAI / OpenAI-compatible --------------------------------------------
def _openai_stream(model, messages, api_key, system, max_tokens, base_url):
    if requests is None:
        raise RuntimeError("The 'requests' package is required for AI providers.")
    chat_messages = []
    if system:
        chat_messages.append({"role": "system", "content": system})
    chat_messages += [{
        "role": m["role"] if m["role"] in ("user", "assistant") else "user",
        "content": m["content"],
    } for m in messages]
    payload = {"model": model, "stream": True, "max_tokens": max_tokens,
               "messages": chat_messages}
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + api_key},
        json=payload, stream=True, timeout=90)
    _check_response(resp)
    for data in _sse_json_lines(resp):
        try:
            delta = data["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError):
            continue
        if delta:
            yield delta


# ---- Claude ----------------------------------------------------------------
def _claude_stream(model, messages, api_key, system, max_tokens):
    if requests is None:
        raise RuntimeError("The 'requests' package is required for AI providers.")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": [{
            "role": m["role"] if m["role"] in ("user", "assistant") else "user",
            "content": m["content"],
        } for m in messages],
    }
    if system:
        payload["system"] = system
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json=payload, stream=True, timeout=90)
    _check_response(resp)
    for data in _sse_json_lines(resp):
        if data.get("type") == "content_block_delta":
            delta = data.get("delta", {}).get("text")
            if delta:
                yield delta


# ---- Public entry point ----------------------------------------------------
def stream_response(provider, model, messages, api_key=None, system=None,
                    max_tokens=DEFAULT_MAX_TOKENS, openai_base_url=None,
                    oauth=None):
    """Stream an assistant reply, yielding text chunks.

    provider         : "gemini" | "openai" | "claude" (or "gemini_oauth" for
                       Gemini authenticated with OAuth 2.0)
    model            : a model from PROVIDERS[provider]["models"] (or any id)
    messages         : [{"role": "user"|"assistant", "content": str}, ...]
    api_key          : optional - falls back to the provider's env variable
    system           : optional system prompt (e.g. facts from the knowledge base)
    max_tokens       : cap on the reply length
    openai_base_url  : custom OpenAI-compatible endpoint (e.g. Gemini's openai API)
    oauth            : optional dict {"client_id":..., "client_secret":...,
                       "token_path":...} to authenticate to Gemini with OAuth 2.0
                       (client ID + secret) instead of an API key.
    """
    if provider == "gemini_oauth" or (provider == "gemini" and oauth):
        if oauth:
            token = _gemini_oauth_token(
                oauth["client_id"], oauth["client_secret"],
                oauth.get("token_path"))
            yield from _gemini_stream(model, messages, None, system,
                                      max_tokens, bearer=token)
        else:
            raise ValueError(
                "Gemini OAuth needs client credentials. Pass "
                "oauth={'client_id': ..., 'client_secret': ...}.")
        return

    key = _get_key(provider, api_key)
    if not key:
        raise ValueError(
            "No API key for %s. Add it in the app's sidebar or set the "
            "%s environment variable. Without a key the chatbot uses its "
            "free offline mode." % (provider, PROVIDERS[provider]["env_key"]))

    if provider == "gemini":
        yield from _gemini_stream(model, messages, key, system, max_tokens)
    elif provider == "openai":
        yield from _openai_stream(
            model, messages, key, system, max_tokens,
            base_url=openai_base_url or "https://api.openai.com/v1")
    elif provider == "claude":
        yield from _claude_stream(model, messages, key, system, max_tokens)
    else:
        raise ValueError("Unknown provider: %r" % provider)


if __name__ == "__main__":
    print("ai_providers.py - providers:", ", ".join(PROVIDERS))
    for name, cfg in PROVIDERS.items():
        if name == "gemini_oauth":
            print("  %-8s %-22s oauth:%s  default model: %s"
                  % (name, cfg["label"],
                     "OK" if OAUTH_AVAILABLE else "missing packages",
                     cfg["default"]))
            continue
        env_ok = "OK" if os.environ.get(cfg["env_key"]) else "missing"
        print("  %-8s %-22s env:%s  default model: %s"
              % (name, cfg["label"], env_ok, cfg["default"]))
    print("\nTest connectivity by calling stream_response() from your app.")
