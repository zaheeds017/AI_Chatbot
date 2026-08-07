"""
app.py - Omni AI, a ChatGPT / Gemini / Blackbox-style chat UI (Streamlit).

Features:
  * Dark, ChatGPT-style interface: message bubbles + copy buttons on code blocks.
  * Left sidebar: persistent chat sessions + AI provider / model / API-key settings.
  * Streaming replies from Gemini / OpenAI / Claude when an API key is supplied.
  * Free offline mode (calculators + knowledge base + code templates) otherwise,
    so the app always works even without a key.
  * Chat history is saved to chat_history.json so it survives restarts.

Run:  streamlit run app.py
"""

import json
import os
import time

import streamlit as st

import ai_providers as ap
import chatbot_engine as engine

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(HERE, "chat_history.json")

st.set_page_config(page_title="Omni AI", page_icon="🤖", layout="wide")


# ---- Styling ----------------------------------------------------------------
APP_CSS = """
<style>
:root {
  --omni-bg: #0f1419;
  --omni-panel: #1a222b;
  --omni-bubble: #1f2c3a;
  --omni-accent: #10a37f;
  --omni-border: #2c3a45;
  --omni-text: #e6edf3;
  --omni-muted: #8b9aa8;
}
.stApp { background-color: var(--omni-bg); }
[data-testid="stSidebar"] {
  background-color: var(--omni-panel);
  border-right: 1px solid var(--omni-border);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 7rem; max-width: 920px; }
h1, h2, h3, h4 { color: var(--omni-text); }
p, li { color: var(--omni-text); }

/* ---- chat bubbles ---- */
[data-testid="stChatMessage"] { display: flex; margin-bottom: 1.1rem; }
[data-testid="stChatMessageContent"] { padding: 4px 0; }

/* user bubble -> right aligned, tinted */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  max-width: 80%;
  background: var(--omni-bubble);
  border: 1px solid var(--omni-border);
  border-radius: 18px 18px 4px 18px;
  padding: 10px 16px;
}

/* assistant message -> full width */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
  [data-testid="stChatMessageContent"] {
  max-width: 95%;
}

/* ---- chat input ---- */
div[data-testid="stChatInput"] {
  background: var(--omni-panel);
  border: 1px solid var(--omni-border);
  border-radius: 26px;
}
div[data-testid="stChatInput"] textarea {
  background: transparent !important;
  color: var(--omni-text) !important;
}
div[data-testid="stChatInput"]:focus-within {
  border-color: var(--omni-accent);
}

/* ---- code blocks + copy button ---- */
pre {
  position: relative;
  background: #0d1117 !important;
  border: 1px solid var(--omni-border);
  border-radius: 10px;
  padding: 12px 14px !important;
}
.omni-copy-btn {
  position: absolute;
  top: 8px;
  right: 10px;
  z-index: 5;
  background: rgba(255,255,255,0.07);
  color: #c8d2db;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 6px;
  font-size: 12px;
  padding: 2px 9px;
  cursor: pointer;
  font-family: inherit;
}
.omni-copy-btn:hover { background: rgba(255,255,255,0.16); color: #fff; }

/* sidebar buttons tidy */
[data-testid="stSidebar"] .stButton button {
  border-radius: 8px;
  justify-content: flex-start;
  text-align: left;
}
div[data-testid="stCaptionContainer"] { color: var(--omni-muted); }
</style>
"""

COPY_JS = """
<script>
function addOmniCopyButtons() {
  document.querySelectorAll('pre').forEach(function (pre) {
    if (pre.dataset.omniCopyReady) return;
    pre.dataset.omniCopyReady = '1';
    var btn = document.createElement('button');
    btn.className = 'omni-copy-btn';
    btn.textContent = 'Copy';
    btn.title = 'Copy code';
    btn.addEventListener('click', function (ev) {
      ev.stopPropagation();
      var code = (pre.querySelector('code') || pre).innerText;
      var done = function () {
        btn.textContent = 'Copied';
        setTimeout(function () { btn.textContent = 'Copy'; }, 1500);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code).then(done, done);
      } else {
        var ta = document.createElement('textarea');
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (e) {}
        document.body.removeChild(ta);
        done();
      }
    });
    pre.appendChild(btn);
  });
}
setInterval(addOmniCopyButtons, 400);
</script>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown(COPY_JS, unsafe_allow_html=True)


# ---- Knowledge base (cached across reruns) ----------------------------------
@st.cache_resource
def get_brain():
    kb = engine.load_kb()
    return kb, engine.Retriever(engine.flatten_faqs(kb))


kb, retriever = get_brain()


# ---- Session persistence -----------------------------------------------------
def _load_sessions():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass
    return {}


def _save_sessions():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.sessions, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _new_session(name=None):
    sid = "%s-%d" % (time.strftime("%H%M%S"), int(time.time() * 1000) % 100000)
    st.session_state.sessions[sid] = {
        "name": name or "New chat",
        "messages": [{"role": "assistant", "content": engine.greeting_text(kb)}],
        "created": time.time(),
    }
    st.session_state.active = sid
    _save_sessions()
    return sid


def _ensure_active():
    if not st.session_state.sessions:
        _new_session()
        return
    active = st.session_state.get("active")
    if active not in st.session_state.sessions:
        latest = max(st.session_state.sessions,
                     key=lambda k: st.session_state.sessions[k].get("created", 0))
        st.session_state.active = latest


if "sessions" not in st.session_state:
    st.session_state.sessions = _load_sessions()
    if not st.session_state.sessions:
        _new_session()
    else:
        _ensure_active()


# ---- Streaming helpers --------------------------------------------------------
def _provider_messages(messages):
    """Strip a leading assistant greeting so conversations start with a user turn."""
    msgs = [dict(m) for m in messages]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


def _offline_stream(text, delay=0.02):
    """Typewriter-style streaming for the offline engine's full reply."""
    lines = text.split("\n")
    buf = []
    for ln in lines:
        buf.append(ln)
        if len(buf) >= 2:
            yield "\n".join(buf)
            buf = []
            time.sleep(delay)
    if buf:
        yield "\n".join(buf)


def _safe_stream(gen):
    """Stream an online reply, converting errors into a visible message."""
    sent = False
    try:
        for chunk in gen:
            sent = True
            yield chunk
    except Exception as ex:
        yield "\n\n> ⚠️ **Error:** %s" % ex
        return
    if not sent:
        yield "_The model returned no text._"


# ---- Sidebar: settings + sessions ---------------------------------------------
with st.sidebar:
    st.markdown("## 🤖 Omni AI")
    st.caption("Study · Coding · Daily life")

    provider_labels = {
        "offline": "Offline (free, no key)",
        "gemini": "✨ Gemini (Google)",
        "gemini_oauth": "✨ Gemini (OAuth 2.0)",
        "openai": "🟢 OpenAI / ChatGPT",
        "claude": "🟠 Claude (Anthropic)",
    }
    provider = st.selectbox("AI engine", list(provider_labels),
                            format_func=lambda x: provider_labels[x])

    model = None
    api_key = ""
    openai_base_url = ""
    client_id = ""
    client_secret = ""
    if provider != "offline":
        cfg = ap.PROVIDERS[provider]
        default = cfg["default"]
        model = st.selectbox("Model", cfg["models"], key="model",
                             index=cfg["models"].index(default)
                             if default in cfg["models"] else 0)
        if provider == "gemini_oauth":
            client_id = st.text_input(
                "OAuth Client ID",
                placeholder="xxxx.apps.googleusercontent.com")
            client_secret = st.text_input("OAuth Client Secret",
                                          type="password")
            st.caption("First use opens a browser to sign in with Google; "
                       "the token is cached in gemini_oauth_token.json.")
        else:
            api_key = st.text_input("API key", type="password",
                                    placeholder="Paste key to go online…")
            if provider == "openai":
                openai_base_url = st.text_input(
                    "Custom OpenAI-compatible URL (optional)",
                    placeholder="https://api.openai.com/v1")
            st.caption("No key? The app falls back to free offline mode.")

    max_tokens = st.slider("Max reply length", 256, 4096, 1024, step=256)

    st.markdown("---")
    if st.button("➕ New chat", use_container_width=True):
        _new_session()
        st.rerun()

    st.markdown("#### 💬 Chat history")
    ids = sorted(st.session_state.sessions,
                 key=lambda k: st.session_state.sessions[k].get("created", 0),
                 reverse=True)
    for sid in ids:
        s = st.session_state.sessions[sid]
        is_active = (sid == st.session_state.get("active"))
        label = ("● " if is_active else "") + s["name"]
        cols = st.columns([5, 1])
        if cols[0].button(label, key="open_" + sid, use_container_width=True):
            st.session_state.active = sid
            st.rerun()
        if cols[1].button("🗑", key="del_" + sid, help="Delete this chat"):
            del st.session_state.sessions[sid]
            if st.session_state.get("active") == sid:
                st.session_state.active = None
            _save_sessions()
            _ensure_active()
            st.rerun()

    st.markdown("---")
    with st.expander("About"):
        st.markdown(
            "**Omni AI** answers across mathematics, science, biology, social "
            "studies, English, geometry, medical, arts, commerce, technology and "
            "daily life.\n\n"
            "- **Offline mode** uses built-in calculators, a knowledge base and "
            "code templates — free and instant.\n"
            "- **Online mode** streams replies from Gemini, ChatGPT or Claude "
            "when you add an API key.\n"
            "- Chat history is saved to `chat_history.json`.")


# ---- Main chat ----------------------------------------------------------------
_ensure_active()
sess = st.session_state.sessions[st.session_state.active]


def _handle_user_input(text):
    if not text or not text.strip():
        return
    text = text.strip()
    if sess["name"] == "New chat":
        sess["name"] = text[:32] + ("…" if len(text) > 32 else "")
    sess["messages"].append({"role": "user", "content": text})
    _save_sessions()

    with st.chat_message("user"):
        st.write(text)

    with st.chat_message("assistant"):
        if provider == "offline":
            reply = engine.get_response(text, _provider_messages(sess["messages"]),
                                        kb, retriever)
            full = st.write_stream(_offline_stream(reply))
        elif provider == "gemini_oauth":
            if client_id and client_secret:
                system = engine.build_system_prompt(kb)
                gen = ap.stream_response(
                    "gemini", model, _provider_messages(sess["messages"]),
                    system=system, max_tokens=max_tokens,
                    oauth={"client_id": client_id, "client_secret": client_secret})
                full = st.write_stream(_safe_stream(gen))
            else:
                st.caption("_Missing OAuth client ID / secret — using offline "
                           "mode. Fill them in the sidebar to go online._")
                reply = engine.get_response(text,
                                            _provider_messages(sess["messages"]),
                                            kb, retriever)
                full = st.write_stream(_offline_stream(reply))
        else:
            key = ap._get_key(provider, api_key)
            if key:
                system = engine.build_system_prompt(kb)
                gen = ap.stream_response(
                    provider, model, _provider_messages(sess["messages"]),
                    api_key=key, system=system, max_tokens=max_tokens,
                    openai_base_url=openai_base_url or None)
                full = st.write_stream(_safe_stream(gen))
            else:
                st.caption("_No API key found — using offline mode. Add a key in "
                           "the sidebar to go online._")
                reply = engine.get_response(text,
                                            _provider_messages(sess["messages"]),
                                            kb, retriever)
                full = st.write_stream(_offline_stream(reply))

    sess["messages"].append({"role": "assistant", "content": full})
    _save_sessions()


# header
st.markdown("## 🤖 Omni AI")
if provider == "offline":
    st.caption("**Mode:** offline — calculators, knowledge base & code templates (free).")
elif provider == "gemini_oauth":
    oauth_ready = bool(client_id and client_secret)
    st.caption("**Mode:** %s · **%s** · **%s**"
               % ("online" if oauth_ready else "offline fallback",
                  provider_labels[provider], model))
else:
    key_state = "online" if ap._get_key(provider, api_key) else "offline fallback"
    st.caption("**Mode:** %s · **%s** · **%s**"
               % (key_state, provider_labels[provider], model))

# examples when the chat is still empty
if len(sess["messages"]) == 1:
    st.markdown("##### Try asking")
    examples = [
        "Write a python program to check prime",
        "Force for mass 10 kg and acceleration 5",
        "What is photosynthesis?",
        "Convert 10 km to miles",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, key="ex_" + ex):
            _handle_user_input(ex)

# conversation
for msg in sess["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_text = st.chat_input("Message Omni AI… (e.g. 'solve 2^10', 'explain this code')")
if user_text:
    _handle_user_input(user_text)
