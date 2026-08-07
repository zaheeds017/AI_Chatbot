# Omni AI — ChatGPT / Gemini / Blackbox-style Chatbot 🤖

A capstone chatbot that works like **ChatGPT, Gemini or Blackbox AI** — a dark
ChatGPT-style chat UI that helps with **all school subjects**, **coding**, and
**everyday applications**, powered by real AI streaming when you add an API key.

- **Mathematics** — safe arithmetic solver that understands words AND symbols ("what is 5 times 8 plus 3 squared?"), percentages, averages, GCD/LCM, prime checks.
- **Science** — physics formula calculators: velocity, acceleration, force, work, power, kinetic/potential energy, density, pressure, momentum, Ohm's law, wave speed.
- **All subjects** — biology, social studies, English, geometry, medical & health, arts, commerce, technology & daily life, answered with a **TF-IDF retriever**.
- **Coding (Blackbox-style)** — ready-made code templates in Python, HTML, CSS, JavaScript and SQL, with **line-by-line explanations** and **copy buttons** in the UI.
- **Geometry** — live area / perimeter / volume calculators for circles, squares, rectangles, triangles, trapezoids, spheres, cylinders, cones, cubes and cuboids, plus Pythagoras.
- **Commerce** — simple & compound interest, discount, GST, profit/loss.
- **Medical** — BMI calculator (metric + imperial units).
- **Daily life** — unit converter (km↔miles, kg↔lb, °C↔°F, litres↔gallons, time, speed...), days-between-dates, mean/average of a list.

---

## Two ways to answer

| Mode | How | Cost |
| --- | --- | --- |
| **Offline** (default) | Built-in calculators + knowledge base + code templates via a TF-IDF retriever | Free, instant, no key |
| **Online** | **Streams** replies from **Gemini / ChatGPT (OpenAI) / Claude** when you paste an API key in the sidebar | Requires your own key |

No API key? The app automatically falls back to offline mode — it never stops working.

---

## Files

```
app.py               # ChatGPT-style dark UI: sessions, settings, streaming, copy buttons
chatbot_engine.py    # the brain: math solver + formula engine + code templates + retriever
ai_providers.py      # streaming connectors for Gemini, OpenAI, Claude (uses requests)
knowledge_base.json  # facts + formulas + code templates  <-- EDIT to add more
chat_history.json    # your chat sessions (created automatically)
.streamlit/config.toml  # dark theme
```

The logic is in `chatbot_engine.py` (no Streamlit), so you can test it directly:
```bash
python chatbot_engine.py
```

---

## ▶️ Run it

```bash
pip install streamlit          # once
streamlit run app.py           # opens http://localhost:8501
```

Try asking things like:

| Category | Example |
| --- | --- |
| Math | `what is 15% of 200?` · `2 to the power of 10` · `average of 5, 7, 9` |
| Physics | `force for mass 10 kg and acceleration 5` · `kinetic energy for mass 2 kg and velocity 5` |
| Geometry | `area of a circle with radius 7` · `hypotenuse of 3 and 4` · `volume of a sphere with radius 5` |
| Commerce | `simple interest on 1000 at 5% for 2 years` · `20% discount on 500` · `gst on 1000 at 18%` |
| Medical | `BMI for 70 kg and 175 cm` |
| Daily life | `convert 10 km to miles` · `25 celsius in fahrenheit` · `how many days between 2024-01-01 and 2024-03-01` |
| Coding | `write a python program to check prime` · `html form` · `css center a div` · `sql query for students` |
| Subjects | `what is photosynthesis?` · `what is a noun?` · `explain supply and demand` |

### Going online (optional)

1. Pick **Gemini / ChatGPT / Claude** in the left sidebar.
2. Paste your API key. Keys are never saved to disk; they are read from the environment variable if not pasted:
   - `GEMINI_API_KEY` for Gemini
   - `OPENAI_API_KEY` for OpenAI (or set a custom OpenAI-compatible URL)
   - `ANTHROPIC_API_KEY` for Claude
3. Replies stream in live with a typewriter effect. The whole knowledge base is sent as grounding, so the AI stays on-topic for your studies.

> OpenAI-compatible note: you can point the **OpenAI** option at Gemini's OpenAI-compatible endpoint
> (`https://generativelanguage.googleapis.com/v1beta/openai/`) or any OpenAI-compatible server.

#### Gemini with OAuth 2.0 (client ID + secret, no API key)

Instead of an API key you can authenticate as a Google account with an **OAuth 2.0
Client ID + Secret** (the option **✨ Gemini (OAuth 2.0)** in the sidebar). This is
future-proof: Google is retiring standard API keys.

1. `pip install google-auth google-auth-oauthlib`
2. In Google Cloud Console → APIs & Services → Credentials → **Create Client**,
   choose **Desktop app**, and note the **Client ID** and **Client secret**.
   (Enable the *Generative Language API* first, and add yourself as a test user
   while the app is in "Testing" status.)
3. Pick **✨ Gemini (OAuth 2.0)** in the sidebar, paste your Client ID + Secret,
   and send your first message — a browser tab opens to sign in with Google.
4. The token is cached in `gemini_oauth_token.json` (auto-refreshed). Keep that
   file private — it can act like a login session.

> Works when the app runs locally (`streamlit run app.py`). It needs a browser on
> the same machine for the one-time login, so it isn't suited to headless servers.

---

## How the offline answers are picked

```
1. Code assistant:  matches your request to a code template (or explains code line by line)
2. Quick tools:     greetings/chitchat/help, unit conversion, average, GCD/LCM, prime,
                    percentage-of, days-between-dates
3. Arithmetic:      the MathSolver turns words into math safely and evaluates it
4. Formulas:        the FormulaEngine matches a formula, extracts the numbers and
                    solves it step by step (with unit conversions)
5. Facts:           TF-IDF retriever finds the best FAQ, showing the subject + confidence
6. Fallback:        "I don't know" + 3 suggested questions
```

The math solver is **safe**: input is whitelisted and evaluated with no Python builtins, so it can never run arbitrary code.

---

## 🎯 Challenges

1. **Make it yours:** add FAQs to `knowledge_base.json`, new formulas to `formulas`, or new snippets to `code_templates` — no code change needed.
2. **Go live with AI:** paste your API key, or set the env variables above, and watch it stream real AI answers.
3. **More code templates:** add templates for any language (React, Java, C++) to `code_templates` — the offline code assistant picks them up automatically.
4. **Deploy:** push to Streamlit Community Cloud and share the link (see notes §9 of Module 9).

> 💡 A chatbot that *computes* problems, *codes* on request, and *grounds its facts* in your knowledge base — while admitting when it doesn't know — is far more trustworthy than one that guesses.
