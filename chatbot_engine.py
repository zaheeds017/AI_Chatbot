"""
chatbot_engine.py - the "brain" of Omni StudyMate, an all-in-one AI study &
daily-life assistant. No Streamlit, no web framework - pure Python.

It answers across ALL school subjects AND everyday tasks:

  * mathematics      - a SAFE arithmetic solver that understands words and symbols
                       ("what is 5 times 8 plus 3 squared?"), percentages,
                       averages, GCD/LCM, prime checks.
  * science          - physics formula calculators: velocity, acceleration, force,
                       work, power, kinetic/potential energy, density, pressure,
                       momentum, Ohm's law, wave speed.
  * geometry         - area/perimeter/volume calculators for circles, squares,
                       rectangles, triangles, trapezoids, spheres, cylinders,
                       cones, cubes, cuboids + Pythagoras theorem.
  * commerce         - simple & compound interest, discount, GST, profit/loss.
  * medical          - BMI calculator (metric + imperial units).
  * daily life       - unit converter (km<->miles, kg<->lb, °C<->°F, litres<->gallons...),
                       days-between-dates, mean/average of a list.
  * all subjects     - a rich FAQ knowledge base (mathematics, science, biology,
                       social studies, English, geometry, medical, arts, commerce,
                       daily life, technology) answered with a TF-IDF retriever.
  * technology       - ready-made CODE templates (Python, HTML, CSS, JavaScript,
                       SQL): 'write a python program to check prime', 'html form',
                       'css center a div', 'explain this code'...

THREE ANSWER MODES (choose from the UI or via USE_REAL_API):
  * MOCK  (default) -> offline, free, instant (calculators + FAQ + code templates)
  * REAL            -> streams answers from a real AI (Gemini / OpenAI / Claude)
                       through ai_providers.py. The app asks for your API key;
                       no key -> it falls back to the offline mock automatically.
  * LEGACY REAL     -> USE_REAL_API = True sends the question to Claude using the
                       anthropic package (kept for CLI users; the app uses ai_providers).
"""

import json
import math as _m
import os
import re
import datetime as _dt
from collections import Counter
from functools import reduce

# ---- Config -------------------------------------------------------------
USE_REAL_API = False          # keep False for the offline mock (no key needed)
MODEL = "claude-opus-5"       # used only when USE_REAL_API is True
SIMILARITY_FLOOR = 0.05       # below this we admit we don't know

HERE = os.path.dirname(__file__)


# ---- Load the knowledge base -------------------------------------------
def load_kb(path=None):
    path = path or os.path.join(HERE, "knowledge_base.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_faqs(kb):
    """Flatten the per-category FAQs into one list, tagging each with its subject."""
    faqs = []
    for cat in kb.get("categories", []):
        for f in cat.get("faqs", []):
            faqs.append({
                **f,
                "category": cat["id"],
                "label": cat["label"],
                "icon": cat.get("icon", ""),
            })
    for f in kb.get("faqs", []):  # backward compatibility with the old JSON shape
        faqs.append({**f, "category": "general", "label": kb.get("domain", "General")})
    return faqs


# ---- Tiny TF-IDF retriever (pure Python) --------------------------------
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of",
    "in", "on", "for", "and", "or", "do", "does", "did", "i", "you", "it", "this",
    "that", "what", "how", "can", "could", "me", "my", "your", "about", "with", "as",
    "at", "by", "from", "will", "would", "should", "if", "so", "we", "they", "its",
    "them", "their", "there", "these", "those", "not", "no", "yes", "am", "have",
    "has", "had", "than", "then", "but", "also", "too", "just", "please", "tell",
    "give", "show", "want", "need", "help",
}

# Spelling variants so "maths" still finds "mathematics", etc.
SYNONYMS = {
    "maths": "mathematics", "math": "mathematics", "bio": "biology",
    "chem": "chemistry", "chemistry": "science", "physics": "science",
    "geo": "geography", "geography": "social_studies", "history": "social_studies",
    "eng": "english", "comp": "computer", "mg": "milligram", "gm": "gram",
}


def tokenize(text):
    """Lowercase, split into word tokens, drop stop-words, apply synonyms."""
    return [SYNONYMS.get(t, t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class Retriever:
    """Ranks FAQ entries by TF-IDF cosine similarity to the user's question."""

    def __init__(self, faqs):
        self.faqs = faqs
        self.docs = [tokenize(f["q"] + " " + f["a"]) for f in faqs]
        self.idf = self._compute_idf(self.docs)
        self.doc_vecs = [self._vectorize(doc) for doc in self.docs]

    def _compute_idf(self, docs):
        n = len(docs)
        df = Counter()
        for doc in docs:
            for term in set(doc):
                df[term] += 1
        return {t: _m.log((n + 1) / (df_t + 1)) + 1 for t, df_t in df.items()}

    def _vectorize(self, tokens):
        tf = Counter(tokens)
        return {t: tf[t] * self.idf.get(t, 0.0) for t in tf}

    @staticmethod
    def _cosine(a, b):
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = _m.sqrt(sum(v * v for v in a.values()))
        nb = _m.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def ranked(self, query, category=None, k=3):
        """Return up to k (score, faq) pairs, optionally within one subject."""
        qv = self._vectorize(tokenize(query))
        scored = []
        for i, dv in enumerate(self.doc_vecs):
            f = self.faqs[i]
            if category and f.get("category") != category:
                continue
            s = self._cosine(qv, dv)
            if s > 0:
                scored.append((s, f))
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:k]

    def best_match(self, query, category=None):
        ranked = self.ranked(query, category, 1)
        if ranked:
            return ranked[0][1], ranked[0][0]
        return None, 0.0

    def suggest(self, query, category=None, k=3):
        return [f for _, f in self.ranked(query, category, k)]


# ---- Number formatting helper -------------------------------------------
def _fmt(x):
    """Pretty-print a number: 2.0 -> 2, 3.14159 stays, 1e+20 becomes scientific."""
    if isinstance(x, bool):
        return "True" if x else "False"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            return "undefined"
        if x == int(x) and abs(x) < 1e16:
            return str(int(x))
        if abs(x) >= 1e15 or (abs(x) < 1e-4 and x != 0):
            return "%.6g" % x
        s = "%.10f" % x
        s = s.rstrip("0").rstrip(".")
        return s
    return str(x)


def _pretty_list(nums):
    nums = [_fmt(n) for n in nums]
    if len(nums) <= 1:
        return ", ".join(nums)
    return ", ".join(nums[:-1]) + " and " + nums[-1]


def _all_numbers(q):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", q)]


# ---- Safe arithmetic solver ---------------------------------------------
class MathSolver:
    """Evaluates arithmetic written with symbols OR English words.

    Safety: input is tokenized and validated against a whitelist, then evaluated
    with no builtins - the expression can never call arbitrary Python.
    """

    FUNCTIONS = {
        "sqrt": _m.sqrt, "abs": abs, "sin": _m.sin, "cos": _m.cos, "tan": _m.tan,
        "log": _m.log10, "ln": _m.log, "exp": _m.exp, "floor": _m.floor,
        "ceil": _m.ceil, "round": round, "min": min, "max": max,
        "factorial": _m.factorial, "asin": _m.asin, "acos": _m.acos,
        "atan": _m.atan, "sinh": _m.sinh, "cosh": _m.cosh, "tanh": _m.tanh,
    }
    CONSTANTS = {"pi": _m.pi, "e": _m.e, "tau": _m.tau}

    _UNITS_NUM = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    }
    _TENS_NUM = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90,
    }
    _SCALE_NUM = {"hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000}
    _ALL_NUM = {**_UNITS_NUM, **_TENS_NUM, **_SCALE_NUM}

    LEAD_PHRASES = [
        "what is the value of", "whats the value of", "what is the", "what are the",
        "whats the", "what's the", "what is", "what's", "whats", "what are", "what was",
        "what were", "what does", "what do", "what equals", "solve the", "solve",
        "calculate the", "calculate", "compute the", "compute", "evaluate", "find the",
        "find", "the value of", "value of", "give me", "tell me the", "tell me",
        "show me", "work out", "how much is", "how much", "how many", "can you",
        "could you", "please", "i want to know", "i need",
    ]

    OP_PHRASES = [
        ("to the power of", "^"),
        ("square root of", "sqrt("),
        ("multiplied by", "*"),
        ("divided by", "/"),
        ("percent of", "/100*"),
        ("square of", "^2"),
        ("square root", "sqrt("),
        ("percent", "/100"),
        ("squared", "^2"),
        ("cubed", "^3"),
        ("divided", "/"),
        ("multiplied", "*"),
        ("multiply", "*"),
        ("divide", "/"),
        ("minus", "-"),
        ("plus", "+"),
        ("times", "*"),
        ("subtract", "-"),
        ("add", "+"),
        ("power", "^"),
        ("over", "/"),
    ]

    def _words_to_num(self, phrase):
        phrase = phrase.replace(" and ", " ").replace(" & ", " ")
        total, current = 0, 0
        for w in phrase.split():
            if w in self._UNITS_NUM:
                current += self._UNITS_NUM[w]
            elif w in self._TENS_NUM:
                current += self._TENS_NUM[w]
            elif w in self._SCALE_NUM:
                scale = self._SCALE_NUM[w]
                if current == 0:
                    current = 1
                if scale == 100:
                    current *= scale
                else:
                    total += current * scale
                    current = 0
            else:
                return None
        return total + current

    def _replace_word_numbers(self, text):
        names = sorted(self._ALL_NUM, key=len, reverse=True)
        pattern = r"\b(?:" + "|".join(names) + r")(?:\s+(?:" + "|".join(names) + r"|and))*"
        return re.sub(pattern, lambda m: str(self._words_to_num(m.group(0))), text,
                      flags=re.IGNORECASE)

    def _strip_lead(self, expr):
        low = expr.lower()
        while True:
            hit = False
            for p in sorted(self.LEAD_PHRASES, key=len, reverse=True):
                if low.startswith(p):
                    expr = expr[len(p):].lstrip()
                    low = expr.lower()
                    hit = True
                    break
            if not hit:
                return expr

    def _looks_mathy(self, q):
        ql = q.lower()
        has_num = bool(re.search(r"\d", ql)) or any(w in ql for w in self._ALL_NUM)
        if not has_num:
            return False
        has_op = any(o in ql for o in "+-*/%^!=x")
        words = ["plus", "minus", "times", "multiply", "divided", "divide", "square",
                 "power", "factorial", "percent", "sqrt", "root", "log", "sin", "cos",
                 "tan", "of"]
        return has_op or any(w in ql for w in words)

    def _normalize(self, q):
        expr = q.strip().lower()
        expr = re.sub(r"[?!.\s]+$", "", expr)
        expr = self._strip_lead(expr)

        expr = re.sub(r"cube\s+root\s+of\s+(\d+(?:\.\d+)?)", r"(\1)**(1/3)", expr)
        expr = self._replace_word_numbers(expr)
        expr = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expr)
        expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"\1/100*\2", expr)
        expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\b", r"\1/100", expr)
        expr = re.sub(r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)\s+of\s+(\d+(?:\.\d+)?)", r"(\1)*\2", expr)
        expr = re.sub(r"(\d+(?:\.\d+)?)\s*degrees?\b", r"(\1*pi/180)", expr)

        # factorial in all its common forms, BEFORE the word->symbol swap
        expr = re.sub(r"(\d+)\s+factorial\b", r"factorial(\1", expr)
        expr = re.sub(r"\bfactorial\s+of\s+(\d+(?:\.\d+)?)", r"factorial(\1", expr)
        expr = re.sub(r"\bfactorial\s+(\d+(?:\.\d+)?)", r"factorial(\1", expr)
        expr = re.sub(r"(\d+)\s*!", r"factorial(\1", expr)

        for phrase, repl in sorted(self.OP_PHRASES, key=lambda t: len(t[0]), reverse=True):
            expr = expr.replace(phrase, repl)

        funcs = "|".join(sorted(self.FUNCTIONS, key=len, reverse=True))
        expr = re.sub(r"\b(%s)\s+(-?\d+(?:\.\d+)?)" % funcs, r"\1(\2", expr)
        expr = re.sub(r"\(\s+", "(", expr)
        expr = re.sub(r"\s+\)", ")", expr)

        opens, closes = expr.count("("), expr.count(")")
        if opens > closes:
            expr += ")" * (opens - closes)

        cleaned = expr
        for name in list(self.FUNCTIONS) + list(self.CONSTANTS):
            cleaned = re.sub(r"\b%s\b" % name, "", cleaned)
        cleaned = cleaned.replace("^", "").replace(",", "")
        if not re.fullmatch(r"[\d+\-*/%^()!.\s]*", cleaned):
            return None
        return expr

    def solve(self, query):
        q = query.strip()
        if not q or not self._looks_mathy(q):
            return None
        expr = self._normalize(q)
        if expr is None:
            return None
        expr_py = expr.replace("^", "**")
        expr_py = re.sub(r"(?<=\d),(?=\d)", "", expr_py)
        ns = dict(self.FUNCTIONS)
        ns.update(self.CONSTANTS)
        try:
            value = eval(expr_py, {"__builtins__": {}}, ns)
        except Exception:
            return None
        if value is None or isinstance(value, complex):
            return None
        if isinstance(value, bool):
            value = int(value)
        return {"expr": expr, "value": value}


# ---- Formula engine (science / geometry / commerce / medical) ------------
class FormulaEngine:
    """Matches a question to a known formula, extracts the numbers and solves it."""

    def match(self, query, formulas):
        q = query.lower()
        best, best_score, best_pos = None, 0, None
        for f in formulas:
            topics = f.get("topics") or []
            if topics and not any(re.search(r"\b%s\b" % t, q) for t in topics):
                continue
            score = 2 if topics else 0
            first_pos = None
            for a in f.get("aliases", []):
                m = re.search(r"\b%s\b" % a, q)
                if m:
                    score += 1
                    if first_pos is None or m.start() < first_pos:
                        first_pos = m.start()
            if score == 0:
                continue
            # Prefer higher score; on a tie, prefer the alias that appears
            # earliest in the question ("force ... acceleration" -> force).
            if best is None:
                best, best_score, best_pos = f, score, first_pos
            elif score > best_score or (score == best_score and first_pos < best_pos):
                best, best_score, best_pos = f, score, first_pos
        return best

    def extract(self, query, formula):
        q = query.lower()
        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", q)]
        values, units_found = {}, {}
        params = formula["params"]

        for p in params:
            matched_alias = None
            # 1) number + unit first (this also applies unit scaling, e.g. g -> kg)
            units = p.get("units", {})
            for u in sorted(units, key=len, reverse=True):
                m = re.search(r"(\d+(?:\.\d+)?)\s*%s(?![a-z0-9])" % re.escape(u), q)
                if m:
                    values[p["key"]] = float(m.group(1)) * units[u]
                    units_found[p["key"]] = u
                    break
            if p["key"] in values:
                continue
            # 2) alias word + number (no unit given)
            for al in p.get("alias", []):
                m = re.search(r"\b%s\b[^\d]{0,16}?(\d+(?:\.\d+)?)" % re.escape(al), q)
                if m:
                    values[p["key"]] = float(m.group(1))
                    matched_alias = al
                    break
            if p["key"] in values:
                if p.get("half") and matched_alias in ("diameter", "dia"):
                    values[p["key"]] /= 2.0
                continue

        missing = [p for p in params if p["key"] not in values]
        consumed = set()
        for p in params:
            if p["key"] in values:
                for i, n in enumerate(nums):
                    if i not in consumed and abs(n - values[p["key"]]) < 1e-9:
                        consumed.add(i)
                        break
        leftovers = [n for i, n in enumerate(nums) if i not in consumed]
        if len(leftovers) == len(missing):
            for p, n in zip(missing, leftovers):
                values[p["key"]] = n
        return values, units_found

    @staticmethod
    def _bmi_comment(bmi):
        if bmi < 18.5:
            cat = "Underweight"
        elif bmi < 25:
            cat = "Normal / healthy weight"
        elif bmi < 30:
            cat = "Overweight"
        else:
            cat = "Obese"
        return (f"\n\nBMI category: **{cat}** (adult reference: <18.5 underweight, "
                "18.5–24.9 normal, 25–29.9 overweight, ≥30 obese)")

    def solve(self, query, formulas):
        f = self.match(query, formulas)
        if not f:
            return None
        values, units = self.extract(query, f)
        params = f["params"]

        missing = [p for p in params if p["key"] not in values]
        if missing:
            found = ", ".join(
                "%s = %s %s" % (p["label"], _fmt(values[p["key"]]), p.get("unit", ""))
                for p in params if p["key"] in values)
            example = f.get("example")
            return ("📐 **%s**\n\nFormula: **%s** (%s)\n\n%s\n\n%s\n"
                    "To calculate it, I still need: **%s**.%s"
                    % (f["name"], f["formula"], f.get("unit", ""), f.get("explanation", ""),
                       ("I found " + found + ".") if found else "I found no numbers.",
                       ", ".join(p["label"] for p in missing),
                       ("\nTry: \"%s\"" % example) if example else ""))

        ns = {p["key"]: values[p["key"]] for p in params}
        ns.update(_m.__dict__)
        try:
            result = eval(f["expr"], {"__builtins__": {}}, ns)
        except Exception:
            return None

        expr_display = f["expr"]
        for p in params:
            expr_display = re.sub(r"\b%s\b" % p["key"], _fmt(values[p["key"]]), expr_display)
        expr_display = expr_display.replace("**", "^")

        extra = ""
        fid = f.get("id")
        if fid == "bmi":
            extra = self._bmi_comment(result)
        elif fid == "compound_interest":
            extra = ("\n\nFinal amount (A): **%s** | Interest earned: **%s**"
                     % (_fmt(result), _fmt(result - values["P"])))
        elif fid == "discount":
            extra = "\n\nYou pay: **%s**" % _fmt(values["price"] - result)
        elif fid == "gst":
            extra = "\n\nTotal including GST: **%s**" % _fmt(values["price"] + result)
        elif fid == "profit_loss":
            if result >= 0:
                extra = "\n\nThat is a **Profit of %s**" % _fmt(result)
            else:
                extra = "\n\nThat is a **Loss of %s**" % _fmt(-result)

        return ("🔬 **%s**\n\nFormula: **%s**\n\n%s = **%s %s**%s\n\n%s"
                % (f["name"], f["formula"], expr_display, _fmt(result),
                   f.get("unit", ""), extra, f.get("explanation", "")))


# ---- Day-to-day special calculators ---------------------------------------
GREETINGS = {
    "hi", "hello", "hey", "hii", "hiii", "hi there", "hello there", "hey there",
    "namaste", "hola", "yo", "good morning", "good afternoon", "good evening",
    "hello bot", "hey bot", "hi bot",
}
THANKS = {
    "thanks", "thank you", "thank you so much", "thanks a lot", "thx", "ty",
    "thankyou", "thank u",
}
BYES = {"bye", "goodbye", "see you", "good night", "see you later", "bye bye"}
CHITCHAT = {
    "how are you": "Running great! 😄 All systems online. What would you like to do today?",
    "how are you doing": "Running great! 😄 All systems online. What would you like to do today?",
    "how are you today": "Running great! 😄 All systems online. What would you like to do today?",
    "what is your name": "I'm **Omni AI** 🤖 - an all-in-one study, coding and daily-life assistant.",
    "who made you": "I was built as a capstone project: a Streamlit chat app powered by the "
                    "engine in `chatbot_engine.py`, backed by `knowledge_base.json` and "
                    "optional real AI via `ai_providers.py`.",
    "what are you made of": "Python, JSON and a healthy dose of Streamlit! 🐍📦",
    "tell me a joke": "Why did the student eat their homework? Because the teacher said it "
                      "was a piece of cake! 🍰",
    "tell me another joke": "What do you call a python that can't code? A 'byte' of "
                            "trouble! 🐍💾",
    "give me a joke": "Why did the student eat their homework? Because the teacher said it "
                      "was a piece of cake! 🍰",
    "what is the meaning of life": "42! But for this bot: learn something new every day. 📚",
    "i love you": "Aww, thanks! 💚 I'm here for your homework, coding and daily questions.",
    "are you human": "Nope - I'm a chatbot! 🤖 I run on Python, with offline calculators and "
                     "an optional real AI connection.",
    "what can you do today": "I solve maths and formulas, answer school-subject questions, "
                             "share code templates, convert units and more. Type **help**!",
    "good": "Glad to hear it! 😊 What can I help you with?",
}

CONVERSIONS = {
    "km": (1000.0, "m"), "kilometer": (1000.0, "m"), "kilometers": (1000.0, "m"),
    "m": (1.0, "m"), "meter": (1.0, "m"), "meters": (1.0, "m"),
    "metre": (1.0, "m"), "metres": (1.0, "m"),
    "cm": (0.01, "m"), "centimeter": (0.01, "m"), "centimeters": (0.01, "m"),
    "centimetre": (0.01, "m"), "centimetres": (0.01, "m"),
    "mm": (0.001, "m"), "millimeter": (0.001, "m"), "millimeters": (0.001, "m"),
    "mile": (1609.34, "m"), "miles": (1609.34, "m"),
    "ft": (0.3048, "m"), "feet": (0.3048, "m"), "foot": (0.3048, "m"),
    "in": (0.0254, "m"), "inch": (0.0254, "m"), "inches": (0.0254, "m"),
    "yard": (0.9144, "m"), "yards": (0.9144, "m"), "yd": (0.9144, "m"),
    "kg": (1.0, "kg"), "kilogram": (1.0, "kg"), "kilograms": (1.0, "kg"),
    "g": (0.001, "kg"), "gram": (0.001, "kg"), "grams": (0.001, "kg"),
    "mg": (1e-06, "kg"), "milligram": (1e-06, "kg"), "milligrams": (1e-06, "kg"),
    "lb": (0.453592, "kg"), "lbs": (0.453592, "kg"),
    "pound": (0.453592, "kg"), "pounds": (0.453592, "kg"),
    "oz": (0.0283495, "kg"), "ounce": (0.0283495, "kg"), "ounces": (0.0283495, "kg"),
    "tonne": (1000.0, "kg"), "ton": (1000.0, "kg"), "tons": (1000.0, "kg"),
    "l": (1.0, "l"), "liter": (1.0, "l"), "liters": (1.0, "l"),
    "litre": (1.0, "l"), "litres": (1.0, "l"),
    "ml": (0.001, "l"), "milliliter": (0.001, "l"), "milliliters": (0.001, "l"),
    "millilitre": (0.001, "l"), "millilitres": (0.001, "l"),
    "gallon": (3.78541, "l"), "gallons": (3.78541, "l"), "gal": (3.78541, "l"),
    "s": (1.0, "s"), "sec": (1.0, "s"), "second": (1.0, "s"), "seconds": (1.0, "s"),
    "min": (60.0, "s"), "minute": (60.0, "s"), "minutes": (60.0, "s"),
    "h": (3600.0, "s"), "hour": (3600.0, "s"), "hours": (3600.0, "s"),
    "hr": (3600.0, "s"), "hrs": (3600.0, "s"),
    "day": (86400.0, "s"), "days": (86400.0, "s"),
    "week": (604800.0, "s"), "weeks": (604800.0, "s"),
    "kph": (1 / 3.6, "m/s"), "kmph": (1 / 3.6, "m/s"), "km/h": (1 / 3.6, "m/s"),
    "kmh": (1 / 3.6, "m/s"), "mph": (0.44704, "m/s"), "mp/h": (0.44704, "m/s"),
    "m/s": (1.0, "m/s"),
    "hectare": (10000.0, "m2"), "hectares": (10000.0, "m2"),
    "acre": (4046.86, "m2"), "acres": (4046.86, "m2"),
}
TEMP = {"celsius": "C", "fahrenheit": "F", "kelvin": "K",
        "degree celsius": "C", "degree fahrenheit": "F"}


def _convert(value, from_unit, to_unit):
    fu, tu = from_unit.lower().strip(), to_unit.lower().strip()
    if fu == tu:
        return value, tu
    if fu in TEMP and tu in TEMP:
        c = value
        if fu == "fahrenheit":
            c = (value - 32) * 5 / 9
        elif fu == "kelvin":
            c = value - 273.15
        if tu == "celsius":
            return c, "°C"
        if tu == "fahrenheit":
            return c * 9 / 5 + 32, "°F"
        return c + 273.15, "K"
    if fu in CONVERSIONS and tu in CONVERSIONS:
        f1, b1 = CONVERSIONS[fu]
        f2, b2 = CONVERSIONS[tu]
        if b1 == b2:
            return value * f1 / f2, tu
    return None, None


def _parse_date(s):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return _dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _is_prime(n):
    n = int(n)
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def _gcd_all(nums):
    return reduce(lambda a, b: _m.gcd(int(a), int(b)), nums)


def capabilities_text(kb):
    lines = []
    for cat in kb.get("categories", []):
        lines.append("%s **%s** — %s" % (cat["icon"], cat["label"], cat["description"]))
    return (
        "I'm an all-in-one AI assistant for studying, coding and daily life. "
        "Here's what I can do:\n\n"
        + "\n".join(lines)
        + "\n\n**💻 Coding:** I share ready-made code templates (Python, HTML, CSS, "
          "JavaScript, SQL) and explain them line by line.\n\n"
        "Try asking:\n"
        "• 'what is 15% of 200?'  • 'force for mass 10 kg and acceleration 5'\n"
        "• 'area of a circle with radius 7'  • 'simple interest on 1000 at 5% for 2 years'\n"
        "• 'BMI for 70 kg and 175 cm'  • 'convert 10 km to miles'\n"
        "• 'write a python program to check prime'  • 'html form'  • 'css center a div'\n"
        "• 'how many days between 2024-01-01 and 2024-03-01'\n"
        "• 'what is a noun?' or any question about any subject."
    )


def _find_template(query, kb):
    """Score every code template by how many of its tags appear in the question."""
    ql = query.lower()
    best, best_score = None, 0
    for t in kb.get("code_templates", []):
        score = 0
        for tag in t.get("tags", []):
            if tag in ql:
                score += 1
        if t.get("language") and t["language"] in ql:
            score += 1
        if score > best_score:
            best, best_score = t, score
    return best


def _explain_template(tpl):
    """A tiny line-by-line walkthrough for offline code explanations."""
    lines = tpl["code"].split("\n")
    notes = []
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("def "):
            notes.append("Line %d: **defines the function** `%s` - a reusable block of logic."
                         % (i, s[4:].split("(")[0]))
        elif s.startswith("class "):
            notes.append("Line %d: **declares the class** `%s` - a blueprint for objects."
                         % (i, s[6:].split("(")[0].split(":")[0]))
        elif s.startswith("for "):
            notes.append("Line %d: a **for loop** - repeats the block that follows." % i)
        elif s.startswith("while "):
            notes.append("Line %d: a **while loop** - repeats while the condition holds." % i)
        elif s.startswith("if ") or s.startswith("elif ") or s.startswith("else"):
            notes.append("Line %d: a **conditional branch** - decides which code runs." % i)
        elif s.startswith("return "):
            notes.append("Line %d: **returns** a value to the caller." % i)
        elif s.startswith("#"):
            notes.append("Line %d: a **comment** - explains what the code does." % i)
        elif "print(" in s:
            notes.append("Line %d: **prints** the result to the console." % i)
    if not notes:
        notes.append("This is a **%s** template - its structure is defined by the %s "
                     "language syntax shown." % (tpl["language"], tpl["language"]))
    return ("```%s\n%s\n```\n\n**How it works, line by line:**\n%s"
            % (tpl["language"], tpl["code"], "\n".join(notes)))


CODE_HARD_HINTS = [
    "write", "make", "create", "generate", "build", "implement", "add",
    "code", "program", "programme", "script", "snippet", "template",
    "html", "css", "javascript", "sql", "form", "div", "page", "website",
    "query", "sort", "loop", "function", "class",
]
CODE_SOFT_HINTS = ["how do i", "how to", "show me", "give me", "i need", "sample",
                   "tell me the code", "code for"]
CODE_SIGNAL = ["python", "code", "program", "html", "css", "javascript", "js",
               "sql", "script", "function", "class", "loop", "snippet"]
CODE_CONCEPT_WORDS = ("what is", "what are", "what's", "define", "describe",
                      "tell me about", "explain in words", "what does")


def try_code(query, kb):
    """Offline code assistant: returns a matching code template when asked."""
    ql = " " + query.lower() + " "

    # 1) explain-a-code request (with or without a matching template)
    if re.search(r"explain\s+(this|the).*(code|program|programme|script|function|class|template)", ql):
        tpl = _find_template(query, kb)
        if tpl:
            return "💡 **%s** - explained:\n\n%s" % (tpl["title"], _explain_template(tpl))
        return ("Paste the code in triple backticks and I'll break it down. In offline "
                "mode I can explain my built-in templates - try **'explain the python "
                "prime program'** - or switch to an online AI model for any code.")

    # conceptual questions ("what is a function?", "define CSS") go to the FAQ
    if ql.strip(" .!?").startswith(CODE_CONCEPT_WORDS):
        return None
    # essay/letter/paragraph requests are writing help, not code
    if re.search(r"write\s+(about|an essay|a paragraph|a letter|a story|a report|a summary)", ql):
        return None

    tpl = _find_template(query, kb)
    hard = any(h in ql for h in CODE_HARD_HINTS)
    soft = any(s in ql for s in CODE_SOFT_HINTS)
    if not (hard or (soft and tpl)):
        return None

    if tpl:
        return ("💻 **%s**\n\n```%s\n%s\n```\n\nWant a variation? Tell me the language "
                "or what to change, or say **'explain this code'** and I'll walk "
                "through it line by line."
                % (tpl["title"], tpl["language"], tpl["code"]))

    if any(s in ql for s in CODE_SIGNAL):
        return ("I have ready-made templates for common tasks: prime check, factorial, "
                "Fibonacci, palindrome, average, sorting, files, classes (Python), HTML "
                "pages/forms, CSS layouts, JavaScript and SQL. Try **'write a python program "
                "to check prime'**, **'html form'**, **'css center a div'** or **'sql query "
                "for students'**.")

    return None


def try_special(query, kb):
    """Quick day-to-day handlers: greetings, help, conversions, dates, averages..."""
    q = re.sub(r"(?<=\d),(?=\d)", "", query.strip())
    ql = q.lower()

    if ql in GREETINGS:
        return ("Hello! 👋 I'm **Omni AI**, your all-in-one study, coding & daily-life "
                "assistant. I solve maths, science and geometry problems, calculate BMI, "
                "interest, GST and discounts, convert units, count days between dates, "
                "share code templates, and answer questions across every school subject. "
                "Type **help** to see what I can do.")
    if ql in THANKS:
        return "You're welcome! 😊 Ask me anything else whenever you need it."
    if ql in BYES:
        return "Goodbye! 👋 Happy studying - come back any time."
    if ql in CHITCHAT:
        return CHITCHAT[ql]
    if ql in {"help", "menu", "commands", "features", "capabilities", "what can you do",
              "what do you do", "how does this work",
              "how does this chatbot work"}:
        return capabilities_text(kb)

    # ---- unit conversion ----
    m1 = re.search(r"convert\s+([\d.]+)\s*([a-z][a-z0-9°/]*)\s+to\s+([a-z][a-z0-9°/]*)", ql)
    m2 = re.search(r"([\d.]+)\s*([a-z][a-z0-9°/]*)\s+to\s+([a-z][a-z0-9°/]*)", ql)
    m3 = re.search(r"([\d.]+)\s*([a-z][a-z0-9°/]*)\s+in\s+([a-z][a-z0-9°/]*)", ql)
    m4 = re.search(r"how\s+many\s+([a-z][a-z0-9°/]*)\s+(?:are\s+)?in\s+([\d.]+)\s*([a-z][a-z0-9°/]*)", ql)
    if m4:
        value, from_unit, to_unit = float(m4.group(2)), m4.group(3), m4.group(1)
    elif m1 or m2:
        mm = m1 or m2
        value, from_unit, to_unit = float(mm.group(1)), mm.group(2), mm.group(3)
    elif m3:
        value, from_unit, to_unit = float(m3.group(1)), m3.group(2), m3.group(3)
    if m4 or m1 or m2 or m3:
        res, unit = _convert(value, from_unit, to_unit)
        if res is not None:
            return ("🔄 **Conversion:** %s %s = **%s %s**"
                    % (_fmt(value), from_unit, _fmt(res), unit))

    # ---- average / mean of a list ----
    if re.search(r"\b(average|mean|avg)\b", ql):
        nums = _all_numbers(q)
        if len(nums) >= 2:
            return ("📊 The **average (mean)** of %s is **%s**."
                    % (_pretty_list(nums), _fmt(sum(nums) / len(nums))))

    # ---- GCD / HCF ----
    if re.search(r"\b(gcd|gcf|hcf)\b", ql):
        nums = _all_numbers(q)
        if len(nums) >= 2:
            return ("🔢 The **GCD / HCF** of %s is **%d**." % (_pretty_list(nums), _gcd_all(nums)))

    # ---- LCM ----
    if re.search(r"\blcm\b", ql):
        nums = _all_numbers(q)
        if len(nums) >= 2:
            lcm = 1
            for n in nums:
                lcm = lcm * int(n) // _m.gcd(lcm, int(n))
            return ("🔢 The **LCM** of %s is **%d**." % (_pretty_list(nums), lcm))

    # ---- prime check ----
    if re.search(r"\bprime\b", ql):
        nums = _all_numbers(q)
        if nums:
            n = int(abs(nums[0]))
            if _is_prime(n):
                return ("🔢 Yes, **%d** is a prime number." % n)
            return ("🔢 No, **%d** is not a prime number." % n)

    # ---- percentage of ----
    m = re.search(r"(?:what|how much|how many)\s+percent(?:age)?\s+of\s+([\d.]+)\s+is\s+([\d.]+)", ql)
    if m:
        whole, part = float(m.group(1)), float(m.group(2))
        return ("📈 **%s** is **%s%%** of **%s**."
                % (_fmt(part), _fmt(part / whole * 100), _fmt(whole)))

    # ---- symbolic equations / unknown variables ----
    if ("=" in ql and re.search(r"[a-z]", ql)) or re.search(r"\bsolve\b|\bfind\s+[xyz]\b|\bsolve\s+for\b", ql) or re.search(r"(?<=\d)[xyz](?!\d)", ql):
        return ("I can't solve symbolic equations yet (like \"2x + 3 = 11\"), but I can do "
                "any arithmetic or known formula! Try 'what is 2*4+3', 'force for mass 10 and "
                "acceleration 5', or ask about a concept from any subject.")

    # ---- days between two dates ----
    dates = re.findall(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b", q)
    if len(dates) >= 2:
        d1, d2 = _parse_date(dates[0]), _parse_date(dates[1])
        if d1 and d2:
            days = abs((d2 - d1).days)
            return ("📅 There are **%d day%s** between %s and %s."
                    % (days, "s" if days != 1 else "", dates[0], dates[1]))

    return None


# ---- Answering modes ------------------------------------------------------
def math_reply(solved):
    return ("🧮 Let me compute that for you:\n\n"
            "  `%s`\n\n"
            "**Answer:** %s" % (solved["expr"], _fmt(solved["value"])))


def answer_mock(query, retriever, kb, category=None):
    """Offline answer pipeline: code -> special tools -> math -> formulas -> FAQ -> fallback."""
    q = query.strip()
    if not q:
        return "Please type a question."

    code = try_code(q, kb)
    if code:
        return code

    special = try_special(q, kb)
    if special:
        return special

    solved = MathSolver().solve(q)
    if solved:
        return math_reply(solved)

    nums = _all_numbers(q)
    if nums:
        formulas = kb.get("formulas", [])
        result = FormulaEngine().solve(q, formulas)
        if result:
            return result

    faq, score = retriever.best_match(q, category)
    if faq and score >= SIMILARITY_FLOOR:
        head = "[%s %s · confidence %d%%]" % (faq.get("icon", ""), faq["label"],
                                              round(score * 100))
        return head + "\n\n" + faq["a"]

    parts = ["Hmm, I don't have a confident answer for that.", ""]
    suggestions = retriever.suggest(q, category, 3)
    if suggestions:
        parts.append("Did you mean:")
        parts += ["• " + s["q"] for s in suggestions]
        parts.append("")
    parts.append("Or try one of my tools: maths, formulas, unit conversion, "
                 "averages, or date differences. Type **help** to see everything.")
    return "\n".join(parts)


def build_system_prompt(kb):
    """Turn the whole knowledge base into a system prompt for a real AI model."""
    lines = []
    for cat in kb.get("categories", []):
        lines.append("CATEGORY: %s" % cat["label"])
        for f in cat.get("faqs", []):
            lines.append("Q: %s\nA: %s" % (f["q"], f["a"]))
    facts = "\n".join(lines)
    return (
        "You are Omni AI, an all-in-one study, coding and daily-life assistant "
        "covering mathematics, science, biology, social studies, English, geometry, "
        "medical, arts, commerce, technology and everyday applications. "
        "Answer accurately and helpfully. For calculations, solve them step by step. "
        "When the user asks for code, write clear, working code with a short "
        "explanation. When the answer is not in the facts below, say you do not know. "
        "Keep replies clear and not too long.\n\nFACTS:\n" + facts
    )


def answer_real(query, history, kb):
    """Online answer via Claude. Only called when USE_REAL_API is True."""
    import anthropic  # imported here so mock mode needs nothing installed

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    system = build_system_prompt(kb)
    messages = history + [{"role": "user", "content": query}]
    resp = client.messages.create(
        model=MODEL, max_tokens=600, system=system, messages=messages
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def greeting_text(kb):
    return ("Hello! I'm **Omni AI** 🤖, your all-in-one study, coding and daily-life "
            "assistant. I can solve maths, science and geometry problems, calculate "
            "BMI, interest, GST and discounts, convert units, count days between dates, "
            "share code templates, and answer questions across mathematics, science, "
            "biology, social studies, English, geometry, medical, arts, commerce, "
            "technology and daily life.\n\n"
            "Type **help** to see everything I can do, or just ask me anything.")


def get_response(query, history, kb, retriever, category=None):
    """Single entry point the app calls. Chooses mock or real automatically."""
    if USE_REAL_API:
        return answer_real(query, history, kb)
    return answer_mock(query, retriever, kb, category)


if __name__ == "__main__":
    # Quick offline self-test:  python chatbot_engine.py
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _kb = load_kb()
    _r = Retriever(flatten_faqs(_kb))
    _tests = [
        "what is 2 plus 2?",
        "what is the square root of 144",
        "what is 15% of 200",
        "force for mass 10 kg and acceleration 5",
        "area of a circle with radius 7",
        "simple interest on 1000 at 5% for 2 years",
        "BMI for 70 kg and 175 cm",
        "convert 10 km to miles",
        "how many days between 2024-01-01 and 2024-03-01",
        "what is photosynthesis",
        "what is a noun",
        "write a python program to check prime",
        "explain the python prime program",
        "what is css",
        "how are you",
        "help",
    ]
    for q in _tests:
        print("Q:", q)
        print("A:", answer_mock(q, _r, _kb), "\n")
