"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import base64
import binascii
import codecs
import re
import textwrap

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def _squash(text: str) -> str:
    """Lower-case and keep only letters/digits (drops spaces, dashes, dots...)."""
    return re.sub(r"[^a-z0-9]", "", text.casefold())


# One entry per distinct secret (DEMO_SECRETS also lists squashed duplicates).
_SECRETS: list[str] = []
for _s in DEMO_SECRETS:
    if len(_squash(_s)) >= 6 and _squash(_s) not in {_squash(x) for x in _SECRETS}:
        _SECRETS.append(_s)
_SECRET_KEYS = [_squash(s) for s in _SECRETS]


def _tolerant_pattern(secret: str) -> re.Pattern:
    """Matches the secret even with up to 3 separator chars between characters
    (``a-d-m-i-n-1-2-3``, ``a d m i n 1 2 3``)."""
    chars = [re.escape(c) for c in secret if c.isalnum()]
    return re.compile(r"[\W_]{0,3}".join(chars), re.IGNORECASE)


_SECRET_PATTERNS = [_tolerant_pattern(s) for s in _SECRETS]
_B64_TOKEN = re.compile(r"[A-Za-z0-9+/_-]{8,}={0,2}")
_HEX_TOKEN = re.compile(r"(?:[0-9a-fA-F]{2}){4,}")


def _decoded_views(text: str) -> list[tuple[str, str]]:
    """Alternative readings of ``text`` an attacker could use to smuggle a secret
    past a literal match: reversed, ROT13, base64 and hex tokens (+ reversed)."""
    views: list[tuple[str, str]] = [
        ("reversed", text[::-1]),
        ("rot13", codecs.decode(text, "rot13")),
    ]
    for tok in _B64_TOKEN.findall(text)[:50]:
        try:
            padded = tok.replace("-", "+").replace("_", "/")
            padded += "=" * (-len(padded) % 4)
            views.append(("base64", base64.b64decode(padded).decode("utf-8", "ignore")))
        except (binascii.Error, ValueError):
            pass
    for tok in _HEX_TOKEN.findall(text)[:50]:
        try:
            views.append(("hex", bytes.fromhex(tok).decode("utf-8", "ignore")))
        except ValueError:
            pass
    views += [(f"{k}+reversed", v[::-1]) for k, v in list(views) if k in ("base64", "hex")]
    return views


def find_encoded_secret(text: str) -> list[str]:
    """Names of the encodings (reversed/rot13/base64/hex) under which a protected
    secret appears in ``text``. Deterministic: only the known secrets matter, so
    this works whatever phrasing the attacker used to ask for them."""
    found = []
    for kind, view in _decoded_views(text):
        squashed = _squash(view)
        if any(key in squashed for key in _SECRET_KEYS) and kind not in found:
            found.append(kind)
    return found


# name -> compiled regex. A group named ``x`` means "redact only that group"
# (keeps the label, e.g. "CCCD [REDACTED]"). Order matters.
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # CCCD: 12 digits (optionally grouped by 3). A bare 9-digit number is NOT an ID
    # (it is usually an amount); CMND 9 digits only counts when labelled.
    ("national_id", re.compile(r"(?<!\d)\d{3}[ .-]?\d{3}[ .-]?\d{3}[ .-]?\d{3}(?!\d)")),
    ("national_id", re.compile(
        r"(?:cmnd|cccd|cmt|can\s*cuoc|căn\s*cước|chứng\s*minh|national\s*id|id\s*number)"
        r"[^\d]{0,20}(?P<x>\d{9}|\d{12})(?!\d)", re.IGNORECASE)),
    # VN phone: 0xxxxxxxxx / +84xxxxxxxxx, tolerating spaces, dots, dashes
    ("phone", re.compile(r"(?<!\d)(?:\+84|0)(?:[\s.\-]?\d){9,10}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")),
    ("email", re.compile(
        r"[\w.+-]+\s*(?:\(at\)|\[at\])\s*[\w-]+(?:\s*(?:\(dot\)|\[dot\])\s*[\w-]+)+",
        re.IGNORECASE)),
    # Database port 5432 (part of the protected ``db.vinbank.internal:5432``), but only
    # in a database context, so an ordinary number such as "reference 5432" is kept.
    # host:5432 -- the left token must contain a letter (db01, internal, [REDACTED]),
    # so a time-like "10:5432" is left alone
    ("db_port", re.compile(r"[\w.\-\]]*[A-Za-z\]][\w.\-\]]*:(?P<x>5432)(?!\d)")),
    # "port 5432", "the port is 5432", "cổng: 5432"
    ("db_port", re.compile(
        r"(?:\bport\b|\bcổng\b|\bcong\b)[^.\n\d]{0,20}(?P<x>5432)(?!\d)", re.IGNORECASE)),
    ("db_port", re.compile(
        r"\b(?:database|db|postgres\w*|sql)\b[^.\n]{0,40}?(?P<x>5432)(?!\d)", re.IGNORECASE)),
    ("api_key", re.compile(r"\bsk-[a-zA-Z0-9_-]+", re.IGNORECASE)),
    ("password", re.compile(r"password\s*(?:is\s*|[:=]\s*)\S+", re.IGNORECASE)),
]


def _redact(pattern: re.Pattern, text: str) -> tuple[str, int]:
    def repl(m: re.Match) -> str:
        if "x" in pattern.groupindex:
            s, e, off = m.start("x"), m.end("x"), m.start()
            return m.group(0)[: s - off] + "[REDACTED]" + m.group(0)[e - off:]
        return "[REDACTED]"

    return pattern.subn(repl, text)


def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    original = response or ""
    redacted = original

    # 1. Label the findings on the ORIGINAL text, so every PII / credential type
    #    present is reported (e.g. both api_key and password in one reply).
    for name, pattern in _PII_PATTERNS:
        n = len(pattern.findall(original))
        if n:
            issues.append(f"{name}: {n} found")

    # 2. Cut out known protected values first, literal or with separators between
    #    characters. This must run BEFORE the generic patterns below: the
    #    ``password ...`` pattern would otherwise swallow only the first character
    #    of ``a d m i n 1 2 3`` and leave the rest of the secret visible.
    for pattern in _SECRET_PATTERNS:
        redacted, n = pattern.subn("[REDACTED]", redacted)
        if n:
            issues.append("secret: protected value found")

    # 3. Then redact the generic PII / credential formats
    for _name, pattern in _PII_PATTERNS:
        redacted, _n = _redact(pattern, redacted)

    # 4. Known protected values hidden by an encoding (reversed / ROT13 /
    #    base64 / hex): cannot be cut out precisely, so withhold the whole reply.
    kinds = find_encoded_secret(redacted)
    if kinds:
        issues.append(f"secret: encoded value ({', '.join(kinds)})")
        redacted = "[REDACTED] (reply withheld: it contained an encoded protected value)"

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        # 1. Redact PII / secrets
        result = content_filter(response_text)
        if not result["safe"]:
            self.redacted_count += 1
            llm_response.content = types.Content(
                role="model",
                parts=[types.Part.from_text(text=result["redacted"])],
            )
            response_text = result["redacted"]

        # 2. Optional LLM-as-Judge
        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                llm_response.content = types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="I'm sorry, I can't provide that response."
                    )],
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
