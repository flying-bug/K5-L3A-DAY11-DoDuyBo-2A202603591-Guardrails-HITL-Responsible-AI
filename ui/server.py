"""
Lab 11 — local UI server.

Run from repo root:   python ui/server.py      → http://127.0.0.1:8765

  GET  /            → ui/index.html   (chat + kết quả bài làm)
  GET  /trace       → ui/trace.html   (xem code chạy từng bước)
  GET  /api/data    → contents of outputs/*.json (read-only, never modified)
  POST /api/chat    → one message through the REAL Blue pipeline
  POST /api/trace   → same run, plus a step-by-step trace of every layer

Stdlib only. Binds to localhost. API keys stay in .env and are never sent
to the browser. Nothing here writes to outputs/ (graded artifacts stay intact).

How the trace works: the real plugin methods are wrapped (on this process's
plugin *instances* only; your source files are untouched) to record the order,
the input/output and the timing of every layer for the current request.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import core.config as cfg  # noqa: E402  (also loads .env)
from core.utils import chat_with_agent  # noqa: E402
from agents.agent import create_blue_agent  # noqa: E402
from assignment.pipeline import (  # noqa: E402
    build_production_plugins,
    _attribute_layer,
    _counters,
)
import assignment.rate_limiter as rate_mod  # noqa: E402
import guardrails.input_guardrails as ig  # noqa: E402
import guardrails.output_guardrails as og  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
MAX_MESSAGE_CHARS = 4000
DATA_FILES = ("results", "attack_results", "metrics", "audit_log")

PLUGINS = build_production_plugins(use_llm_judge=False)
AGENT, RUNNER = create_blue_agent(PLUGINS)
LOCK = threading.Lock()  # plugins keep mutable counters → one request at a time

# ---------------------------------------------------------------------------
# Per-request recorder
# ---------------------------------------------------------------------------
EVENTS: list[dict] = []   # one entry per plugin callback actually executed
RAW: dict = {}            # raw LLM reply (before output plugins)


def _limiter():
    return next(p for p in PLUGINS if p.name == "rate_limiter")


def rate_limit_state() -> dict:
    lim = _limiter()
    window = lim.user_windows.get("student", [])
    now = time.time()
    used = sum(1 for t in window if t > now - lim.window_seconds)
    return {"used": used, "max": lim.max_requests, "window": lim.window_seconds}


def _text_of(content) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


def _instrument(plugin, method: str) -> None:
    original = getattr(plugin, method)

    async def wrapper(*args, **kwargs):
        pre = rate_limit_state() if plugin.name == "rate_limiter" else None
        before_text = None
        if plugin.name == "output_guardrail":
            before_text = plugin._extract_text(kwargs.get("llm_response"))
        t0 = time.perf_counter()
        result = await original(*args, **kwargs)
        t1 = time.perf_counter()
        after_text = None
        if plugin.name == "output_guardrail":
            after_text = plugin._extract_text(kwargs.get("llm_response"))
        EVENTS.append({
            "stage": plugin.name, "t0": t0, "t1": t1, "ms": (t1 - t0) * 1000,
            "blocked": result is not None and plugin.name != "output_guardrail",
            "pre": pre, "before": before_text, "after": after_text,
        })
        return result

    setattr(plugin, method, wrapper)


for _p in PLUGINS:
    if hasattr(_p, "on_user_message_callback"):
        _instrument(_p, "on_user_message_callback")
    if _p.name == "output_guardrail":
        _instrument(_p, "after_model_callback")


def _capture_raw(text: str) -> str:
    """Runner output hook: runs right after the LLM call, before output plugins."""
    RAW["t"] = time.perf_counter()
    RAW["text"] = text
    return text


RUNNER.output_hooks.append(_capture_raw)


def execute(message: str) -> dict:
    """Run one message through the real pipeline and collect what happened."""
    message = (message or "")[:MAX_MESSAGE_CHARS]
    with LOCK:
        EVENTS.clear()
        RAW.clear()
        before = _counters(PLUGINS)
        started = time.perf_counter()
        error, reply = None, ""
        try:
            reply, _ = asyncio.run(chat_with_agent(AGENT, RUNNER, message))
        except Exception as exc:  # network / quota / bad key
            error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()
        layer = _attribute_layer(before, _counters(PLUGINS))
        return {
            "message": message, "reply": reply or "", "error": error,
            "layer": layer, "blocked": layer is not None,
            "events": list(EVENTS), "raw": dict(RAW),
            "started": started, "finished": finished,
            "rate_limit": rate_limit_state(),
        }


# ---------------------------------------------------------------------------
# /api/chat
# ---------------------------------------------------------------------------
def handle_chat(message: str) -> dict:
    run = execute(message)
    return {
        "reply": run["reply"], "blocked": run["blocked"], "layer": run["layer"],
        "checks": {"injection": ig.detect_injection(run["message"]),
                   "topic": ig.topic_filter(run["message"])},
        "latency_ms": round((run["finished"] - run["started"]) * 1000),
        "error": run["error"], "rate_limit": run["rate_limit"],
    }


# ---------------------------------------------------------------------------
# /api/trace
# ---------------------------------------------------------------------------
def _code(obj) -> dict:
    lines, start = inspect.getsourcelines(obj)
    return {
        "file": Path(inspect.getsourcefile(obj)).relative_to(ROOT).as_posix(),
        "line": start,
        "source": "".join(lines),
    }


def _visible(text: str) -> str:
    """Show invisible / control characters explicitly so the user can see them."""
    out = []
    for ch in text:
        if ig._INVISIBLE_CHARS.match(ch) or (ord(ch) < 32 and ch not in "\n\t"):
            out.append(f"⟨U+{ord(ch):04X}⟩")
        else:
            out.append(ch)
    return "".join(out)


def _br(label: str, taken: bool, then: str) -> dict:
    """One possible branch of a step: condition, whether it was taken, what follows."""
    return {"label": label, "taken": bool(taken), "then": then}


def _clip(text, n: int = 300) -> str:
    text = text if isinstance(text, str) else str(text)
    return text if len(text) <= n else text[:n] + "…"


def _step(sid, title, status, summary, details=None, ms=None, code=None, extra=None,
          inp=None, out=None, branches=None):
    return {"id": sid, "title": title, "status": status, "summary": summary,
            "details": details or [], "ms": None if ms is None else round(ms, 2),
            "code": code, "extra": extra,
            "input": None if inp is None else _clip(inp),
            "output": None if out is None else _clip(out),
            "branches": branches or []}


def build_trace(run: dict) -> dict:
    msg = run["message"]
    ev = {e["stage"]: e for e in run["events"]}
    steps = []

    # 0. Receive + canonicalize ------------------------------------------------
    canon = ig._canonicalize(msg)
    hidden = sum(1 for ch in msg if ig._INVISIBLE_CHARS.match(ch))
    steps.append(_step(
        "receive", "Nhận câu hỏi & chuẩn hóa", "info",
        "Chuẩn hóa Unicode (NFKC), xóa ký tự ẩn, bỏ dấu tiếng Việt, gộp khoảng trắng "
        "— để regex không bị qua mặt bằng ký tự lạ.",
        [["Độ dài", f"{len(msg)} ký tự"],
         ["Ký tự ẩn bị xóa", str(hidden)],
         ["Gốc", _visible(msg)[:400] or "(rỗng)"],
         ["Sau chuẩn hóa", canon[:400] or "(rỗng)"]],
        code=_code(ig._canonicalize),
        inp=_visible(msg) or "(rỗng)", out=canon or "(rỗng)",
        branches=[
            _br("Có ký tự ẩn / dấu / khoảng trắng thừa → làm sạch", canon != msg, "Rate limiter"),
            _br("Văn bản đã sạch → giữ nguyên", canon == msg, "Rate limiter"),
        ]))

    stopped = None  # layer name that blocked

    # 1. Rate limiter ---------------------------------------------------------
    e = ev.get("rate_limiter")
    if e:
        pre = e["pre"]
        blocked = e["blocked"]
        steps.append(_step(
            "rate_limiter", "Rate limiter (sliding window)",
            "block" if blocked else "pass",
            "Vượt giới hạn → chặn, không gọi LLM." if blocked
            else "Còn trong giới hạn → ghi timestamp và cho qua.",
            [["Đã dùng trước câu này", f"{pre['used']}/{pre['max']} trong {pre['window']}s"],
             ["Quyết định", "CHẶN" if blocked else "CHO QUA"]],
            ms=e["ms"], code=_code(rate_mod.RateLimitPlugin.on_user_message_callback),
            inp=f"user_id = student · đã dùng {pre['used']}/{pre['max']} trong {pre['window']}s",
            out=(_clip(run["reply"]) if blocked else "None (cho qua)"),
            branches=[
                _br(f"used < {pre['max']} → ghi timestamp, return None", not blocked, "Input guardrail"),
                _br(f"used ≥ {pre['max']} → trả “Rate limit exceeded”", blocked, "KẾT THÚC (không gọi LLM)"),
            ]))
        if blocked:
            stopped = "rate_limiter"
    else:
        steps.append(_step("rate_limiter", "Rate limiter (sliding window)", "skip",
                           "Không chạy.", code=_code(rate_mod.RateLimitPlugin.on_user_message_callback)))

    # 2. Input guardrail ------------------------------------------------------
    e = ev.get("input_guardrail")
    if stopped:
        steps.append(_step("input_guardrail", "Input guardrail", "skip",
                           f"Bỏ qua vì {stopped} đã chặn.",
                           code=_code(ig.InputGuardrailPlugin.on_user_message_callback)))
    elif e:
        long_msg = ig.too_long(msg)
        steps.append(_step(
            "length", "Input · giới hạn độ dài", "block" if long_msg else "pass",
            (f"{len(msg)} ký tự > giới hạn {ig.MAX_INPUT_CHARS} → chặn trước khi tốn công xử lý."
             if long_msg else f"{len(msg)} ký tự ≤ giới hạn {ig.MAX_INPUT_CHARS} → tiếp tục."),
            [["Độ dài", f"{len(msg)} ký tự"], ["Giới hạn", f"{ig.MAX_INPUT_CHARS} ký tự"]],
            code=_code(ig.too_long), inp=f"{len(msg)} ký tự",
            out="BLOCK" if long_msg else "ALLOW",
            branches=[
                _br(f"Dài hơn {ig.MAX_INPUT_CHARS} ký tự → BLOCK (chống làm tràn token / tốn tiền)",
                    long_msg, "KẾT THÚC (chặn)"),
                _br("Trong giới hạn → ALLOW", not long_msg, "Phát hiện injection"),
            ]))
        found = None
        if long_msg:
            stopped = "input_guardrail"
            steps.append(_step("injection", "Input · phát hiện prompt injection", "skip",
                               "Bỏ qua vì câu quá dài đã bị chặn.", code=_code(ig.find_injection)))
            steps.append(_step("topic", "Input · lọc chủ đề", "skip",
                               "Bỏ qua vì câu quá dài đã bị chặn.", code=_code(ig.topic_filter)))
        else:
            found = ig.find_injection(msg)
            matched_idx = found[0] if found else None
            pats = [{"i": i, "pattern": p,
                     "state": ("match" if i == matched_idx else
                               "no" if matched_idx is None or i < matched_idx else "unchecked")}
                    for i, p in enumerate(ig.INJECTION_PATTERNS)]
            steps.append(_step(
                "injection", "Input · phát hiện prompt injection",
                "block" if found else "pass",
                (f"Khớp regex #{found[0] + 1}: “{found[2]}”" if found
                 else f"Không khớp trong {len(pats)} regex."),
                [["Kết quả detect_injection()", ig.detect_injection(msg)],
                 ["Đoạn khớp", found[2] if found else "–"]],
                code=_code(ig.find_injection), extra={"patterns": pats},
                inp=canon or "(rỗng)",
                out="BLOCK" if found else "ALLOW",
                branches=[
                    _br(f"Khớp 1 trong {len(pats)} regex → BLOCK", bool(found), "KẾT THÚC (chặn)"),
                    _br("Không regex nào khớp → ALLOW", not found, "Lọc chủ đề"),
                ]))
            if found:
                stopped = "input_guardrail"
                steps.append(_step(
                    "topic", "Input · lọc chủ đề", "skip",
                    "Chưa cần chạy vì injection đã chặn câu này.", code=_code(ig.topic_filter)))
            else:
                low = canon.lower()
                blocked_hits = [t for t in cfg.BLOCKED_TOPICS if re.search(rf"\b{re.escape(t)}", low)]
                allowed_hits = [t for t in cfg.ALLOWED_TOPICS if re.search(rf"\b{re.escape(t)}", low)]
                m = ig._EXTRA_BANKING_RE.search(low)
                if m:
                    allowed_hits.append(m.group(0))
                verdict = ig.topic_filter(msg)
                if blocked_hits:
                    why = f"Có chủ đề bị cấm: {', '.join(blocked_hits)}"
                elif not allowed_hits:
                    why = "Không có từ khóa banking nào → lạc đề"
                else:
                    why = f"Có từ khóa banking: {', '.join(allowed_hits)}"
                steps.append(_step(
                    "topic", "Input · lọc chủ đề",
                    "block" if verdict == "BLOCK" else "pass", why,
                    [["Từ khóa bị cấm", ", ".join(blocked_hits) or "–"],
                     ["Từ khóa banking", ", ".join(allowed_hits) or "–"],
                     ["Kết quả topic_filter()", verdict]],
                    code=_code(ig.topic_filter),
                    inp=low or "(rỗng)", out=verdict,
                    branches=[
                        _br("Có từ khóa bị cấm (hack, bomb, …) → BLOCK", bool(blocked_hits),
                            "KẾT THÚC (chặn)"),
                        _br("Không có từ khóa banking nào (lạc đề / rỗng) → BLOCK",
                            not blocked_hits and not allowed_hits, "KẾT THÚC (chặn)"),
                        _br("Có từ khóa banking, không có từ cấm → ALLOW",
                            not blocked_hits and bool(allowed_hits), "Input guardrail quyết định"),
                    ]))
                if verdict == "BLOCK":
                    stopped = "input_guardrail"
        steps.append(_step(
            "input_guardrail", "Input guardrail → quyết định",
            "block" if stopped == "input_guardrail" else "pass",
            "Trả message chặn, không gọi LLM." if stopped == "input_guardrail"
            else "Cả hai đều ALLOW → trả None, cho qua.",
            ms=e["ms"], code=_code(ig.InputGuardrailPlugin.on_user_message_callback),
            inp=(f"độ dài = {len(msg)} ký tự (vượt {ig.MAX_INPUT_CHARS})" if long_msg else
                 f"detect_injection = {ig.detect_injection(msg)} · topic_filter = "
                 f"{'(không chạy)' if found else ig.topic_filter(msg)}"),
            out=(_clip(run["reply"]) if stopped == "input_guardrail" else "None (cho qua)"),
            branches=[
                _br("quá dài → trả “Message too long…”", long_msg, "KẾT THÚC (chặn)"),
                _br("injection = BLOCK → trả “Request blocked…”", bool(found), "KẾT THÚC (chặn)"),
                _br("injection = ALLOW, topic = BLOCK → trả “only help with banking…”",
                    (not found) and (not long_msg) and stopped == "input_guardrail",
                    "KẾT THÚC (chặn)"),
                _br("cả hai = ALLOW → return None", stopped != "input_guardrail", "Gọi LLM"),
            ]))
    else:
        steps.append(_step("input_guardrail", "Input guardrail", "skip", "Không chạy.",
                           code=_code(ig.InputGuardrailPlugin.on_user_message_callback)))

    # 3. LLM ------------------------------------------------------------------
    called = "text" in run["raw"]
    if called:
        t_in = ev["input_guardrail"]["t1"] if "input_guardrail" in ev else run["started"]
        llm_ms = (run["raw"]["t"] - t_in) * 1000
        raw = run["raw"]["text"]
        steps.append(_step(
            "llm", "Gọi LLM (agent Blue)", "pass",
            "Câu hỏi qua hết lớp trước → gửi tới model.",
            [["Model", f"{cfg.get_blue_provider()}:{cfg.get_blue_model()}"],
             ["Độ dài trả lời thô", f"{len(raw)} ký tự"],
             ["Trả lời thô (đầu)", raw[:300] or "(trống)"]],
            ms=llm_ms, inp=msg or "(rỗng)", out=raw or "(trống)",
            branches=[
                _br("API trả lời bình thường", True, "Output guardrail"),
                _br("Lỗi mạng / quota / sai key → báo lỗi", False, "KẾT THÚC (lỗi)"),
            ]))
    else:
        reason = (f"Bị chặn ở {stopped} nên không tốn API." if stopped
                  else (f"Lỗi khi gọi: {run['error']}" if run["error"] else "Không chạy."))
        steps.append(_step(
            "llm", "Gọi LLM (agent Blue)", "skip", reason, inp=msg or "(rỗng)", out="(không gọi)",
            branches=[
                _br("Các lớp trước cho qua → gọi model", False, "Output guardrail"),
                _br("Một lớp trước đã chặn → không gọi, không tốn API", bool(stopped), "KẾT THÚC"),
                _br("Gọi nhưng lỗi API", bool(run["error"]) and not stopped, "KẾT THÚC (lỗi)"),
            ]))

    # 4. Output guardrail -----------------------------------------------------
    e = ev.get("output_guardrail")
    if e and called:
        raw = run["raw"]["text"]
        res = og.content_filter(raw)
        changed = (e["before"] != e["after"])
        steps.append(_step(
            "output_guardrail", "Output guardrail (che PII / secret)",
            "block" if changed else "pass",
            ("Phát hiện và đã che: " + ", ".join(res["issues"])) if changed
            else "Không thấy PII / secret → giữ nguyên câu trả lời.",
            [["Vấn đề phát hiện", ", ".join(res["issues"]) or "không có"],
             ["Trước khi lọc (đầu)", (e["before"] or "")[:200]],
             ["Sau khi lọc (đầu)", (e["after"] or "")[:200]]],
            ms=e["ms"], code=_code(og.content_filter),
            inp=raw, out=e["after"] or "(trống)",
            branches=[
                _br("Thấy PII / secret (SĐT, email, CCCD, sk-…, password) → thay bằng [REDACTED]",
                    changed, "Trả cho người dùng (đã che)"),
                _br("Sạch → giữ nguyên", not changed, "Trả cho người dùng"),
            ]))
        if changed:
            stopped = stopped or "output_guardrail"
    else:
        steps.append(_step(
            "output_guardrail", "Output guardrail (che PII / secret)", "skip",
            "Không chạy vì LLM không được gọi.", code=_code(og.content_filter),
            inp="(không có câu trả lời để lọc)", out="(không chạy)",
            branches=[
                _br("Có PII / secret → [REDACTED]", False, "Trả cho người dùng (đã che)"),
                _br("Sạch → giữ nguyên", False, "Trả cho người dùng"),
            ]))

    return {
        "message": msg,
        "steps": steps,
        "reply": run["reply"],
        "blocked": run["blocked"],
        "layer": run["layer"],
        "error": run["error"],
        "total_ms": round((run["finished"] - run["started"]) * 1000),
        "rate_limit": run["rate_limit"],
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def read_outputs() -> dict:
    out = {}
    for name in DATA_FILES:
        path = ROOT / "outputs" / f"{name}.json"
        try:
            out[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out[name] = None
    return out


PAGES = {"/": "index.html", "/index.html": "index.html",
         "/trace": "trace.html", "/trace.html": "trace.html"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, obj) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        if self.path in PAGES:
            html = (Path(__file__).parent / PAGES[self.path]).read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
        elif self.path == "/api/data":
            data = read_outputs()
            data["rate_limit_live"] = rate_limit_state()
            self._json(200, data)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path not in ("/api/chat", "/api/trace"):
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(min(length, 20_000)) or b"{}")
            message = str(payload.get("message", ""))
        except (ValueError, TypeError):
            return self._json(400, {"error": "invalid JSON"})
        if self.path == "/api/chat":
            return self._json(200, handle_chat(message))
        return self._json(200, build_trace(execute(message)))

    def log_message(self, fmt, *args):  # quieter console
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Lab 11 UI → http://{HOST}:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
