"""
Assignment 11 — Rate Limiter starter (TODO).

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    PRUNE_THRESHOLD = 256  # only sweep when many users are tracked

    def _prune(self, now: float) -> None:
        """Forget users with no request inside the window, so memory does not grow
        without bound when an attacker rotates user ids."""
        if len(self.user_windows) <= self.PRUNE_THRESHOLD:
            return
        cutoff = now - self.window_seconds
        for uid in [u for u, w in self.user_windows.items() if not w or w[-1] <= cutoff]:
            del self.user_windows[uid]

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        self._prune(now)
        window = self.user_windows[user_id]

        # 1. Drop timestamps that fell out of the sliding window
        while window and window[0] <= now - self.window_seconds:
            window.popleft()

        # 2. Over the limit -> block (no LLM call)
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            self.blocked_count += 1
            return self._block_response(
                f"Rate limit exceeded. Try again in {wait:.0f}s."
            )

        # 3. Within limit -> record and let it through
        window.append(now)
        return None
