import json
import time
import urllib.error
import urllib.request

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ModelQueryError(RuntimeError):
    pass


class _TransientContentError(ModelQueryError):
    """A 200-OK reply that is unusable (no choices / no content / empty)."""
    pass


class ChatModel:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        auth_token: str,
        max_completion_tokens: int = 0,
        request_timeout: float = 180.0,
        max_attempts: int = 5,
    ) -> None:
        self.model_name = model_name
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.auth_token = auth_token
        self.max_completion_tokens = int(max_completion_tokens or 0)
        self.request_timeout = request_timeout
        self.max_attempts = max(1, int(max_attempts))
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def query(self, messages: list) -> str:
        """Send the conversation and return the assistant message text."""
        payload = {"model": self.model_name, "messages": messages}
        if self.max_completion_tokens > 0:
            payload["max_tokens"] = self.max_completion_tokens
        body = json.dumps(payload).encode("utf-8")
        last_error = "unknown error"
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self._post(body)
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail[:300]}"
                if exc.code not in _RETRYABLE_STATUS:
                    raise ModelQueryError(f"model request was rejected: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                try:
                    text = self._extract_content(raw)
                except _TransientContentError as exc:
                    # 200-OK but unusable (Google soft-empty/finish_reason=error):
                    # fall through to the existing backoff and retry in-place
                    # rather than forfeit the round.
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    self.calls += 1
                    return text
            if attempt < self.max_attempts:
                time.sleep(min(20.0, 1.5 ** attempt))
        raise ModelQueryError(f"model request failed after {self.max_attempts} attempts: {last_error}")

    def _post(self, body: bytes) -> str:
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def _extract_content(self, raw: str) -> str:
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise ModelQueryError(f"model returned invalid JSON: {raw[:300]}") from exc
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            self.prompt_tokens += _as_int(usage.get("prompt_tokens"))
            self.completion_tokens += _as_int(usage.get("completion_tokens"))
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise _TransientContentError(f"model response has no choices: {raw[:300]}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(
                str(part.get("text") or "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise _TransientContentError(f"model response has no text content: {raw[:300]}")
        if not content.strip():
            raise _TransientContentError(f"model returned empty content: {raw[:200]}")
        return content


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return str(exc)


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
