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

    def query(self, messages: list, on_delta=None) -> str:
        """Send the conversation and return the assistant message text.

        When ``on_delta`` is given, stream the completion (SSE) and call
        ``on_delta(piece)`` for each reasoning/content delta as it arrives, so a
        caller can render the model's output live instead of waiting for the full
        reply. Without it, behaviour is the original single-shot request.
        """
        streaming = on_delta is not None
        payload = {"model": self.model_name, "messages": messages}
        if self.max_completion_tokens > 0:
            payload["max_tokens"] = self.max_completion_tokens
        if streaming:
            payload["stream"] = True
        body = json.dumps(payload).encode("utf-8")
        last_error = "unknown error"
        emitted = 0

        def _emit(piece: str) -> None:
            nonlocal emitted
            emitted += 1
            on_delta(piece)

        for attempt in range(1, self.max_attempts + 1):
            try:
                if streaming:
                    text = self._post_stream(body, _emit)
                else:
                    text = self._extract_content(self._post(body))
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc)
                last_error = f"HTTP {exc.code}: {detail[:300]}"
                if exc.code not in _RETRYABLE_STATUS:
                    raise ModelQueryError(f"model request was rejected: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # Once a stream has printed tokens, a silent retry would duplicate
                # the visible output — end the round on the error instead.
                if emitted:
                    raise ModelQueryError(
                        f"stream interrupted after {emitted} chunk(s): {last_error}"
                    ) from exc
            except _TransientContentError as exc:
                # 200-OK but unusable (Google soft-empty/finish_reason=error):
                # fall through to the existing backoff and retry in-place rather
                # than forfeit the round — unless we already streamed partial text.
                last_error = f"{type(exc).__name__}: {exc}"
                if emitted:
                    raise ModelQueryError(f"stream produced unusable content: {last_error}") from exc
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

    def _post_stream(self, body: bytes, on_delta) -> str:
        """POST with stream=true, emit each SSE delta via on_delta, return the
        full assistant *content* (reasoning is streamed for display only, never
        folded into the text the agent loop parses for its command block).

        If the upstream ignores ``stream`` and returns a single JSON completion,
        fall back to extracting it whole — so a non-streaming provider still works.
        """
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.auth_token}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        parts: list = []
        buffered: list = []
        saw_sse = False
        with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    if not saw_sse:
                        buffered.append(line)  # may be a non-SSE JSON body
                    continue
                saw_sse = True
                data = stripped[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    self.prompt_tokens += _as_int(usage.get("prompt_tokens"))
                    self.completion_tokens += _as_int(usage.get("completion_tokens"))
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                if not isinstance(delta, dict):
                    continue
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    on_delta(reasoning)
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    parts.append(piece)
                    on_delta(piece)
        if not saw_sse:
            text = self._extract_content("\n".join(buffered))
            on_delta(text)
            return text
        text = "".join(parts)
        if not text.strip():
            raise _TransientContentError("stream produced empty content")
        return text

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
