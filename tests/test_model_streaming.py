from __future__ import annotations

import importlib.util
from pathlib import Path

# model.py is stdlib-only; load it directly (the bundle's agent.py / agent/ split
# isn't importable as a normal package outside load_agent_module).
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src" / "xninja" / "bundled_agent" / "agent" / "model.py"
)


def _load_model_module():
    spec = importlib.util.spec_from_file_location("xninja_bundled_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_mod = _load_model_module()
ChatModel = model_mod.ChatModel


class FakeResponse:
    """Minimal stand-in for the urlopen() context manager: iterates byte lines."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return b"".join(self._lines)


def _model():
    return ChatModel(model_name="m", base_url="http://x/v1", auth_token="t")


def _patch_urlopen(monkeypatch, lines):
    monkeypatch.setattr(
        model_mod.urllib.request, "urlopen", lambda *a, **k: FakeResponse(lines)
    )


def test_stream_emits_content_deltas_and_returns_full_text(monkeypatch):
    _patch_urlopen(monkeypatch, [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n',
        b"data: [DONE]\n",
    ])
    got = []
    text = _model().query([{"role": "user", "content": "hi"}], on_delta=got.append)
    assert got == ["Hello", " world"]
    assert text == "Hello world"


def test_stream_shows_reasoning_but_excludes_it_from_parsed_text(monkeypatch):
    # The agent loop parses the returned text for its bash block, so reasoning
    # must be displayed (streamed) yet never folded into that text.
    _patch_urlopen(monkeypatch, [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"```bash\\necho hi\\n```"}}]}\n',
        b"data: [DONE]\n",
    ])
    got = []
    text = _model().query([{"role": "user", "content": "hi"}], on_delta=got.append)
    assert got == ["thinking", "```bash\necho hi\n```"]
    assert text == "```bash\necho hi\n```"


def test_non_sse_response_falls_back_to_whole_completion(monkeypatch):
    # Upstream ignored stream=true and returned one JSON object: still works.
    _patch_urlopen(monkeypatch, [b'{"choices":[{"message":{"content":"whole answer"}}]}'])
    got = []
    text = _model().query([{"role": "user", "content": "hi"}], on_delta=got.append)
    assert text == "whole answer"
    assert got == ["whole answer"]


def test_non_streaming_query_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        model_mod.ChatModel,
        "_post",
        lambda self, body: '{"choices":[{"message":{"content":"plain"}}]}',
    )
    text = _model().query([{"role": "user", "content": "hi"}])
    assert text == "plain"
