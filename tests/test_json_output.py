from __future__ import annotations

from xninja.cli import _ContentSplitter, _clean_command, _SENTINEL


def _run(splitter: _ContentSplitter, chunks: list[str]) -> list:
    out: list = []
    for chunk in chunks:
        out.extend(splitter.feed(chunk))
    out.extend(splitter.flush())
    return out


def _joined(segments: list, channel: str) -> str:
    return "".join(text for ch, text in segments if ch == channel)


def test_splitter_extracts_think_across_deltas():
    # <think> / </think> split mid-tag across deltas must not leak or drop text.
    segments = _run(_ContentSplitter(), ["Sure. <thi", "nk>plan it</thi", "nk>do", "ne"])
    assert _joined(segments, "reasoning") == "plan it"
    assert _joined(segments, "message") == "Sure. done"
    assert "<think" not in _joined(segments, "message")


def test_splitter_drops_fenced_code():
    # The fenced command renders as a tool row; it must not appear inline in the
    # streamed message. Fence markers are split across deltas here too.
    segments = _run(_ContentSplitter(), ["Do this:\n``", "`bash\necho hi\n``", "`\nDone."])
    msg = _joined(segments, "message")
    assert "Do this:" in msg and "Done." in msg
    assert "echo hi" not in msg and "```" not in msg


def test_splitter_plain_message_passthrough():
    segments = _run(_ContentSplitter(), ["hello ", "world"])
    assert _joined(segments, "message") == "hello world"


def test_clean_command_strips_submission_sentinel():
    assert _clean_command(f"echo 'hi' > f.txt && echo {_SENTINEL}") == "echo 'hi' > f.txt"
    assert _clean_command(f"echo {_SENTINEL}") == ""  # bare submission → no tool row
    assert _clean_command("ls -la") == "ls -la"
