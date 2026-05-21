from __future__ import annotations

from xninja.bundled_agent.agent import _render_log_item, _stream_delta_text


def test_render_model_response_as_transcript_lines():
    item = """MODEL_RESPONSE:
<plan>Check README and edit it.</plan>
<edit path=\"README.md\" op=\"replace\"><old>a</old><new>b</new></edit>
<command>cat README.md</command>
<final>Done</final>
"""

    assert _render_log_item(item) == [
        "thinking: Check README and edit it.",
        "edit: replace README.md",
        "tools: cat README.md",
        "final: Done",
    ]


def test_render_observation_as_result_lines():
    item = """OBSERVATION 1/1:
COMMAND:
cat README.md

EXIT_CODE:
0

DURATION_SECONDS:
0.001

STDOUT:
hello
"""

    assert _render_log_item(item) == []


def test_render_wait_and_step_lines():
    assert _render_log_item("\n\n===== STEP 2 =====\n") == ["", "Step 2"]
    assert _render_log_item("MODEL_WAIT: step=1 attempt=1 waited=30s") == ["waiting: 30s"]


def test_render_edit_success_keeps_short_result():
    item = """OBSERVATION 1/1:
COMMAND:
<edit path='README.md' op='replace'>

EXIT_CODE:
0

DURATION_SECONDS:
0.001

STDOUT:
Replaced 1 occurrence in README.md
"""

    assert _render_log_item(item) == [
        "edited: <edit path='README.md' op='replace'> (exit 0)",
        "  Replaced 1 occurrence in README.md",
    ]


def test_render_failure_shows_output():
    item = """OBSERVATION 1/1:
COMMAND:
pytest

EXIT_CODE:
1

DURATION_SECONDS:
0.001

STDOUT:
FAILED test_example.py::test_nope
"""

    assert _render_log_item(item) == [
        "tested: pytest (exit 1)",
        "  FAILED test_example.py::test_nope",
    ]


def test_stream_delta_text_reads_reasoning_and_content():
    chunk = {"choices": [{"delta": {"reasoning": "think ", "content": "say"}}]}

    assert _stream_delta_text(chunk) == "think say"


def test_stream_delta_text_handles_empty_shape():
    assert _stream_delta_text({"choices": [{"delta": {}}]}) == ""
    assert _stream_delta_text({}) == ""


def test_heartbeat_disabled_during_model_stream(monkeypatch):
    from xninja.bundled_agent.agent import _start_model_wait_heartbeat

    monkeypatch.setenv("XNINJA_STREAM_LOGS", "rendered")
    monkeypatch.setenv("XNINJA_STREAM_MODEL", "1")

    assert _start_model_wait_heartbeat([], 1, 1) is None


def test_render_model_response_summarizes_many_tools():
    item = """MODEL_RESPONSE:
<command>cat a.py</command>
<command>cat b.py</command>
<command>cat c.py</command>
<command>cat d.py</command>
<command>cat e.py</command>
"""

    assert _render_log_item(item) == [
        "tools: cat a.py; cat b.py; cat c.py; ... plus 2 more",
    ]


def test_rendered_stream_lines_can_be_colored(monkeypatch):
    monkeypatch.setenv("XNINJA_COLOR", "always")

    assert "\033[" in _render_log_item("\n\n===== STEP 2 =====\n")[1]
    assert "\033[" in _render_log_item("MODEL_RESPONSE:\n<final>Done</final>")[0]
