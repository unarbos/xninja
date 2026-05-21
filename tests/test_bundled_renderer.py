from __future__ import annotations

from xninja.bundled_agent.agent import _render_log_item


def test_render_model_response_as_transcript_lines():
    item = """MODEL_RESPONSE:
<plan>Check README and edit it.</plan>
<edit path=\"README.md\" op=\"replace\"><old>a</old><new>b</new></edit>
<command>cat README.md</command>
<final>Done</final>
"""

    assert _render_log_item(item) == [
        "Plan:",
        "  Check README and edit it.",
        "Edit: replace README.md",
        "Tool: cat README.md",
        "Final: Done",
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

    assert _render_log_item(item) == ["Result: exit 0 from cat README.md", "  hello"]


def test_render_wait_and_step_lines():
    assert _render_log_item("\n\n===== STEP 2 =====\n") == ["Step 2"]
    assert _render_log_item("MODEL_WAIT: step=1 attempt=1 waited=5s") == [
        "Waiting for model: step=1 attempt=1 waited=5s"
    ]
