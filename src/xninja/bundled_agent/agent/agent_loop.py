from __future__ import annotations
import re
import time
from dataclasses import dataclass, field

from .environment import execute_command, truncate_text
from .model import ChatModel, ModelQueryError
from .prompts import (
    COMPLETION_SENTINEL,
    SYSTEM_PROMPT,
    build_task_prompt,
    format_help_message,
    render_observation,
)
from .repo_diff import collect_repo_patch

_ACTION_BLOCK_RE = re.compile(r"```(?:bash|sh)?\s*\n(.*?)\n?```", re.DOTALL)
_MAX_FORMAT_RETRIES = 3


@dataclass
class AgentRunConfig:
    repo_dir: str
    model_name: str
    base_url: str
    auth_token: str
    max_steps: int = 50
    command_timeout: int = 15
    max_tokens: int = 8192
    max_observation_chars: int = 16000
    max_log_chars: int = 260000
    wall_clock_limit: float = 0.0


@dataclass
class AgentOutcome:
    success: bool
    patch: str
    logs: str
    steps: int
    cost: float | None
    message: str
    exit_status: str = "Submitted"
    transcript: list = field(default_factory=list)


def run_agent_loop(*, config: AgentRunConfig, task: str, on_event=None) -> AgentOutcome:
    """Run the query -> act -> observe loop.

    When ``on_event`` is given it is called with small dict events as work
    happens — ``{"type": "step"|"token"|"result"|"notice", ...}`` — so a CLI can
    render the agent's progress live instead of only seeing the buffered logs at
    the end. The event stream is display-only; it never changes the outcome.
    """
    model = ChatModel(
        model_name=config.model_name,
        base_url=config.base_url,
        auth_token=config.auth_token,
        max_completion_tokens=config.max_tokens,
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task if "<task>" in task else build_task_prompt(task_text=task)},
    ]
    started = time.monotonic()
    log_lines: list = []
    exit_status = "LimitsExceeded"
    message = f"step limit of {config.max_steps} reached"
    format_retries = 0
    on_delta = (lambda piece: on_event({"type": "token", "text": piece})) if on_event else None

    for step in range(1, max(1, config.max_steps) + 1):
        if 0 < config.wall_clock_limit <= time.monotonic() - started:
            exit_status = "TimeExceeded"
            message = f"wall clock limit of {config.wall_clock_limit:.0f}s reached"
            break
        if on_event:
            on_event({"type": "step", "n": step})
        try:
            reply = model.query(messages, on_delta=on_delta)
        except ModelQueryError as exc:
            exit_status = "ModelError"
            message = str(exc)
            log_lines.append(f"[step {step}] model error: {exc}")
            if on_event:
                on_event({"type": "notice", "text": f"model error: {exc}"})
            break
        messages.append({"role": "assistant", "content": reply})
        log_lines.append(f"[step {step}] assistant:\n{reply}")

        actions = _ACTION_BLOCK_RE.findall(reply)
        commands = [action.strip() for action in actions if action.strip()]
        if len(commands) != 1:
            format_retries += 1
            if format_retries > _MAX_FORMAT_RETRIES:
                exit_status = "FormatError"
                message = "model kept replying without exactly one bash code block"
                break
            messages.append({"role": "user", "content": format_help_message()})
            log_lines.append(f"[step {step}] format retry {format_retries}")
            if on_event:
                on_event({"type": "notice", "text": "reformatting — need exactly one command block"})
            continue
        format_retries = 0
        command = commands[0]

        result = execute_command(command, cwd=config.repo_dir, timeout=config.command_timeout)
        output_text = result.get("output") or ""
        log_lines.append(f"[step {step}] $ {command}\n{truncate_text(output_text, 2000)}")
        if on_event:
            on_event({"type": "result", "output": truncate_text(output_text, 2000)})
        if _is_submission(output_text, result.get("returncode")):
            exit_status = "Submitted"
            message = f"submitted after {step} step(s)"
            break
        observation = render_observation(
            returncode=int(result.get("returncode") or 0),
            output_text=truncate_text(output_text, config.max_observation_chars),
            remaining_steps=config.max_steps - step,
        )
        messages.append({"role": "user", "content": observation})

    patch = collect_repo_patch(config.repo_dir)
    logs = truncate_text("\n".join(log_lines), config.max_log_chars)
    return AgentOutcome(
        success=bool(patch.strip()),
        patch=patch,
        logs=logs,
        steps=model.calls,
        cost=None,
        message=message,
        exit_status=exit_status,
        transcript=messages,
    )


def _is_submission(output_text: str, returncode) -> bool:
    lines = output_text.lstrip().splitlines()
    return bool(lines) and lines[0].strip() == COMPLETION_SENTINEL and not returncode
