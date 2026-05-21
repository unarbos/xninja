from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Literal, Sequence

from xninja import __version__
from xninja.agent import bundled_agent_source, resolve_agent_source, run_agent, update_cached_agent
from xninja.config import (
    XninjaConfig,
    config_path,
    config_with_env,
    load_config,
    redact_secret,
    save_config,
)
from xninja.models import OPENROUTER_API_BASE, RECOMMENDED_MODELS, resolve_model
from xninja.patches import apply_patch, patch_summary, patch_text, repo_is_git_worktree
from xninja.permissions import apply_patch_allowed, remember_apply_patch


ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
}


ColorMode = Literal["auto", "always", "never"]


def color_enabled(mode: ColorMode | None = None) -> bool:
    selected = mode or os.environ.get("XNINJA_COLOR", "auto")
    if selected == "always":
        return True
    if selected == "never":
        return False
    return os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"


def style(text: str, *names: str) -> str:
    if not color_enabled():
        return text
    prefix = "".join(ANSI_CODES[name] for name in names if name in ANSI_CODES)
    return f"{prefix}{text}{ANSI_CODES['reset']}" if prefix else text


def meta(name: str, value: object) -> str:
    return f"{style(name + ':', 'bold', 'cyan')} {style(str(value), 'magenta')}"


def brand(text: str = "xninja") -> str:
    return style(text, "bold", "magenta")


def section(title: str) -> None:
    rendered = brand(title) if title == "xninja" else style(title, "bold", "cyan")
    print("\n" + rendered)


def transcript_line(label: str, value: object) -> str:
    return f"{style(label + ':', 'bold', 'cyan')} {style(str(value), 'magenta')}"


def fmt_elapsed_compact(elapsed_secs: int) -> str:
    if elapsed_secs < 60:
        return f"{elapsed_secs}s"
    if elapsed_secs < 3600:
        minutes, seconds = divmod(elapsed_secs, 60)
        return f"{minutes}m {seconds:02}s"
    hours, remainder = divmod(elapsed_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02}m {seconds:02}s"


def working_status(elapsed_secs: int | None = None, interrupt_hint: str = "Ctrl-C") -> str:
    details = f"{interrupt_hint} to interrupt"
    if elapsed_secs is not None:
        details = f"{fmt_elapsed_compact(elapsed_secs)} • {details}"
    return f"{style('Working', 'bold', 'cyan')} {style(f'({details})', 'magenta')}"


def colorize_patch_line(line: str) -> str:
    if line.startswith("@@"):
        return style(line, "bold", "cyan")
    if line.startswith("+++"):
        return style(line, "bold", "green")
    if line.startswith("---"):
        return style(line, "bold", "red")
    if line.startswith("+"):
        return style(line, "green")
    if line.startswith("-"):
        return style(line, "red")
    if line.startswith("diff --git"):
        return style(line, "bold", "magenta")
    if line.startswith("index "):
        return style(line, "dim")
    return line


def colorize_patch(text: str) -> str:
    return "\n".join(colorize_patch_line(line) for line in text.splitlines())


def footer_hint(*parts: str) -> str:
    return style(" · ".join(part for part in parts if part), "cyan")


def info(text: str) -> None:
    print(style(text, "cyan"))


def warn(text: str) -> None:
    print(style(text, "red"))


def success(text: str) -> None:
    print(style(text, "green"))


def error(text: str) -> None:
    print(style(text, "red"), file=sys.stderr)


COMMAND_NAMES = {"run", "exec", "e", "config", "models", "agent"}
GLOBAL_OPTIONS_WITH_VALUE = {"--repo", "-C", "--cd", "--model", "-m", "--color", "--agent-ref"}


def add_run_options(parser: argparse.ArgumentParser, repo_help: str) -> None:
    parser.add_argument("--repo", "-C", "--cd", default=".", help=repo_help)
    parser.add_argument("--model", "-m", help="OpenRouter model id")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="when to use ANSI color")
    parser.add_argument("--agent-ref", help="unarbos/ninja ref to fetch/use instead of bundled agent")
    parser.add_argument("--apply", action="store_true", help="apply the returned patch after preview")
    parser.add_argument("--raw-logs", action="store_true", help="show raw agent logs instead of the rendered transcript")
    parser.add_argument("--verbose", action="store_true", help="alias for --raw-logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xninja", description="Run the ninja coding agent locally.")
    parser.add_argument("--version", action="version", version=f"xninja {__version__}")
    add_run_options(parser, "repository path for the task")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", aliases=("exec", "e"), help="run one task")
    run_parser.add_argument("task", nargs="+", help="task prompt")
    add_run_options(run_parser, "repository path")

    config_parser = subparsers.add_parser("config", help="configure OpenRouter and defaults")
    config_parser.add_argument("--show", action="store_true", help="show current config with secrets redacted")
    config_parser.add_argument("--model", help="set default model without prompting")
    config_parser.add_argument("--api-key", help="set OpenRouter API key without prompting")

    subparsers.add_parser("models", help="list recommended OpenRouter models")

    agent_parser = subparsers.add_parser("agent", help="inspect or update the bundled ninja agent")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", required=True)
    agent_subparsers.add_parser("info", help="show bundled agent metadata")
    update_parser = agent_subparsers.add_parser("update", help="cache an agent.py from unarbos/ninja")
    update_parser.add_argument("--ref", required=True, help="branch, tag, or commit")

    return parser


def build_prompt_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xninja", description="Run the ninja coding agent locally.")
    parser.add_argument("--version", action="version", version=f"xninja {__version__}")
    add_run_options(parser, "repository path for the task")
    parser.add_argument("prompt", nargs="*", help="one-shot task prompt")
    return parser


def first_positional_arg(argv: Sequence[str]) -> str | None:
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in GLOBAL_OPTIONS_WITH_VALUE:
            skip_next = True
            continue
        if arg.startswith("--repo=") or arg.startswith("--cd=") or arg.startswith("--model=") or arg.startswith("--color=") or arg.startswith("--agent-ref="):
            continue
        if arg.startswith("-"):
            continue
        return arg
    return None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    first = first_positional_arg(argv)
    if first is not None and first not in COMMAND_NAMES:
        args = build_prompt_parser().parse_args(argv)
        args.command = None
        return args
    return build_parser().parse_args(argv)


def print_config(config: XninjaConfig) -> None:
    print(meta("config", config_path()))
    print(meta("openrouter_api_key", redact_secret(config.openrouter_api_key)))
    print(meta("default_model", config.default_model))
    print(meta("allow_apply_patch", config.allow_apply_patch))
    allowed = ", ".join(config.allowed_shell_commands) or "(none)"
    print(meta("allowed_shell_commands", allowed))


def configure(args: argparse.Namespace) -> int:
    current = load_config()
    if args.show:
        print_config(config_with_env(current))
        return 0
    api_key = args.api_key or getpass.getpass("OpenRouter API key: ").strip()
    default_model = args.model or input(f"Default model [{current.default_model}]: ").strip()
    updated = replace(
        current,
        openrouter_api_key=api_key or current.openrouter_api_key,
        default_model=default_model or current.default_model,
    )
    path = save_config(updated)
    success(f"Saved config to {path}")
    print_config(config_with_env(updated))
    return 0


def list_models(config: XninjaConfig) -> int:
    active = config_with_env(config).default_model
    for choice in RECOMMENDED_MODELS:
        marker = "*" if choice.model_id == active else " "
        print(f"{style(marker, 'green')} {style(choice.model_id, 'bold', 'magenta')} {style('-', 'cyan')} {style(choice.label + ':', 'cyan')} {style(choice.note, 'dim')}")
    return 0


def prompt_apply(config: XninjaConfig) -> tuple[bool, XninjaConfig]:
    if apply_patch_allowed(config):
        return True, config
    info("Review the patch above carefully. Only apply it if it is useful.")
    answer = input(f"{style('Apply patch?', 'bold', 'cyan')} {style('[y/N/always]', 'magenta')} ").strip().lower()
    if answer == "always":
        updated = remember_apply_patch(config)
        save_config(updated)
        return True, updated
    return answer in {"y", "yes"}, config


def copy_repo_for_agent(repo_path: Path, root: Path) -> Path:
    work_repo = root / repo_path.name
    shutil.copytree(
        repo_path,
        work_repo,
        ignore=shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", "build", "dist"),
        symlinks=True,
    )
    return work_repo


def run_git(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        text=True,
        capture_output=True,
        check=False,
    )


def commit_agent_baseline(repo_path: Path) -> None:
    run_git(repo_path, ["config", "user.email", "xninja@example.invalid"])
    run_git(repo_path, ["config", "user.name", "xninja"])
    run_git(repo_path, ["add", "-A"])
    result = run_git(repo_path, ["commit", "--allow-empty", "-m", "xninja baseline"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "failed to create xninja baseline commit")


def stream_agent_logs_enabled(source: object) -> bool:
    return source is not None and "bundled_agent" in str(getattr(source, "path", ""))


def printable_agent_logs(logs: object) -> str:
    if logs is None:
        return ""
    return str(logs).strip()


def run_task(
    repo: Path,
    task: str,
    explicit_model: str | None,
    agent_ref: str | None,
    apply_requested: bool,
    raw_logs: bool = False,
    color: ColorMode = "auto",
) -> int:
    repo_path = repo.expanduser().resolve()
    if not repo_path.exists():
        error(f"Repo path does not exist: {repo_path}")
        return 2
    if not repo_is_git_worktree(repo_path):
        error(f"Repo path is not a git worktree: {repo_path}")
        return 2

    stored_config = load_config()
    config = config_with_env(stored_config)
    api_key = config.openrouter_api_key
    if not api_key:
        error("OpenRouter API key is not configured. Run `xninja config` first.")
        return 2

    model = resolve_model(explicit_model, os.environ.get("XNINJA_MODEL"), stored_config.default_model)
    source = resolve_agent_source(agent_ref)
    previous_color = os.environ.get("XNINJA_COLOR")
    os.environ["XNINJA_COLOR"] = color
    section("xninja")
    print(transcript_line("user", task))
    print(transcript_line("model", model))
    print(transcript_line("cwd", repo_path))
    stream_logs = stream_agent_logs_enabled(source)
    print(working_status())

    previous_stream_setting = os.environ.get("XNINJA_STREAM_LOGS")
    previous_model_stream_setting = os.environ.get("XNINJA_STREAM_MODEL")
    if stream_logs:
        os.environ["XNINJA_STREAM_LOGS"] = "raw" if raw_logs else "rendered"
        if raw_logs:
            os.environ["XNINJA_STREAM_MODEL"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="xninja-agent-") as work_root:
            work_repo = copy_repo_for_agent(repo_path, Path(work_root))
            commit_agent_baseline(work_repo)
            result = run_agent(source, work_repo, task, model, OPENROUTER_API_BASE, api_key)
    finally:
        if stream_logs:
            if previous_stream_setting is None:
                os.environ.pop("XNINJA_STREAM_LOGS", None)
            else:
                os.environ["XNINJA_STREAM_LOGS"] = previous_stream_setting
            if previous_model_stream_setting is None:
                os.environ.pop("XNINJA_STREAM_MODEL", None)
            else:
                os.environ["XNINJA_STREAM_MODEL"] = previous_model_stream_setting
        if previous_color is None:
            os.environ.pop("XNINJA_COLOR", None)
        else:
            os.environ["XNINJA_COLOR"] = previous_color
    patch = patch_text(result)
    logs = printable_agent_logs(result.get("logs"))
    if logs and not stream_logs:
        section("Thinking Trace")
        print(logs)
    if not patch.strip():
        warn("\nAgent returned no patch.")
        return 1

    section("Patch Preview")
    print(colorize_patch(patch_summary(patch)))

    should_apply = apply_requested
    if not should_apply:
        should_apply, _ = prompt_apply(stored_config)
    if not should_apply:
        info("Patch left unapplied.")
        return 0

    applied = apply_patch(repo_path, patch)
    if applied.returncode != 0:
        error("Patch did not apply cleanly. No files were changed by xninja.")
        print(applied.stdout, end="")
        print(applied.stderr, end="", file=sys.stderr)
        return applied.returncode
    success("Patch applied.")
    return 0


def interactive(args: argparse.Namespace) -> int:
    previous_color = os.environ.get("XNINJA_COLOR")
    os.environ["XNINJA_COLOR"] = args.color
    try:
        section("xninja")
        print(transcript_line("cwd", Path(args.repo).expanduser().resolve()))
        print(footer_hint("enter to send", "Ctrl-D to exit", "--help for options"))
        while True:
            try:
                task = input(style("xninja> ", "bold", "magenta")).strip()
            except EOFError:
                print()
                return 0
            if not task:
                continue
            code = run_task(Path(args.repo), task, args.model, args.agent_ref, args.apply, args.raw_logs or args.verbose, args.color)
            if code not in {0, 1}:
                return code
    finally:
        if previous_color is None:
            os.environ.pop("XNINJA_COLOR", None)
        else:
            os.environ["XNINJA_COLOR"] = previous_color


def agent_info() -> int:
    source = bundled_agent_source()
    print(meta("path", source.path))
    for key in ("source_repo", "ref", "commit", "path"):
        print(meta(key, source.metadata.get(key, '')))
    return 0


def agent_update(args: argparse.Namespace) -> int:
    source = update_cached_agent(args.ref)
    success(f"cached: {source.path}")
    for key, value in source.metadata.items():
        print(meta(key, value))
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "config":
        return configure(args)
    if args.command == "models":
        return list_models(load_config())
    if args.command == "agent":
        if args.agent_command == "info":
            return agent_info()
        if args.agent_command == "update":
            return agent_update(args)
    if args.command in {"run", "exec", "e"}:
        return run_task(Path(args.repo), " ".join(args.task), args.model, args.agent_ref, args.apply, args.raw_logs or args.verbose, args.color)
    prompt = getattr(args, "prompt", [])
    if prompt:
        return run_task(Path(args.repo), " ".join(prompt), args.model, args.agent_ref, args.apply, args.raw_logs or args.verbose, args.color)
    return interactive(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(tuple(sys.argv[1:] if argv is None else argv))
    try:
        return dispatch(args)
    except KeyboardInterrupt:
        error("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
